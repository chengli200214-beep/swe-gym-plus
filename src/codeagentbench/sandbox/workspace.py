"""Per-run workspaces with explicit repository and artifact boundaries."""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import Request, urlopen
from dataclasses import dataclass
from pathlib import Path

from codeagentbench.models import TaskRecord


def workspace_digest(path: str | Path) -> str:
    """Hash tracked diff, status and untracked file contents for recovery checks."""

    path = Path(path)
    pieces: list[bytes] = []
    if (path / ".git").exists():
        for command in (["git", "-c", "core.fsmonitor=false", "status", "--porcelain=v1"],
                        ["git", "-c", "core.fsmonitor=false", "diff", "--no-ext-diff", "--no-textconv", "--binary"]):
            result = subprocess.run(command, cwd=path, capture_output=True, check=False, timeout=30)
            pieces.append(result.stdout)
        # Tracked file contents are already represented by `git diff`. Hashing
        # the complete checkout on every checkpoint is quadratic in practice
        # for large SWE-Gym repositories. Only untracked files need a content
        # digest in addition to Git's status output.
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=path,
            capture_output=True,
            check=False,
            timeout=30,
        )
        untracked = [item for item in result.stdout.split(b"\0") if item]
        for encoded_relative in sorted(untracked):
            file = path / Path(encoded_relative.decode(errors="surrogateescape"))
            try:
                content = (b"<symlink>" + str(file.readlink()).encode("utf-8", errors="surrogateescape")) if file.is_symlink() else file.read_bytes()
            except OSError:
                content = b"<unreadable>"
            pieces.append(encoded_relative + b"\0" + hashlib.sha256(content).digest())
    else:
        for file in sorted(path.rglob("*")):
            if file.is_file() and ".git" not in file.parts:
                relative = file.relative_to(path).as_posix().encode("utf-8")
                try:
                    content = (b"<symlink>" + str(file.readlink()).encode("utf-8", errors="surrogateescape")) if file.is_symlink() else file.read_bytes()
                except OSError:
                    content = b"<unreadable>"
                pieces.append(relative + b"\0" + hashlib.sha256(content).digest())
    return hashlib.sha256(b"\0".join(pieces)).hexdigest()


@dataclass(frozen=True)
class Workspace:
    run_id: str
    path: Path
    base_commit: str
    image: str = "local"

    @property
    def digest(self) -> str:
        return workspace_digest(self.path)

    def diff(self) -> str:
        result = subprocess.run(["git", "-c", "core.fsmonitor=false", "diff", "--no-ext-diff", "--no-textconv", "--binary"], cwd=self.path, capture_output=True, text=True, check=False, timeout=30)
        return result.stdout

    def changed_files(self) -> tuple[str, ...]:
        result = subprocess.run(["git", "-c", "core.fsmonitor=false", "status", "--porcelain=v1"], cwd=self.path, capture_output=True, text=True, check=False, timeout=30)
        files: list[str] = []
        for line in result.stdout.splitlines():
            if len(line) > 3:
                files.append(line[3:].strip().strip('"'))
        return tuple(files)


class WorkspaceManager:
    """Create separate checkouts; command execution still needs OS isolation."""

    def __init__(self, root: str | Path, *, cache_root: str | Path | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_root = Path(cache_root) if cache_root is not None else self.root / ".repo_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def create(self, task: TaskRecord, run_id: str) -> Workspace:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", run_id):
            raise ValueError("invalid run_id")
        target = self.root / run_id / "workspace"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError(f"workspace already exists for run {run_id!r}; use a new run id")
        snapshot = self._get_or_build_snapshot(task)
        self._clone_snapshot(snapshot, target)
        return Workspace(run_id, target, task.base_commit)

    def _get_or_build_snapshot(self, task: TaskRecord) -> Path:
        """Materialize one immutable base snapshot and reuse it across runs.

        SWE smoke tasks often share a repository but use different commits. The
        old path downloaded and initialized a full Git repository for every
        candidate/control workspace. Caching the synthetic base repository
        keeps workspaces isolated while avoiding repeated ``git add`` over a
        large checkout.
        """

        key = hashlib.sha256(f"{task.repo}\0{task.base_commit}".encode("utf-8")).hexdigest()[:24]
        cached = self.cache_root / key
        if cached.is_dir() and (cached / ".git").is_dir():
            return cached
        if cached.exists():
            raise RuntimeError(f"incomplete repository cache at {cached}")

        staging_parent = Path(tempfile.mkdtemp(prefix=".building-", dir=self.cache_root))
        staging = staging_parent / "snapshot"
        try:
            source = Path(task.repo)
            if source.exists() and source.is_dir():
                if (source / ".git").exists():
                    self._archive_git_snapshot(source, task.base_commit, staging)
                else:
                    shutil.copytree(source, staging, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
                    self._init_local_repo(staging, "local")
            else:
                self._archive_remote_snapshot(task.repo, task.base_commit, staging)
            try:
                staging.replace(cached)
            except FileExistsError:
                # Another process may have completed the same immutable
                # snapshot while this one was building. The atomic rename
                # makes the winner safe to reuse.
                if cached.is_dir() and (cached / ".git").is_dir():
                    return cached
                raise
            return cached
        finally:
            shutil.rmtree(staging_parent, ignore_errors=True)

    @staticmethod
    def _clone_snapshot(snapshot: Path, target: Path) -> None:
        result = subprocess.run(
            # The snapshot is immutable and owned by this harness. Sharing
            # its object database keeps large SWE repositories cheap to clone;
            # each target still has an independent working tree and index.
            ["git", "clone", "--quiet", "--shared", str(snapshot), str(target)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if result.returncode:
            raise RuntimeError(f"could not clone cached snapshot: {result.stderr.strip()}")

    @staticmethod
    def _archive_git_snapshot(source: Path, base_commit: str, target: Path) -> None:
        revision = base_commit if base_commit not in {"", "local", "HEAD"} else "HEAD"
        archived = subprocess.run(["git", "archive", "--format=tar", revision], cwd=source, capture_output=True, check=False, timeout=120)
        if archived.returncode:
            raise RuntimeError(f"base commit {revision!r} unavailable: {archived.stderr.decode(errors='replace').strip()}")
        WorkspaceManager._extract_archive(archived.stdout, target)
        WorkspaceManager._init_local_repo(target, "local")

    @staticmethod
    def _archive_remote_snapshot(repo: str, base_commit: str, target: Path) -> None:
        if "://" not in repo and not repo.startswith("git@"):
            revision = base_commit if base_commit not in {"", "local", "HEAD"} else "HEAD"
            # Use GitHub's official archive host directly; some cloud networks
            # stall on github.com redirects even when codeload is reachable.
            archive_url = f"https://codeload.github.com/{repo}/tar.gz/{revision}"
            try:
                request = Request(archive_url, headers={"User-Agent": "CodeAgentBench/0.1"})
                with urlopen(request, timeout=120.0) as response:  # noqa: S310 - URL is built from a dataset repo id
                    WorkspaceManager._extract_github_archive(response.read(), target)
                WorkspaceManager._init_local_repo(target, "local")
                return
            except (OSError, tarfile.TarError) as exc:
                archive_error = str(exc)
        else:
            archive_error = ""
        remote = repo if "://" in repo or repo.startswith("git@") else f"https://github.com/{repo}.git"
        with tempfile.TemporaryDirectory(prefix="cab-fetch-") as temporary:
            stage = Path(temporary) / "repo"
            try:
                clone = subprocess.run(["git", "clone", "--quiet", remote, str(stage)], capture_output=True, text=True, timeout=180.0, check=False)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"timed out cloning {remote}") from exc
            if clone.returncode:
                detail = clone.stderr.strip() or archive_error
                raise RuntimeError(f"could not clone {remote}: {detail}")
            WorkspaceManager._archive_git_snapshot(stage, base_commit, target)

    @staticmethod
    def _extract_github_archive(data: bytes, target: Path) -> None:
        """Extract GitHub's single top-level directory as a safe snapshot."""

        target.mkdir(parents=True, exist_ok=False)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            members = archive.getmembers()
            prefix = members[0].name.split("/", 1)[0] if members else ""
            target_resolved = target.resolve()
            for member in members:
                relative_name = member.name[len(prefix):].lstrip("/") if prefix else member.name
                if not relative_name:
                    continue
                destination = (target / relative_name).resolve()
                if destination != target_resolved and target_resolved not in destination.parents:
                    raise RuntimeError("remote archive contains a path outside the workspace")
                if member.issym() or member.islnk():
                    raise RuntimeError("remote archive contains a link; refusing unsafe extraction")
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read())

    @staticmethod
    def _extract_archive(data: bytes, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=False)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            target_resolved = target.resolve()
            for member in archive.getmembers():
                destination = (target / member.name).resolve()
                if destination != target_resolved and target_resolved not in destination.parents:
                    raise RuntimeError("git archive contains a path outside the workspace")
            archive.extractall(target)

    @staticmethod
    def _init_local_repo(target: Path, base_commit: str) -> None:
        subprocess.run(["git", "init", "--quiet"], cwd=target, check=True, capture_output=True, timeout=30)
        subprocess.run(["git", "add", "-A"], cwd=target, check=True, capture_output=True, timeout=120)
        subprocess.run(
            ["git", "-c", "maintenance.auto=false", "-c", "gc.auto=0", "-c", "user.name=CodeAgentBench", "-c", "user.email=bench@localhost", "commit", "--quiet", "-m", "base"],
            cwd=target,
            check=True,
            capture_output=True,
            timeout=120,
        )
        if base_commit and base_commit not in {"HEAD", "local"}:
            WorkspaceManager._checkout(target, base_commit)

    @staticmethod
    def _checkout(target: Path, base_commit: str) -> None:
        result = subprocess.run(["git", "checkout", "--quiet", base_commit], cwd=target, capture_output=True, text=True, check=False, timeout=120)
        if result.returncode:
            raise RuntimeError(f"base commit {base_commit!r} unavailable: {result.stderr.strip()}")
