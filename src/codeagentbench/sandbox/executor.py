"""Bash tool adapter with timeouts, bounded output and audit-friendly receipts."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from codeagentbench.harness.recovery import ActionJournal
from codeagentbench.models import ToolIntent, ToolReceipt
from codeagentbench.sandbox.workspace import workspace_digest


class BashExecutor:
    """Execute the same bash-style interface used by mini-swe-agent.

    The legacy local backend checks the starting cwd and filters credential-like
    variables, but does not isolate the command. The Linux bwrap backend mounts
    only a per-run checkout and read-only system binaries, clears the child
    environment, and disables network access. Use bwrap for untrusted commands.
    """

    def __init__(self, workspace: str | Path, journal: ActionJournal | None = None, output_limit: int = 20_000, *, backend: str = "local") -> None:
        if backend not in {"local", "bwrap"}:
            raise ValueError(f"unknown executor backend: {backend}")
        self.workspace = Path(workspace).resolve()
        self.journal = journal
        self.output_limit = output_limit
        self.backend = backend
        self.bash = shutil.which("bash") or shutil.which("wsl.exe")
        self.bwrap = shutil.which("bwrap") if backend == "bwrap" else None
        if backend == "bwrap" and (os.name == "nt" or not self.bwrap):
            raise RuntimeError("bwrap executor requires bubblewrap on Linux")

    def execute(self, intent: ToolIntent, *, intent_already_recorded: bool = False) -> ToolReceipt:
        cwd = Path(intent.cwd).resolve()
        if cwd != self.workspace and self.workspace not in cwd.parents:
            raise ValueError("tool cwd must be inside the run workspace")
        if self.journal and not intent_already_recorded:
            self.journal.record_intent(intent)
        started = time.monotonic()
        env = {"PATH": "/usr/bin:/bin"} if self.backend == "bwrap" else self._tool_environment()
        env["CI"] = "1"
        command_text, temporary_script = self._normalize_command(intent.command)
        use_wsl = bool(self.bash and self._is_wsl_bash() and os.name != "nt")
        use_native_bash = bool(self.bash and not self._is_wsl_bash())
        if use_wsl:
            # System32\bash.exe is WSL on Windows. On a Linux host, convert
            # the path only when a WSL-style bash is actually executable.
            wsl_path = self._wsl_path(cwd)
            command_text = re.sub(r"(?<![\w.-])python(?=\s)", "python.exe", command_text)
            command_text = f"cd {shlex.quote(wsl_path)} && {command_text}"
        # Native bash does not need a login profile: on hosted workspaces it
        # can print a platform banner into every tool result, wasting context.
        # Keep WSL's existing login behavior for its path/environment setup.
        if self.backend == "bwrap":
            command = self._bwrap_command(command_text, cwd)
        elif use_native_bash:
            command = [self.bash, "-c", command_text]
        elif use_wsl:
            command = [self.bash, "-lc", command_text]
        else:
            command = command_text
        try:
            try:
                result = subprocess.run(command, cwd=cwd, env=env, shell=not (self.backend == "bwrap" or use_wsl or use_native_bash), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=intent.timeout_seconds, check=False)
                stdout, stderr = result.stdout, result.stderr
                receipt = ToolReceipt(intent.action_id, intent.command, result.returncode, stdout[: self.output_limit], stderr[: self.output_limit], time.monotonic() - started, False, workspace_digest(self.workspace))
            except subprocess.TimeoutExpired as exc:
                receipt = ToolReceipt(intent.action_id, intent.command, None, str(exc.stdout or "")[: self.output_limit], str(exc.stderr or "")[: self.output_limit], time.monotonic() - started, True, workspace_digest(self.workspace), "timeout")
        finally:
            if temporary_script:
                try:
                    Path(temporary_script).unlink()
                except OSError:
                    pass
        if self.journal:
            self.journal.record_receipt(receipt)
        return receipt

    def _bwrap_command(self, command_text: str, cwd: Path) -> list[str]:
        """Expose only system binaries and this run's checkout to an untrusted shell."""

        assert self.bwrap is not None
        relative = cwd.relative_to(self.workspace)
        sandbox_cwd = "/workspace" if relative == Path(".") else "/workspace/" + relative.as_posix()
        command = [
            self.bwrap,
            "--unshare-user", "--unshare-pid", "--unshare-net",
            "--die-with-parent", "--new-session", "--cap-drop", "ALL",
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64",
            "--dev", "/dev", "--tmpfs", "/tmp",
            "--bind", str(self.workspace), "/workspace",
        ]
        git_metadata = self.workspace / ".git"
        if git_metadata.exists():
            # The model may inspect Git state but must not install hooks or
            # change core.fsmonitor before the parent calls Git outside bwrap.
            command.extend(["--ro-bind", str(git_metadata), "/workspace/.git"])
        command.extend([
            "--chdir", sandbox_cwd,
            "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "HOME", "/tmp",
            "--setenv", "TMPDIR", "/tmp",
            "--setenv", "CI", "1",
            "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--", "/usr/bin/bash", "-c", command_text,
        ])
        return command

    @staticmethod
    def _normalize_command(command: str) -> tuple[str, str | None]:
        """Translate a common bash Python heredoc for the Windows fallback."""

        if os.name != "nt":
            return command, None
        match = re.fullmatch(
            r"python(?:\.exe)?\s+-\s+<<-?\s*(['\"]?)([A-Za-z_][\w-]*)\1\r?\n(.*?)\r?\n\s*\2\s*",
            command,
            flags=re.DOTALL,
        )
        if not match:
            return command, None
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".py", encoding="utf-8", delete=False)
        try:
            handle.write(match.group(3))
            script = handle.name
        finally:
            handle.close()
        return f'python "{script}"', script

    @staticmethod
    def _tool_environment() -> dict[str, str]:
        """Preserve runtime prerequisites while excluding obvious secrets.

        A tiny allow-list is insufficient on Windows: Python's asyncio and
        native extension loading rely on standard user/system variables that
        are not present in that list. The local runner therefore inherits the
        normal process environment except for credential-like names. A real
        container backend should still provide a stricter, explicit allow-list.
        """
        secret_markers = ("key", "token", "secret", "password", "credential")
        return {
            name: value
            for name, value in os.environ.items()
            if not any(marker in name.casefold() for marker in secret_markers)
        }

    def _is_wsl_bash(self) -> bool:
        return str(self.bash).lower().replace("/", "\\").endswith("\\system32\\bash.exe")

    @staticmethod
    def _wsl_path(path: Path) -> str:
        value = str(path)
        if len(value) >= 2 and value[1] == ":":
            return "/mnt/" + value[0].lower() + value[2:].replace("\\", "/")
        return value.replace("\\", "/")
