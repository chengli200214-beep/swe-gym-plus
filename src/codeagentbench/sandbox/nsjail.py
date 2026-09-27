"""Explicit no-namespace NsJail backend for the constrained AutoDL container.

See docs/adr/0004-autodl-no-namespace-executor.md. Provisioning is trusted;
only the supervised, unprivileged chroot executes task code.
"""
from __future__ import annotations

import json
import math
import os
import platform
import sqlite3
import subprocess
from pathlib import Path
from typing import Callable

from codeagentbench.sandbox.bounded_process import ProcessResult, run_bounded

# Explicit AMD64 runtime calls. No sockets, tracing, mounts, device creation,
# credential changes, namespaces, process-group changes or io_uring.
SYSCALLS = """
read, write, readv, writev, pread64, pwrite64, open, openat, close,
newstat, newfstat, newlstat, newfstatat, statx, access, faccessat, faccessat2,
lseek, mmap, mprotect, munmap, mremap, madvise, brk,
arch_prctl, set_tid_address, set_robust_list, rseq, prlimit64, getrlimit,
getrandom, getuid, geteuid, getgid, getegid, getgroups, getpid, getppid, gettid,
getpgrp, getpgid, getsid, getcwd, chdir, fchdir, newuname, sysinfo,
execve, exit, exit_group, rt_sigaction, rt_sigprocmask, rt_sigreturn,
rt_sigsuspend, rt_sigtimedwait, sigaltstack, kill, tgkill, wait4, waitid,
capget, fcntl, dup, dup2, dup3, pipe, pipe2, poll, ppoll, select, pselect6,
epoll_create1, epoll_ctl, epoll_wait, epoll_pwait, eventfd2, futex,
sched_yield, sched_getaffinity, sched_getparam, sched_getscheduler,
clock_gettime, clock_getres, clock_nanosleep, nanosleep, gettimeofday, times,
getrusage, alarm, setitimer, getitimer, restart_syscall,
getdents, getdents64, readlink, readlinkat, rename, renameat, renameat2,
unlink, unlinkat, mkdir, mkdirat, rmdir, link, linkat, symlink, symlinkat,
chmod, fchmod, fchmodat, chown, fchown, lchown, fchownat,
truncate, ftruncate, umask, utime, utimes, futimesat, utimensat,
fsync, fdatasync, statfs, fstatfs, flock, sendfile64, copy_file_range,
getxattr, fgetxattr, lgetxattr, listxattr, flistxattr, llistxattr,
fsetxattr,
prctl { option == 21 || option == 39 },
ioctl { cmd == 0x5401 || cmd == 0x5413 || cmd == 0x541B || cmd == 0x5421 || cmd == 0x5451 },
clone { (clone_flags & 0xFE82A080) == 0 }, fork, vfork
""".strip()
# Mask forbids all new namespaces, CLONE_PARENT and CLONE_UNTRACED (and IO).
SECCOMP = f"ALLOW {{ {SYSCALLS} }} ERRNO(38) {{ clone3 }} DEFAULT ERRNO(1)"


def _lease_uid(workspace: Path, state_root: Path) -> int:
    import pwd
    state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if state_root.is_symlink() or state_root.stat().st_uid != 0 or state_root.stat().st_mode & 0o077:
        raise RuntimeError("NsJail state must be a private root-owned directory")
    with sqlite3.connect(state_root / "uids.sqlite3", timeout=10) as database:
        database.execute("CREATE TABLE IF NOT EXISTS leases (workspace TEXT PRIMARY KEY, uid INTEGER UNIQUE)")
        database.execute("BEGIN IMMEDIATE")
        row = database.execute("SELECT uid FROM leases WHERE workspace=?", (str(workspace),)).fetchone()
        if row:
            return row[0]
        occupied = {entry.pw_uid for entry in pwd.getpwall()}
        used = {row[0] for row in database.execute("SELECT uid FROM leases")}
        for uid in range(100000, 200000):
            if uid not in occupied and uid not in used:
                database.execute("INSERT INTO leases VALUES (?, ?)", (str(workspace), uid))
                return uid
    raise RuntimeError("NsJail UID lease pool exhausted")


class NsjailSandbox:
    def __init__(self, workspace: Path) -> None:
        if os.name != "posix" or os.geteuid() != 0 or platform.machine() != "x86_64":
            raise RuntimeError("no-namespace NsJail currently requires Linux AMD64 trusted supervisor")
        if workspace.name != "workspace" or workspace.parent.name != "sandbox":
            raise RuntimeError("NsJail requires a dedicated <run>/sandbox/workspace")
        self.workspace = workspace
        self.root = workspace.parent
        self.run_root = self.root.parent
        self.binary = Path(os.environ.get("CODEAGENTBENCH_NSJAIL", "/root/autodl-tmp/bin/nsjail"))
        self.template = Path(os.environ.get("CODEAGENTBENCH_ROOTFS", "/root/autodl-tmp/nsjail-rootfs-20260927"))
        for path in (self.binary, self.template):
            info = path.stat()
            if path.is_symlink() or info.st_uid != 0 or info.st_mode & 0o022:
                raise RuntimeError("NsJail executable/template must be trusted and root-owned")
        self.uid = _lease_uid(workspace, Path(os.environ.get("CODEAGENTBENCH_NSJAIL_STATE", "/root/autodl-tmp/nsjail-state")))

    def _prepare(self) -> None:
        marker = self.run_root / "nsjail-prepared.json"
        identity = {"workspace": str(self.workspace), "uid": self.uid, "template": str(self.template)}
        if marker.exists():
            if json.loads(marker.read_text()) != identity:
                raise RuntimeError("NsJail workspace identity changed")
            return
        if (self.root / "usr").exists():
            raise RuntimeError("incomplete NsJail preparation; use a fresh run")
        for required in ("usr/bin/python3", "usr/bin/bash", "usr/bin/git"):
            if not (self.template / required).exists():
                raise RuntimeError("NsJail template is missing required tools")
        subprocess.run(["cp", "-a", "--reflink=auto", str(self.template) + "/.", str(self.root)],
                       check=True, timeout=180, capture_output=True)
        for name in ("tmp", "dev/shm", "var/tmp"):
            target = self.root / name
            target.mkdir(parents=True, exist_ok=True)
            os.chown(target, 0, 0)
            target.chmod(0o1777)
        # Sticky + root ownership prevents replacing or renaming .git.
        self.workspace.chmod(0o1777)
        os.chown(self.workspace, 0, 0)
        for directory, dirs, files in os.walk(self.workspace, followlinks=False):
            parent = Path(directory)
            if parent == self.workspace:
                dirs[:] = [name for name in dirs if name != ".git"]
            else:
                os.chown(parent, self.uid, self.uid, follow_symlinks=False)
            for name in files:
                os.chown(parent / name, self.uid, self.uid, follow_symlinks=False)
        git = self.workspace / ".git"
        if not git.is_dir() or git.is_symlink() or (git / "objects/info/alternates").exists():
            raise RuntimeError("NsJail requires private, dissociated Git metadata")
        for directory, dirs, files in os.walk(git, followlinks=False):
            for target in [Path(directory)] + [Path(directory) / name for name in files]:
                if target.is_symlink():
                    raise RuntimeError("unexpected symlink in Git metadata")
                os.chown(target, 0, 0)
                target.chmod(0o555 if target.is_dir() else 0o444)
        marker.write_text(json.dumps(identity), encoding="utf-8")

    def execute(self, command: str, cwd: Path, timeout: float, output_limit: int, *,
                cancellation_requested: Callable[[], bool] | None = None) -> ProcessResult:
        import fcntl
        with (self.run_root / "nsjail.lock").open("a") as lock:
            # Same-workspace concurrency is rejected, not serialized invisibly.
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._prepare()
            return self._execute_locked(command, cwd, timeout, output_limit, cancellation_requested)

    def _execute_locked(self, command: str, cwd: Path, timeout: float, output_limit: int,
                        cancellation_requested: Callable[[], bool] | None) -> ProcessResult:
        relative = cwd.relative_to(self.workspace).as_posix()
        sandbox_cwd = "/workspace" + ("/" + relative if relative != "." else "")
        config = self.run_root / "nsjail.cfg"
        config.write_text('\n'.join([
            'name: "codeagentbench-autodl"', 'mode: ONCE', f'cwd: {json.dumps(sandbox_cwd)}',
            *[f"clone_new{name}: false" for name in ("net", "user", "ns", "pid", "ipc", "uts", "cgroup", "time")],
            "mount_proc: false", "keep_env: false", "keep_caps: false", "skip_setsid: true",
            "disable_no_new_privs: false", f'uidmap {{ inside_id: "{self.uid}" }}',
            f'gidmap {{ inside_id: "{self.uid}" }}', f"time_limit: {math.ceil(timeout) + 1}", "max_cpus: 4",
            "rlimit_as: 1024", "rlimit_cpu: 120", "rlimit_core: 0", "rlimit_fsize: 16",
            "rlimit_nofile: 64", "rlimit_nproc: 16", "rlimit_nproc_type: VALUE",
            *[f'envar: {json.dumps(value)}' for value in (
                "PATH=/usr/bin:/bin", "HOME=/tmp", "TMPDIR=/tmp", "CI=1", "LANG=C.UTF-8",
                "PYTHONDONTWRITEBYTECODE=1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
                "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
                "GIT_CONFIG_COUNT=1", "GIT_CONFIG_KEY_0=safe.directory", "GIT_CONFIG_VALUE_0=/workspace",
                "AWS_ACCESS_KEY_ID=testing", "AWS_SECRET_ACCESS_KEY=testing",
                "AWS_DEFAULT_REGION=us-east-1", "AWS_EC2_METADATA_DISABLED=true")],
            f"seccomp_string: {json.dumps(SECCOMP)}", "",
        ]), encoding="utf-8")
        return run_bounded([str(self.binary), "--quiet", "--config", str(config), "--chroot", str(self.root),
                            "--", "/usr/bin/bash", "-c", command], timeout=timeout,
                           output_limit=output_limit,
                           cancellation_requested=cancellation_requested,
                           writable_paths=tuple(self.root / name for name in
                                                ("workspace", "tmp", "dev/shm", "var/tmp")))
