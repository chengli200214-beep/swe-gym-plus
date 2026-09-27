"""Linux process-group supervision with bounded capture and disk accounting."""
from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessResult:
    exit_code: int | None
    stdout: str
    stderr: str
    status: str


def writable_usage(paths: tuple[Path, ...], *, bytes_limit: int, files_limit: int) -> None:
    """Walk directory descriptors, never following guest-controlled symlinks."""
    total_bytes = total_files = 0

    def walk(fd: int) -> None:
        nonlocal total_bytes, total_files
        with os.scandir(fd) as entries:
            for item in entries:
                # .git is host-owned and immutable during execution.
                if item.name == ".git":
                    continue
                try:
                    info = item.stat(follow_symlinks=False)
                    total_files += 1
                    total_bytes += info.st_size
                    if total_bytes > bytes_limit or total_files > files_limit:
                        raise RuntimeError("writable filesystem budget exceeded")
                    if item.is_dir(follow_symlinks=False):
                        child = os.open(item.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                        try:
                            walk(child)
                        finally:
                            os.close(child)
                except (FileNotFoundError, NotADirectoryError):
                    # A concurrently removed/replaced entry is checked next pass.
                    continue

    for path in paths:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            walk(fd)
        finally:
            os.close(fd)


def run_bounded(command: list[str], *, timeout: float, output_limit: int,
                writable_paths: tuple[Path, ...], bytes_limit: int = 256 * 1024 * 1024,
                files_limit: int = 20000) -> ProcessResult:
    """Kill the entire group on any exit, deadline, quota failure or exception.

    NsJail must use skip_setsid and the guest policy must deny setsid/setpgid.
    Disk scanning is a periodic guard, not an instantaneous filesystem quota.
    RLIMIT_FSIZE bounds individual files between scans.
    """
    if os.name != "posix" or timeout <= 0 or output_limit <= 0:
        raise ValueError("bounded process requires Linux and positive limits")
    writable_usage(writable_paths, bytes_limit=bytes_limit, files_limit=files_limit)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    captured = 0
    status = "completed"
    deadline = time.monotonic() + timeout
    next_scan = time.monotonic()
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, cwd="/", env={"PATH": "/usr/bin:/bin"},
                               close_fds=True, start_new_session=True)
    try:
        with selectors.DefaultSelector() as selector:
            for name in buffers:
                pipe = getattr(process, name)
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, name)
            while selector.get_map():
                now = time.monotonic()
                if now >= deadline:
                    status = "timeout"
                    break
                if now >= next_scan:
                    try:
                        writable_usage(writable_paths, bytes_limit=bytes_limit, files_limit=files_limit)
                    except (RuntimeError, OSError, RecursionError):
                        status = "disk_limit"
                        break
                    next_scan = now + 0.5
                for key, _ in selector.select(min(0.1, max(0, deadline - now))):
                    chunk = os.read(key.fileobj.fileno(), 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    captured += len(chunk)
                    buffer = buffers[key.data]
                    buffer.extend(chunk[:max(0, output_limit - len(buffer))])
                    if captured > max(1024 * 1024, output_limit * 4):
                        status = "output_limit"
                        break
                if status != "completed":
                    break
                if process.poll() is not None:
                    # Descendants must not outlive a completed shell.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        if status == "completed":
            remaining = deadline - time.monotonic()
            try:
                process.wait(timeout=max(0.01, remaining))
            except subprocess.TimeoutExpired:
                status = "timeout"
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
        for name in buffers:
            getattr(process, name).close()
    if status != "completed":
        buffers["stderr"].extend(("\nHarness resource limit: " + status).encode())
    return ProcessResult(process.returncode if status == "completed" else None,
                         buffers["stdout"].decode("utf-8", errors="replace"),
                         buffers["stderr"].decode("utf-8", errors="replace"), status)
