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

    The runner checks the starting cwd and filters credential-like variables.
    This is not a sandbox: a shell command can read or write files outside its
    starting cwd. Real model-generated commands require a separate low-privilege
    container backend without secrets or private data mounted into it.
    """

    def __init__(self, workspace: str | Path, journal: ActionJournal | None = None, output_limit: int = 20_000) -> None:
        self.workspace = Path(workspace).resolve()
        self.journal = journal
        self.output_limit = output_limit
        self.bash = shutil.which("bash") or shutil.which("wsl.exe")

    def execute(self, intent: ToolIntent, *, intent_already_recorded: bool = False) -> ToolReceipt:
        cwd = Path(intent.cwd).resolve()
        if cwd != self.workspace and self.workspace not in cwd.parents:
            raise ValueError("tool cwd must be inside the run workspace")
        if self.journal and not intent_already_recorded:
            self.journal.record_intent(intent)
        started = time.monotonic()
        env = self._tool_environment()
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
        if use_native_bash:
            command = [self.bash, "-c", command_text]
        elif use_wsl:
            command = [self.bash, "-lc", command_text]
        else:
            command = command_text
        try:
            try:
                result = subprocess.run(command, cwd=cwd, env=env, shell=not (use_wsl or use_native_bash), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=intent.timeout_seconds, check=False)
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
