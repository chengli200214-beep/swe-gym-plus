"""Parse external model actions at the runtime protocol boundary."""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass

from codeagentbench.adapters.text_edit import TextEdit, edit_command, validate_edit
from codeagentbench.adapters.source_read import SourceRead, read_command, validate_read


@dataclass(frozen=True)
class AgentAction:
    """Parsed model action."""

    command: str = ""
    done: bool = False
    message: str = ""
    edit: TextEdit | None = None
    read: SourceRead | None = None

    @property
    def executable(self) -> bool:
        return not self.done and bool(self.command or self.edit or self.read)

    def to_dict(self) -> dict:
        if self.edit is not None:
            return {"edit": self.edit.to_dict(), "done": False, "message": self.message}
        if self.read is not None:
            return {"read": self.read.to_dict(), "done": False, "message": self.message}
        return {"command": self.command, "done": self.done, "message": self.message}

    def tool_command(self, *, expected_sha256: str | None = None) -> str:
        if self.edit is not None:
            return edit_command(self.edit, expected_sha256=expected_sha256)
        return read_command(self.read) if self.read is not None else self.command


_DSML_MARKER = "\uFF5C\uFF5CDSML\uFF5C\uFF5C"
_DSML_COMMAND = re.compile(
    rf"<{re.escape(_DSML_MARKER)}\s+parameter\s+name=[\"']command[\"'][^>]*>"
    rf"(.*?)</{re.escape(_DSML_MARKER)}\s+parameter>",
    flags=re.DOTALL | re.IGNORECASE,
)
_DSML_TAG_COMMAND = re.compile(
    r"<command(?:\s[^>]*)?>(.*?)</command>",
    flags=re.DOTALL | re.IGNORECASE,
)
_DSML_INVOKE_COMMAND = re.compile(
    rf"<{re.escape(_DSML_MARKER)}\s+invoke\s+name=[\"']command[\"'][^>]*>"
    rf"(.*?)</{re.escape(_DSML_MARKER)}\s+parameter>",
    flags=re.DOTALL | re.IGNORECASE,
)
_DSML_INLINE_COMMAND = re.compile(
    rf"<{re.escape(_DSML_MARKER)}\s+invoke\s+name=[\"']command[\"']\s*:\s*"
    r"(\"(?:\\.|[^\"\\])*\")",
    flags=re.DOTALL | re.IGNORECASE,
)
_DSML_PLAIN_PARAMETER_COMMAND = re.compile(
    r"<parameter\s+name=[\"']command[\"'][^>]*>(.*?)</parameter>",
    flags=re.DOTALL | re.IGNORECASE,
)
_MALFORMED_ACTION = re.compile(
    r'''^\s*\{\s*["']command["']\s*:\s*["'](?P<command>.*?)["']\s*,\s*["']done["']\s*:\s*(?P<done>true|false)'''
    r'''(?:\s*,\s*["']message["']\s*:\s*["'](?P<message>.*?)["'])?\s*\}\s*$''',
    flags=re.DOTALL | re.IGNORECASE,
)
_EDIT_COMMAND_FENCE = re.compile(
    r"(?:\*\*)?edit\s+command[\s*]*:[\s*]*```(?:bash|sh|shell)?\s*(.*?)\s*```",
    flags=re.DOTALL | re.IGNORECASE,
)
def parse_action(text: str) -> AgentAction:
    """Parse the JSON action protocol plus narrowly-scoped model formatting repairs."""
    candidate = text.strip()
    original_candidate = candidate
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    embedded_fence = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    had_fence = bool(re.match(r"^\s*```(?:json)?(?:\s|$)", candidate, flags=re.IGNORECASE))
    if fenced:
        candidate = fenced.group(1).strip()
    elif embedded_fence:
        # Small models sometimes prepend an explanation before a fenced action.
        # Parse only the fenced payload; do not mine arbitrary prose for a
        # command.
        candidate = embedded_fence.group(1).strip()
        had_fence = True
    elif had_fence:
        # Keep parsing the JSON body when a model opened a fence but stopped
        # before emitting its closing marker.
        candidate = re.sub(r"^\s*```(?:json)?\s*", "", candidate, count=1, flags=re.IGNORECASE).strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        payload = None
        if had_fence and "\\'" in candidate:
            # JSON does not use backslash-single-quote escapes, but models
            # frequently copy shell quoting into a JSON string. This narrow
            # repair is safe because a single quote has no special meaning in
            # a JSON string.
            try:
                payload = json.loads(candidate.replace("\\'", "'"))
            except json.JSONDecodeError:
                payload = None

        if payload is None:
            edit_command = _EDIT_COMMAND_FENCE.search(original_candidate)
            if edit_command:
                command = edit_command.group(1).strip()
                if command:
                    return AgentAction(command=command, message="parsed labeled edit command")
            dsml_command = _DSML_COMMAND.search(candidate)
            if dsml_command:
                command = html.unescape(dsml_command.group(1)).strip()
                if command:
                    return AgentAction(command=command, message="parsed DeepSeek DSML shell call")
            # Some DeepSeek-compatible endpoints serialize the tool call with a
            # bare <command> element instead of a named parameter.
            dsml_tag_command = _DSML_TAG_COMMAND.search(candidate)
            if dsml_tag_command:
                command = html.unescape(dsml_tag_command.group(1)).strip()
                if command:
                    return AgentAction(command=command, message="parsed DeepSeek DSML command tag")
            dsml_invoke_command = _DSML_INVOKE_COMMAND.search(candidate)
            if dsml_invoke_command:
                command = html.unescape(dsml_invoke_command.group(1)).strip()
                if command:
                    return AgentAction(command=command, message="parsed DeepSeek DSML invoke command")
            dsml_inline_command = _DSML_INLINE_COMMAND.search(candidate)
            if dsml_inline_command:
                try:
                    command = json.loads(dsml_inline_command.group(1)).strip()
                except (TypeError, json.JSONDecodeError):
                    command = html.unescape(dsml_inline_command.group(1).strip().strip('"'))
                if command:
                    return AgentAction(command=command, message="parsed DeepSeek DSML inline command")
            dsml_plain_parameter = _DSML_PLAIN_PARAMETER_COMMAND.search(candidate)
            if dsml_plain_parameter:
                command = html.unescape(dsml_plain_parameter.group(1)).strip()
                if command:
                    return AgentAction(command=command, message="parsed DeepSeek DSML plain parameter")

            # Some compatible chat models prepend prose and emit several JSON
            # candidates. Decode all complete objects without evaluating or
            # repairing arbitrary text, then use the last action-shaped object;
            # models commonly put their consolidated command last.
            decoder = json.JSONDecoder()
            action_candidates: list[dict[str, object]] = []
            for index, character in enumerate(candidate):
                if character != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(candidate[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and any(k in parsed for k in ("command", "done", "edit", "read")):
                    action_candidates.append(parsed)
            if action_candidates:
                # Some models emit an entire imagined tool transcript in one
                # assistant message and end it with done=true. Those commands
                # have not been executed, so never accept the synthetic final
                # completion. Start with the first concrete action and let the
                # runtime observe each real receipt before asking for the next.
                payload = action_candidates[0] if len(action_candidates) >= 3 else action_candidates[-1]

            if payload is None:
                # A frequent model formatting error is an unescaped quote inside
                # the command itself, for example findstr /c:"launch_template".
                # Repair only the explicitly-shaped action envelope; do not try
                # to evaluate or broadly rewrite arbitrary model prose.
                malformed = _MALFORMED_ACTION.match(candidate)
                if malformed:
                    command = malformed.group("command").replace("\\\\", "\\").replace('\\"', '"')
                    message = malformed.group("message") or ""
                    return AgentAction(
                        command=command.strip(),
                        done=malformed.group("done").lower() == "true",
                        message=message,
                    )
                # Never execute a truncated command: a missing suffix can turn
                # a valid edit into a different operation. Preserve raw history
                # and fail at the protocol boundary instead of inventing bytes.
                raise ValueError("model response is not valid JSON") from None
    if not isinstance(payload, dict):
        raise ValueError("model action must be a JSON object")
    done = payload.get("done", False)
    if not isinstance(done, bool):
        raise ValueError("done must be a JSON boolean")
    if "edit" in payload:
        if done or "command" in payload or set(payload) - {"edit", "done", "message"}:
            raise ValueError("edit cannot include command, done=true or unknown fields")
        return AgentAction(edit=validate_edit(payload["edit"]), message=str(payload.get("message", "")))
    if "read" in payload:
        if done or "command" in payload or set(payload) - {"read", "done", "message"}:
            raise ValueError("read cannot include command, done=true or unknown fields")
        return AgentAction(read=validate_read(payload["read"]), message=str(payload.get("message", "")))
    command = payload.get("command", "")
    if not isinstance(command, str) or (not done and not command.strip()):
        raise ValueError("action needs a non-empty command unless done=true")
    return AgentAction(command=command.strip(), done=done, message=str(payload.get("message", "")))
