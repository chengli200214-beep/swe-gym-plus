"""Observe-action runtime that adapts a chat model to a bash coding loop."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Callable

from codeagentbench.adapters.model import ChatModel, ModelResponse
from codeagentbench.harness.budget import BudgetExceeded, BudgetLedger
from codeagentbench.harness.context import ContextManager
from codeagentbench.harness.recovery import ActionJournal
from codeagentbench.models import AgentTaskView, RunConfig, RunState, TaskRecord, ToolIntent
from codeagentbench.sandbox.executor import BashExecutor
from codeagentbench.sandbox.workspace import Workspace
from codeagentbench.storage.artifacts import ArtifactStore


@dataclass(frozen=True)
class AgentAction:
    """Parsed model action."""

    command: str = ""
    done: bool = False
    message: str = ""


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
_TRUNCATED_COMMAND_ACTION = re.compile(
    r'''^\s*\{\s*["']command["']\s*:\s*["'](?P<command>.*)$''',
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class RuntimeResult:
    run_id: str
    status: str
    diff: str
    steps: int
    failure_reason: str = ""
    state: RunState | None = None
    visible_test_passed: bool = False


def parse_action(text: str) -> AgentAction:
    """Parse the JSON action protocol plus narrowly-scoped model formatting repairs."""

    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    had_fence = bool(re.match(r"^\s*```(?:json)?(?:\s|$)", candidate, flags=re.IGNORECASE))
    if fenced:
        candidate = fenced.group(1).strip()
    elif had_fence:
        # Keep parsing the JSON body when a model opened a fence but stopped
        # before emitting its closing marker.
        candidate = re.sub(r"^\s*```(?:json)?\s*", "", candidate, count=1, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
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
        payload = None
        action_candidates: list[dict[str, object]] = []
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and ("command" in parsed or "done" in parsed):
                action_candidates.append(parsed)
        if action_candidates:
            # Some models emit an entire imagined tool transcript in one
            # assistant message and end it with done=true. Those commands
            # have not been executed, so never accept the synthetic final
            # completion. Start with the first concrete action and let the
            # runtime observe each real receipt before asking for the next.
            if len(action_candidates) >= 3:
                payload = action_candidates[0]
            else:
                payload = action_candidates[-1]
        if payload is None:
            # A frequent model formatting error is an unescaped quote inside
            # the command itself, for example findstr /c:"launch_template".
            # Repair only the explicitly-shaped action envelope; do not try to
            # evaluate or broadly rewrite arbitrary model prose.
            malformed = _MALFORMED_ACTION.match(candidate)
            if malformed:
                command = malformed.group("command").replace("\\\\", "\\").replace('\\"', '"')
                message = malformed.group("message") or ""
                return AgentAction(
                    command=command.strip(),
                    done=malformed.group("done").lower() == "true",
                    message=message,
                )
            # Small local models occasionally start an explicit JSON/fenced
            # action and stop before emitting the closing quote/braces.  Only
            # repair this shape when the response itself starts with the
            # action envelope; never mine an arbitrary prose response for a
            # command.  The shell receipt will expose any genuinely truncated
            # command to the next model turn.
            truncated = _TRUNCATED_COMMAND_ACTION.match(candidate)
            if truncated and (had_fence or candidate.lstrip().startswith('{')):
                command = truncated.group("command").strip()
                if command.endswith("```"):
                    command = command[:-3].rstrip()
                if command.endswith('"'):
                    command = command[:-1]
                command = command.replace("\\\\", "\\").replace('\\"', '"').strip()
                if command:
                    return AgentAction(command=command, message="parsed truncated JSON command")
            raise ValueError("model response is not valid JSON") from None
    if not isinstance(payload, dict):
        raise ValueError("model action must be a JSON object")
    done = bool(payload.get("done", False))
    command = payload.get("command", "")
    if not isinstance(command, str) or (not done and not command.strip()):
        raise ValueError("action needs a non-empty command unless done=true")
    return AgentAction(command=command.strip(), done=done, message=str(payload.get("message", "")))


class AgentRuntime:
    """Run a model in a bounded workspace with durable action boundaries."""

    def __init__(self, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or ArtifactStore()

    def run(
        self,
        task: TaskRecord,
        workspace: Workspace,
        model: ChatModel,
        config: RunConfig,
        *,
        run_id: str | None = None,
        failure_injector: Callable[[str], None] | None = None,
        ledger: BudgetLedger | None = None,
    ) -> RuntimeResult:
        run_id = run_id or f"{task.instance_id}-{int(time.time())}"
        run_dir = self.artifact_store.start_run(run_id, task.instance_id, config)
        journal = ActionJournal(run_dir / "actions.jsonl")
        executor = BashExecutor(workspace.path, journal)
        ledger = ledger or BudgetLedger(config.max_tokens, config.max_seconds, config.max_cost_usd, config.max_tool_calls)
        state = RunState(run_id, task.instance_id, status="running", remaining_tokens=config.max_tokens, remaining_seconds=config.max_seconds)
        context = ContextManager(task.issue)
        view = task.agent_view()
        messages = [{"role": "system", "content": self._system_prompt()}, {"role": "user", "content": self._initial_prompt(view)}]
        self.artifact_store.append_event(run_id, {"type": "prompt", "content": messages[-1]["content"]})
        previous_signature = ""
        repeated = 0
        visible_test_passed = False
        self._checkpoint(state, ledger, workspace, messages)
        try:
            for step in range(config.max_steps):
                state.step = step
                response: ModelResponse = model.complete(messages, temperature=config.temperature)
                ledger.consume(tokens=response.prompt_tokens + response.completion_tokens, cost_usd=response.cost_usd)
                self.artifact_store.append_event(run_id, {"type": "model", "content": response.text, "prompt_tokens": response.prompt_tokens, "completion_tokens": response.completion_tokens, "cost_usd": response.cost_usd})
                messages.append({"role": "assistant", "content": response.text})
                try:
                    action = parse_action(response.text)
                except ValueError as exc:
                    if visible_test_passed and workspace.diff():
                        # A model may emit a malformed follow-up after it has
                        # already produced and tested a patch. Preserve the
                        # candidate for independent evaluation, but keep the
                        # protocol warning in the durable summary.
                        state.status = "completed"
                        state.failure_reason = f"post-test protocol warning: {exc}"
                    else:
                        state.status, state.failure_reason = "failed", str(exc)
                    break
                if action.done:
                    state.status = "completed"
                    break
                action_id = f"{run_id}-action-{step:04d}"
                pre_digest = workspace.digest
                intent = ToolIntent(action_id, action.command, str(workspace.path), min(120.0, config.max_seconds), True, action_id, pre_digest)
                state.pending_action_id = action_id
                journal.record_intent(intent)
                self._checkpoint(state, ledger, workspace, messages)
                if failure_injector:
                    failure_injector("after_intent")
                receipt = executor.execute(intent, intent_already_recorded=True)
                ledger.consume(seconds=receipt.duration_seconds, tool_calls=1)
                state.pending_action_id = None
                state.last_action_id = action_id
                if receipt.exit_code == 0 and any(token in action.command.lower() for token in ("pytest", "tox", "unittest", "test")):
                    visible_test_passed = True
                if receipt.stdout:
                    # Keep a bounded, recent slice of successful exploration
                    # output so compression does not make the model repeat
                    # the same lookup after a long file listing.
                    context.add("tool_output", receipt.stdout[:4_000], source=action_id, priority=30)
                if receipt.stderr:
                    context.add(
                        "test_failure" if receipt.exit_code else "tool_output",
                        receipt.stderr[:4_000],
                        source=action_id,
                        priority=80 if receipt.exit_code else 30,
                    )
                observation = {"exit_code": receipt.exit_code, "stdout": receipt.stdout, "stderr": receipt.stderr, "timed_out": receipt.timed_out}
                messages.append({"role": "user", "content": "Tool result:\n" + json.dumps(observation, ensure_ascii=False)})
                signature = hashlib.sha256(f"{action.command}\0{pre_digest}\0{receipt.exit_code}\0{receipt.stdout}\0{receipt.stderr}".encode("utf-8")).hexdigest()
                repeated = repeated + 1 if signature == previous_signature else 0
                previous_signature = signature
                if repeated >= 2:
                    state.status, state.failure_reason = "failed", "no progress: identical command, workspace and result repeated"
                    self.artifact_store.append_event(run_id, {"type": "no_progress", "action_id": action_id})
                    break
                self.artifact_store.append_event(run_id, {"type": "tool", "intent": asdict(intent), "receipt": asdict(receipt)})
                diff = workspace.diff()
                if diff:
                    context.add("diff", diff, source=action_id, priority=90, confirmed=True)
                if step == 5 and not diff:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Harness checkpoint: six actions have been used without a patch. "
                                "You have enough context to implement the issue. Stop broad exploration, "
                                "read only the exact target method if needed, and apply the smallest "
                                "implementation change in your next action. Do not run another search "
                                "or test-discovery command first."
                            ),
                        }
                    )
                context_text = context.render(max_chars=8_000)
                if sum(len(message.get("content", "")) for message in messages) > 16_000:
                    messages = [
                        messages[0],
                        {"role": "user", "content": self._initial_prompt(view)},
                        {"role": "user", "content": "Evidence summary after context compression:\n" + context_text},
                    ]
                state.context_compressions = len(context.compressions)
                self._checkpoint(state, ledger, workspace, messages)
            else:
                state.status, state.failure_reason = "failed", "maximum agent steps reached"
        except BudgetExceeded as exc:
            state.status, state.failure_reason = "blocked", str(exc)
        except Exception as exc:
            state.status, state.failure_reason = "interrupted", str(exc)
            self._checkpoint(state, ledger, workspace, messages)
            raise
        snapshot = ledger.snapshot
        state.spent_tokens = snapshot.tokens
        state.spent_seconds = snapshot.seconds
        state.spent_cost_usd = snapshot.cost_usd
        state.remaining_tokens = snapshot.remaining_tokens
        state.remaining_seconds = snapshot.remaining_seconds
        state.messages = messages
        state.pending_action_id = None if state.status != "interrupted" else state.pending_action_id
        self._checkpoint(state, ledger, workspace, messages)
        result = RuntimeResult(run_id, state.status, workspace.diff(), state.step + 1, state.failure_reason, state, visible_test_passed)
        self.artifact_store.save_summary(run_id, {"run_id": run_id, "task_id": task.instance_id, "status": state.status, "diff": result.diff, "steps": result.steps, "failure_reason": result.failure_reason, "budget": asdict(snapshot)})
        return result

    def recover(self, run_id: str, workspace: Workspace) -> list[dict[str, str]]:
        """Inspect a checkpoint and return conservative decisions for a worker."""

        run_dir = self.artifact_store.run_dir(run_id)
        journal = ActionJournal(run_dir / "actions.jsonl")
        return [asdict(item) for item in journal.decide_recovery(workspace.digest)]

    def _checkpoint(self, state: RunState, ledger: BudgetLedger, workspace: Workspace, messages: list[dict[str, str]]) -> None:
        state.messages = messages
        snapshot = ledger.snapshot
        state.spent_tokens = snapshot.tokens
        state.spent_seconds = snapshot.seconds
        state.spent_cost_usd = snapshot.cost_usd
        state.remaining_tokens = snapshot.remaining_tokens
        state.remaining_seconds = snapshot.remaining_seconds
        self.artifact_store.save_checkpoint(state, budget=asdict(snapshot), digest=workspace.digest)

    @staticmethod
    def _system_prompt() -> str:
        if os.name == "nt":
            shell_note = (
                "The workspace uses Windows cmd.exe: you are already in the repository root; "
                "do not cd to /repo. Use dir, type, findstr, python and git with Windows paths. "
                "Do not use Unix-only commands or paths such as tail, grep, sed, /repo, "
                "or bash pipelines."
            )
        else:
            shell_note = (
                "The workspace uses Linux bash: you are already in the repository root. "
                "Use pwd, ls, find, grep, sed, awk, python and git as needed. "
                "Do not use Windows-only commands such as dir, type, findstr, or PowerShell syntax."
            )
        return (
            "You are a coding agent. Inspect and modify the repository with the available shell. "
            f"{shell_note} "
            'Return exactly JSON: {"command":"...", "done":false, "message":"..."}. '
            "Set done=true only after testing. Do not reveal or ask for gold patches. "
            "Use a short observe-edit-test loop: after at most 3 exploration commands, "
            "read the exact implementation lines and apply the smallest patch; do not repeat "
            "directory listings or commands whose output you already have. Run the allowed test "
            "command before finishing. If an allowed test target or test class is absent from the "
            "checkout, treat it as a hidden evaluation test and implement the issue from the "
            "problem statement instead of spending more actions searching for it. Do not finish "
            "without a concrete diff unless the issue is proven unrelated to the repository."
        )

    @staticmethod
    def _initial_prompt(view: AgentTaskView) -> str:
        allowed_test_command = view.allowed_test_command
        test_tokens = allowed_test_command.split()
        if len(allowed_test_command) > 2_000 and len(test_tokens) > 7:
            # Keep the agent prompt small for manifests containing dozens of
            # regression nodes. The evaluator still runs the complete frozen
            # command from EvalSpec in a fresh workspace.
            allowed_test_command = (
                " ".join(test_tokens[:7])
                + "  # representative tests; full regression is run independently"
            )
        return json.dumps(
            {
                "instance_id": view.instance_id,
                "repo": view.repo,
                "base_commit": view.base_commit,
                "issue": view.issue,
                "allowed_test_command": allowed_test_command,
                "context": view.context,
            },
            ensure_ascii=False,
        )
