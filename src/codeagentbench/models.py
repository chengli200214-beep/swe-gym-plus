"""Stable data contracts shared by the runtime, evaluator and service."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """An evaluator result; ``unknown`` is intentionally not treated as pass."""

    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvalSpec:
    """Evaluation-only information that must never enter the agent context."""

    test_patch: str = ""
    gold_patch: str = ""
    fail_to_pass: tuple[str, ...] = ()
    pass_to_pass: tuple[str, ...] = ()
    test_command: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvalSpec":
        data = data or {}

        def values(value: Any) -> tuple[str, ...]:
            if value is None:
                return ()
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    try:
                        value = ast.literal_eval(value)
                    except (SyntaxError, ValueError):
                        return (value,)
            if isinstance(value, (list, tuple)):
                return tuple(str(item) for item in value)
            return (str(value),)

        return cls(
            test_patch=str(data.get("test_patch", "")),
            gold_patch=str(data.get("gold_patch", data.get("patch", ""))),
            fail_to_pass=values(data.get("fail_to_pass", data.get("FAIL_TO_PASS", ()))),
            pass_to_pass=values(data.get("pass_to_pass", data.get("PASS_TO_PASS", ()))),
            test_command=str(data.get("test_command", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "fail_to_pass": list(self.fail_to_pass),
            "pass_to_pass": list(self.pass_to_pass),
        }


@dataclass(frozen=True)
class AgentTaskView:
    """The deliberately redacted task representation visible to an agent."""

    instance_id: str
    repo: str
    base_commit: str
    issue: str
    test_patch: str = ""
    allowed_test_command: str = ""
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskRecord:
    """A normalized SWE-Gym task with an isolated evaluation specification."""

    instance_id: str
    repo: str
    base_commit: str
    issue: str
    split: str = "unspecified"
    group_id: str = ""
    eval_spec: EvalSpec = field(default_factory=EvalSpec)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRecord":
        eval_data = data.get("eval_spec")
        if eval_data is None:
            eval_data = data
        return cls(
            instance_id=str(data.get("instance_id", data.get("id", ""))),
            repo=str(data.get("repo", data.get("repository", ""))),
            base_commit=str(data.get("base_commit", data.get("base_sha", ""))),
            issue=str(data.get("issue", data.get("problem_statement", ""))),
            split=str(data.get("split", "unspecified")),
            group_id=str(data.get("group_id", data.get("pr", data.get("pull_number", "")))),
            eval_spec=EvalSpec.from_dict(eval_data),
            metadata=dict(data.get("metadata", {})),
        )

    def agent_view(self) -> AgentTaskView:
        """Return public task context without evaluator-only tests or labels."""

        public_context = self.metadata.get("context") or self.metadata.get("hints_text", "")
        return AgentTaskView(
            instance_id=self.instance_id,
            repo=self.repo,
            base_commit=self.base_commit,
            issue=self.issue,
            # A manifest's test patch and test command belong to the sealed
            # evaluator. Only a separately curated visible command may be
            # shown to the agent; it must not be derived from EvalSpec.
            allowed_test_command=str(self.metadata.get("agent_test_command") or ""),
            context=str(public_context),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "issue": self.issue,
            "split": self.split,
            "group_id": self.group_id,
            "eval_spec": self.eval_spec.to_dict(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RunConfig:
    """A reproducible run configuration and its whole-task budget."""

    model: str = "scripted"
    temperature: float = 0.2
    max_steps: int = 12
    max_tool_calls: int = 32
    max_tokens: int = 32_000
    max_seconds: float = 900.0
    max_cost_usd: float = 5.0
    candidate_count: int = 1
    harness: str = "reliable"
    image: str = "local"
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolIntent:
    """An auditable action prepared before it is executed."""

    action_id: str
    command: str
    cwd: str
    timeout_seconds: float = 60.0
    side_effect: bool = True
    idempotency_key: str = ""
    pre_digest: str = ""


@dataclass(frozen=True)
class ToolReceipt:
    """The result written after an action finishes or times out."""

    action_id: str
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool
    post_digest: str
    status: str = "completed"


@dataclass(frozen=True)
class Candidate:
    """One independent rollout candidate; evaluator labels are kept separate."""

    candidate_id: str
    run_id: str
    diff: str
    status: str
    visible_test_passed: bool = False
    tool_errors: int = 0
    changed_files: tuple[str, ...] = ()
    duration_seconds: float = 0.0
    token_count: int = 0
    cost_usd: float = 0.0
    evaluation: EvaluationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """A structured independent evaluation result."""

    verdict: Verdict
    exit_code: int | None
    fail_to_pass: bool | None
    pass_to_pass: bool | None
    stdout: str = ""
    stderr: str = ""
    failed_tests: tuple[str, ...] = ()
    duration_seconds: float = 0.0
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict is Verdict.PASSED

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["verdict"] = self.verdict.value
        result["failed_tests"] = list(self.failed_tests)
        return result


@dataclass(frozen=True)
class SelectionResult:
    """Decision made from agent-visible evidence, before final evaluation."""

    selected_candidate_id: str | None
    ranking: tuple[str, ...]
    scores: dict[str, float]
    reason: str
    oracle_coverage_at_k: bool = False


@dataclass
class RunState:
    """Serializable state used for checkpointing and safe recovery."""

    run_id: str
    task_id: str
    status: str = "created"
    step: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_action_id: str | None = None
    pending_action_id: str | None = None
    remaining_tokens: int = 32_000
    remaining_seconds: float = 900.0
    spent_tokens: int = 0
    spent_seconds: float = 0.0
    spent_cost_usd: float = 0.0
    context_compressions: int = 0
    failure_reason: str = ""
    next_step: int = 0
    pending_model: bool = False
    pending_action_text: str = ""
    tested_diff: str | None = None
    previous_signature: str = ""
    repeated: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        return cls(**data)
