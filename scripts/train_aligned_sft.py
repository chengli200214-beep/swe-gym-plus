"""Isolated last-action-only encoder for round two; old encoder remains intact."""
from codeagentbench import train_sft
from codeagentbench.training.action_context import encode_next_action

if __name__ == "__main__":
    train_sft.encode_example = encode_next_action
    raise SystemExit(train_sft.main())
