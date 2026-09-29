"""Next-action SFT with deployment alignment checked before GPU allocation."""
import argparse
import json
from pathlib import Path
import sys

from codeagentbench import train_sft
from codeagentbench.training.action_context import encode_next_action
from codeagentbench.training.readiness import audit_readiness, prompt_hash


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        return train_sft.main(argv)
    # Only data/config flags are inspected here. The existing trainer owns its
    # full CLI and validation; honor the same working-directory override rule.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--typed-action-sampling-weight", type=float)
    args, _ = parser.parse_known_args(argv)
    try:
        from codeagentbench.runtime import AgentRuntime

        overrides = {"train_file": str(args.train_file.resolve())} if args.train_file else {}
        if args.typed_action_sampling_weight is not None:
            overrides["typed_action_sampling_weight"] = args.typed_action_sampling_weight
        config = train_sft.build_config(args.config.resolve(), overrides)
        system = AgentRuntime._system_prompt()
        report = audit_readiness(train_sft.load_records(config.train_file),
                                 expected_system_prompt=system,
                                 expected_prompt_policy=config.raw.get("deployment_prompt_policy"))
        if config.raw.get("deployment_system_prompt_sha256") != prompt_hash(system):
            report["errors"]["deployment_prompt_not_frozen_or_changed"] = 1
            report["ready"] = False
        print(json.dumps({"training_readiness": report}, ensure_ascii=False))
        if not report["ready"]:
            return 2
    except (OSError, ValueError, RuntimeError) as exc:
        # Do not print data lines or model outputs in an error path.
        print(f"training preflight failed ({type(exc).__name__}); inspect local config/data", file=sys.stderr)
        return 2
    original_encoder = train_sft.encode_example
    try:
        train_sft.encode_example = encode_next_action
        return train_sft.main(argv)
    finally:
        train_sft.encode_example = original_encoder


if __name__ == "__main__":
    raise SystemExit(main())
