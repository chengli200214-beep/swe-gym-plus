"""One shared, bounded action/receipt view for offline SFT and local inference."""
from __future__ import annotations

import json

POLICY = "last-action-v2"


def compact_context(messages: list[dict[str, str]], *, output_chars: int = 4000) -> list[dict[str, str]]:
    """Keep the original task and most recent *real* interaction, without labels.

    A tool observation must be the harness's labelled JSON receipt, not text
    invented by the assistant. Previous assistant actions remain context only.
    The same bounded view is used for both Base and SFT inference.
    """
    if output_chars < 1 or len(messages) < 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError("expected system + initial user task")
    result = [dict(messages[0]), dict(messages[1])]
    tool_indices = [i for i, m in enumerate(messages[2:], 2) if m["role"] == "user" and m["content"].startswith("Tool result:\n")]
    if not tool_indices:
        return result
    index = tool_indices[-1]
    action_indices = [i for i in range(2, index) if messages[i]["role"] == "assistant"]
    if not action_indices:
        raise ValueError("real tool receipt has lost its previous action")
    from codeagentbench.adapters.action import parse_action
    action = parse_action(messages[action_indices[-1]]["content"])
    if action.done or not action.command:
        raise ValueError("tool receipt must follow an executable action")
    observation = json.loads(messages[index]["content"].split("\n", 1)[1])
    required = {"exit_code", "stdout", "stderr", "timed_out"}
    if not required <= observation.keys():
        raise ValueError("incomplete real tool receipt")
    bounded = {k: observation[k] for k in ("exit_code", "stdout", "stderr", "timed_out")}
    for key in ("stdout", "stderr"):
        text = str(bounded[key])
        bounded[key] = text[:output_chars]
        if len(text) > output_chars or observation.get(key + "_truncated") is True:
            bounded[key + "_truncated"] = True
    result += [
        {"role": "assistant", "content": json.dumps({"command": action.command, "done": False, "message": action.message}, ensure_ascii=False, separators=(",", ":"))},
        {"role": "user", "content": "Tool result:\n" + json.dumps(bounded, ensure_ascii=False)},
    ]
    # Keep factual failure feedback deterministically in both data and rollout.
    if bounded["exit_code"] != 0 or bounded["timed_out"]:
        result.append({"role": "user", "content": "The previous command failed or timed out. Use its actual error to choose a different command; do not repeat it unchanged."})
    return result


def encode_next_action(tokenizer, messages, max_seq_len):
    """Mask all history, including prior assistant turns; train only next action."""
    if len(messages) < 3 or messages[-1]["role"] != "assistant":
        raise ValueError("last message must be the next assistant action")
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("chat template is not generation-prefix stable")
    if len(ids) > max_seq_len:
        raise ValueError("next-action example exceeds token budget; never truncate")
    if len(ids) == len(prefix_ids):
        raise ValueError("empty next-action supervision")
    return {"input_ids": ids, "labels": [-100] * len(prefix_ids) + ids[len(prefix_ids):]}
