"""Dynamic harness — externally stored rules and few-shot examples.

Loaded by draft.py to inject learned rules into the drafting prompt.
Updated by improve.py after analyzing past eval failures from Phoenix.
Stored in data/harness.json (external state, not in agent context — avoids confabulation).
"""

import json
import os
import logging

logger = logging.getLogger(__name__)

HARNESS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "harness.json")

_DEFAULT = {"version": 1, "few_shot": [], "rules": [], "updated_from_patterns": []}


def load_harness() -> dict:
    try:
        with open(HARNESS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(_DEFAULT)


def save_harness(h: dict) -> None:
    os.makedirs(os.path.dirname(HARNESS_PATH), exist_ok=True)
    with open(HARNESS_PATH, "w") as f:
        json.dump(h, f, indent=2)


def clear_harness() -> None:
    """Reset harness to empty default state."""
    save_harness(dict(_DEFAULT))
    logger.info("Harness cleared to default (v1, no rules)")


def backup_harness() -> dict:
    """Return a deep copy of current harness state."""
    import copy
    return copy.deepcopy(load_harness())


def restore_harness(backup: dict) -> None:
    """Restore harness from a previous backup."""
    save_harness(backup)
    logger.info("Harness restored to v%d", backup.get("version", 1))


def harness_prompt_block() -> str:
    """Return prompt injection block from learned rules/exemplars. Empty if no harness."""
    h = load_harness()
    parts = []
    rules = h.get("rules", [])
    if rules:
        parts.append(
            "LEARNED RULES (derived from your past failures — follow strictly):\n"
            + "\n".join(f"- {r}" for r in rules[:5])
        )
    few_shot = h.get("few_shot", [])
    if few_shot:
        exemplars = [e.get("exemplar", "") for e in few_shot[:2] if e.get("exemplar")]
        if exemplars:
            parts.append("EXEMPLARS of past high-quality sections:\n" + "\n---\n".join(exemplars))
    return "\n\n".join(parts)
