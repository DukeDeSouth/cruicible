"""Save pattern — store successful research examples as few-shot memory.

In MVP, patterns are stored locally as JSON. When Phoenix experiments
are integrated (D2), they'll also be pushed to Phoenix datasets.
"""

import json
import os
import time

PATTERNS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "patterns")


def save_pattern(query: str, final_brief: str, scores: dict) -> dict:
    """Save a successful research pattern for future few-shot reference.

    Args:
        query: The original research query.
        final_brief: The completed research brief content.
        scores: Aggregated evaluation scores.

    Returns:
        A dict with 'saved' (bool) and 'pattern_id'.
    """
    os.makedirs(PATTERNS_DIR, exist_ok=True)

    pattern_id = f"pattern_{int(time.time())}"
    pattern = {
        "id": pattern_id,
        "query": query,
        "brief_preview": final_brief[:500],
        "scores": scores,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    filepath = os.path.join(PATTERNS_DIR, f"{pattern_id}.json")
    with open(filepath, "w") as f:
        json.dump(pattern, f, indent=2)

    return {"saved": True, "pattern_id": pattern_id, "path": filepath}
