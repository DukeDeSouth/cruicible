"""Save pattern — store successful research examples in Phoenix datasets.

Successful PAER loops are saved to Phoenix Cloud as dataset examples,
enabling few-shot reference for future runs. Falls back to local JSON
if Phoenix is unavailable.
"""

import json
import os
import time
import logging

logger = logging.getLogger(__name__)

PATTERNS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "patterns")


def _save_to_phoenix(pattern_id: str, query: str, brief_preview: str, scores: dict) -> bool:
    """Attempt to save pattern to Phoenix dataset. Returns True on success."""
    api_key = os.environ.get("PHOENIX_API_KEY", "")
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "")
    if not api_key or not endpoint:
        return False

    from phoenix.client import Client

    client = Client(base_url=endpoint, api_key=api_key)
    client.datasets.create_dataset(
        name=f"cruicible-pattern-{pattern_id}",
        examples=[{
            "input": {"query": query},
            "output": {"brief_preview": brief_preview},
            "metadata": {
                "scores": scores,
                "pattern_id": pattern_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }],
    )
    return True


def _save_to_local(pattern_id: str, query: str, brief_preview: str, scores: dict) -> str:
    """Fallback: save pattern as local JSON."""
    os.makedirs(PATTERNS_DIR, exist_ok=True)
    pattern = {
        "id": pattern_id,
        "query": query,
        "brief_preview": brief_preview,
        "scores": scores,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    filepath = os.path.join(PATTERNS_DIR, f"{pattern_id}.json")
    with open(filepath, "w") as f:
        json.dump(pattern, f, indent=2)
    return filepath


def save_pattern(query: str, final_brief: str, scores: dict) -> dict:
    """Save a successful research pattern for future few-shot reference.

    Args:
        query: The original research query.
        final_brief: The completed research brief content.
        scores: Aggregated evaluation scores.

    Returns:
        A dict with 'saved' (bool), 'pattern_id', and 'storage' backend.
    """
    pattern_id = f"pattern_{int(time.time())}"
    brief_preview = final_brief[:500]

    try:
        if _save_to_phoenix(pattern_id, query, brief_preview, scores):
            logger.info("Pattern %s saved to Phoenix dataset", pattern_id)
            return {"saved": True, "pattern_id": pattern_id, "storage": "phoenix"}
    except Exception as e:
        logger.warning("Phoenix dataset save failed: %s", e)

    filepath = _save_to_local(pattern_id, query, brief_preview, scores)
    logger.info("Pattern %s saved locally: %s", pattern_id, filepath)
    return {"saved": True, "pattern_id": pattern_id, "storage": "local"}
