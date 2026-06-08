"""Replan step — fix a failed step based on evaluation feedback."""

import json
from cruicible.llm import generate


def replan_step(step_id: int, original_action: str, evaluation: dict) -> dict:
    """Generate a revised action plan for a step that failed evaluation.

    Args:
        step_id: The step number that failed.
        original_action: What was originally planned.
        evaluation: The evaluation result with scores and explanation.

    Returns:
        A dict with 'step_id', 'revised_action', and 'revised_search_query'.
    """
    prompt = f"""A research step failed quality evaluation. Suggest a revised approach.

Original action: {original_action}
Evaluation scores: {json.dumps(evaluation.get('scores', {}))}
Failure reason: {evaluation.get('explanation', 'Unknown')}

Respond with JSON: {{
  "revised_action": "<what to do differently>",
  "revised_search_query": "<better search query>"
}}
Only output valid JSON, no markdown."""

    try:
        text = generate(prompt)
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
    except Exception:
        result = {
            "revised_action": f"{original_action} — add more sources and citations",
            "revised_search_query": f"{original_action} detailed analysis with sources",
        }

    result["step_id"] = step_id
    return result
