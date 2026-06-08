"""Self-evaluate — external grounded verification of drafted content.

Uses a mixed rubric:
  1. Code eval: has_citations (regex)
  2. LLM-as-judge: faithfulness (Gemini direct call)
  3. LLM-as-judge: completeness (Gemini direct call)

All evaluation calls are traced to Phoenix via OpenInference.
"""

import re
import json
from cruicible.llm import generate


PASS_THRESHOLD = 0.7


def self_evaluate(step_id: int, content: str, sources: list[str], query: str) -> dict:
    """Evaluate a drafted section against quality rubric.

    Args:
        step_id: The step number being evaluated.
        content: The drafted markdown content.
        sources: List of source URLs used.
        query: The original research query for context.

    Returns:
        A dict with 'passed' (bool), 'scores' (dict), 'explanation', 'step_id'.
    """
    scores: dict[str, float] = {}
    explanations: list[str] = []

    # 1. Code eval: has_citations
    citation_pattern = r'\[.*?\]\(https?://[^\)]+\)'
    citations_found = re.findall(citation_pattern, content)
    scores["has_citations"] = 1.0 if len(citations_found) > 0 else 0.0
    if scores["has_citations"] == 0.0:
        explanations.append("No inline citations found")

    # 2. LLM-as-judge: faithfulness
    faith_prompt = f"""You are an impartial judge. Evaluate whether EVERY factual claim
in the DRAFT is supported by the provided SOURCES.

DRAFT:
{content}

SOURCES:
{chr(10).join(sources) if sources else "(no sources provided)"}

Score from 0.0 to 1.0 where:
- 1.0 = every claim is well-supported
- 0.5 = some claims lack support
- 0.0 = mostly unsupported

Respond with JSON: {{"score": <float>, "reason": "<brief explanation>"}}
Only output valid JSON, no markdown."""

    try:
        faith_text = generate(faith_prompt)
        if faith_text.startswith("```"):
            faith_text = faith_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        faith_data = json.loads(faith_text)
        scores["faithfulness"] = float(faith_data.get("score", 0.5))
        if scores["faithfulness"] < PASS_THRESHOLD:
            explanations.append(f"Faithfulness low: {faith_data.get('reason', 'N/A')}")
    except Exception:
        scores["faithfulness"] = 0.5
        explanations.append("Faithfulness check failed — defaulting to 0.5")

    # 3. LLM-as-judge: completeness
    comp_prompt = f"""You are an impartial judge. Evaluate whether the DRAFT adequately
covers the research task.

TASK: {query}
DRAFT:
{content}

Score from 0.0 to 1.0 where:
- 1.0 = comprehensive, covers all key aspects
- 0.5 = partial coverage
- 0.0 = mostly incomplete

Respond with JSON: {{"score": <float>, "reason": "<brief explanation>"}}
Only output valid JSON, no markdown."""

    try:
        comp_text = generate(comp_prompt)
        if comp_text.startswith("```"):
            comp_text = comp_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        comp_data = json.loads(comp_text)
        scores["completeness"] = float(comp_data.get("score", 0.5))
        if scores["completeness"] < PASS_THRESHOLD:
            explanations.append(f"Completeness low: {comp_data.get('reason', 'N/A')}")
    except Exception:
        scores["completeness"] = 0.5
        explanations.append("Completeness check failed — defaulting to 0.5")

    avg_score = sum(scores.values()) / len(scores)
    passed = avg_score >= PASS_THRESHOLD and scores["has_citations"] > 0

    return {
        "passed": passed,
        "scores": scores,
        "avg_score": round(avg_score, 3),
        "explanation": "; ".join(explanations) if explanations else "All checks passed",
        "step_id": step_id,
    }
