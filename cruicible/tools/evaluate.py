"""Self-evaluate — external grounded verification of drafted content.

Uses a mixed rubric:
  1. Code eval: source_references (regex — URLs or [Name](link) patterns)
  2. Code eval: specificity (no weasel phrases)
  3. LLM-as-judge: faithfulness (Gemini)
  4. LLM-as-judge: completeness (Gemini)

Eval results are logged as Phoenix span annotations when PHOENIX_API_KEY is set.
"""

import os
import re
import json
import logging
from cruicible.llm import generate

logger = logging.getLogger(__name__)

PASS_THRESHOLD = 0.7

WEASEL_PHRASES = [
    "it is widely believed",
    "many experts say",
    "studies show",
    "research suggests",
    "it is known that",
    "sources indicate",
    "according to reports",
]


def _check_source_refs(content: str) -> tuple[float, str]:
    """Check for source references — URLs, [Name](link), or (Source: ...) patterns."""
    url_pattern = r'\[.*?\]\(https?://[^\)]+\)'
    source_tag = r'\(Source:\s*[^\)]+\)'
    named_ref = r'\[[\w\s\.]+\]'

    url_refs = re.findall(url_pattern, content)
    source_refs = re.findall(source_tag, content)
    named_refs = re.findall(named_ref, content)

    total_refs = len(url_refs) + len(source_refs)

    paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 50]
    if not paragraphs:
        return 0.0, "No substantial paragraphs found"

    if total_refs >= len(paragraphs):
        return 1.0, f"{total_refs} source references for {len(paragraphs)} paragraphs"
    elif total_refs > 0:
        ratio = total_refs / max(len(paragraphs), 1)
        return round(ratio, 2), f"Only {total_refs} refs for {len(paragraphs)} paragraphs — need more"
    elif named_refs:
        return 0.3, f"Found {len(named_refs)} named refs but no URLs or source tags"
    else:
        return 0.0, "No source references found"


def _check_specificity(content: str) -> tuple[float, str]:
    """Check that content doesn't rely on weasel phrases."""
    found = []
    lower = content.lower()
    for phrase in WEASEL_PHRASES:
        if phrase in lower:
            found.append(phrase)

    if not found:
        return 1.0, "No weasel phrases detected"
    elif len(found) <= 2:
        return 0.6, f"Weasel phrases found: {', '.join(found)}"
    else:
        return 0.2, f"Too many vague claims: {', '.join(found)}"


def _log_to_phoenix(step_id: int, scores: dict, passed: bool, explanation: str) -> None:
    """Log evaluation results as Phoenix span annotations (best-effort)."""
    try:
        api_key = os.environ.get("PHOENIX_API_KEY", "")
        endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "")
        if not api_key or not endpoint:
            return

        from phoenix.client import Client

        client = Client(endpoint=endpoint, headers={"api_key": api_key})

        for metric_name, score_val in scores.items():
            label = "pass" if score_val >= PASS_THRESHOLD else "fail"
            try:
                client.annotations.add_annotation(
                    annotation_name=f"cruicible_{metric_name}",
                    annotator_kind="LLM",
                    label=label,
                    score=score_val,
                    explanation=f"Step {step_id}: {explanation}",
                )
            except Exception:
                pass

        logger.info("Eval annotations logged to Phoenix for step %d", step_id)
    except Exception as e:
        logger.debug("Phoenix annotation skipped: %s", e)


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

    # 1. Code eval: source references
    ref_score, ref_reason = _check_source_refs(content)
    scores["source_references"] = ref_score
    if ref_score < PASS_THRESHOLD:
        explanations.append(f"Source refs: {ref_reason}")

    # 2. Code eval: specificity
    spec_score, spec_reason = _check_specificity(content)
    scores["specificity"] = spec_score
    if spec_score < PASS_THRESHOLD:
        explanations.append(f"Specificity: {spec_reason}")

    # 3. LLM-as-judge: faithfulness
    faith_prompt = f"""You are a strict fact-checking judge. Evaluate whether EVERY factual claim
in the DRAFT is directly supported by the provided SOURCES.

DRAFT:
{content}

SOURCES:
{chr(10).join(sources) if sources else "(no sources provided)"}

Be strict:
- Claims without clear source backing = unsupported
- Vague generalizations without evidence = unsupported
- Score 0.0 if most claims lack support

Score from 0.0 to 1.0.
Respond with JSON only: {{"score": <float>, "reason": "<brief explanation>"}}"""

    try:
        faith_text = generate(faith_prompt)
        if faith_text.startswith("```"):
            faith_text = faith_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        faith_data = json.loads(faith_text)
        scores["faithfulness"] = float(faith_data.get("score", 0.5))
        if scores["faithfulness"] < PASS_THRESHOLD:
            explanations.append(f"Faithfulness: {faith_data.get('reason', 'N/A')}")
    except Exception:
        scores["faithfulness"] = 0.5
        explanations.append("Faithfulness check failed — defaulting to 0.5")

    # 4. LLM-as-judge: completeness
    comp_prompt = f"""You are a strict research quality judge. Evaluate whether the DRAFT
adequately covers ALL aspects of the research task.

TASK: {query}
DRAFT:
{content}

Be strict:
- Missing key aspects = incomplete
- Superficial treatment = partial
- Score 0.0 if mostly incomplete

Score from 0.0 to 1.0.
Respond with JSON only: {{"score": <float>, "reason": "<brief explanation>"}}"""

    try:
        comp_text = generate(comp_prompt)
        if comp_text.startswith("```"):
            comp_text = comp_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        comp_data = json.loads(comp_text)
        scores["completeness"] = float(comp_data.get("score", 0.5))
        if scores["completeness"] < PASS_THRESHOLD:
            explanations.append(f"Completeness: {comp_data.get('reason', 'N/A')}")
    except Exception:
        scores["completeness"] = 0.5
        explanations.append("Completeness check failed — defaulting to 0.5")

    avg_score = sum(scores.values()) / len(scores)
    passed = avg_score >= PASS_THRESHOLD and scores["source_references"] > 0

    explanation = "; ".join(explanations) if explanations else "All checks passed"

    _log_to_phoenix(step_id, scores, passed, explanation)

    return {
        "passed": passed,
        "scores": scores,
        "avg_score": round(avg_score, 3),
        "explanation": explanation,
        "step_id": step_id,
    }
