"""Cross-run self-improvement — read past eval failures, derive harness updates.

This is the core Level A differentiator: the agent reads its OWN history
from Phoenix, finds failure patterns, and updates its harness to prevent
the same failures in future runs. Empirical before/after evidence via
Phoenix experiments.
"""

import os
import time
import logging
from collections import Counter

from cruicible.harness import (
    load_harness, save_harness, clear_harness, backup_harness, restore_harness,
)

logger = logging.getLogger(__name__)

_RULE_MAP = {
    "cruicible_source_references": (
        "Extract EVERY URL from search results and cite each claim inline as [Name](URL)."
    ),
    "cruicible_specificity": (
        "Replace any general statement with a named source plus a specific number or date."
    ),
    "cruicible_completeness": (
        "Explicitly cover every sub-question implied by the research step action."
    ),
    "cruicible_faithfulness": (
        "State only facts present verbatim in the search results; never extrapolate."
    ),
}

EXPERIMENT_QUERIES = [
    "What are the latest developments in nuclear fusion energy as of 2025?",
    "How does Apple Vision Pro compare to Meta Quest 3?",
    "What is the current state of the US national debt?",
    "What are the health benefits and risks of intermittent fasting?",
    "How is AI being used in drug discovery?",
]


def _phoenix_client():
    api_key = os.environ.get("PHOENIX_API_KEY", "")
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "")
    if not api_key or not endpoint:
        return None
    from phoenix.client import Client
    return Client(base_url=endpoint, api_key=api_key)


def improve_from_history(project: str = "cruicible") -> dict:
    """Read own past eval failures from Phoenix, find patterns, update harness.

    Args:
        project: Phoenix project name to read spans from.

    Returns:
        Dict with 'updated' (bool), 'patterns', 'new_rules', 'version'.
    """
    client = _phoenix_client()
    if client is None:
        return {"updated": False, "reason": "Phoenix not configured"}

    try:
        spans_df = client.spans.get_spans_dataframe(
            project_identifier=project,
            limit=100,
            timeout=30,
        )
        if spans_df is None or spans_df.empty:
            return {"updated": False, "reason": "No spans found yet — run the agent first"}

        annotations_df = client.spans.get_span_annotations_dataframe(
            spans_dataframe=spans_df,
            project_identifier=project,
        )
        if annotations_df is None or annotations_df.empty:
            return {"updated": False, "reason": "No annotations found — run agent with eval first"}

        logger.info("Annotations DataFrame: %d rows, columns=%s", len(annotations_df), list(annotations_df.columns))
        if len(annotations_df) > 0:
            logger.info("Sample row: %s", annotations_df.iloc[0].to_dict())

        fail_counter: Counter = Counter()
        sample_reasons: dict[str, str] = {}

        for _, row in annotations_df.iterrows():
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
            label = row_dict.get("label", row_dict.get("result.label", ""))
            name = row_dict.get("annotation_name", row_dict.get("name", ""))
            score = row_dict.get("score", row_dict.get("result.score", None))

            if not name:
                for k, v in row_dict.items():
                    if "name" in k.lower() and v:
                        name = str(v)
                        break

            if label == "fail" and name in _RULE_MAP:
                fail_counter[name] += 1
                if name not in sample_reasons:
                    expl = row_dict.get("explanation", row_dict.get("result.explanation", ""))
                    sample_reasons[name] = str(expl)[:200]
            elif score is not None and float(score) < 0.7 and name in _RULE_MAP:
                fail_counter[name] += 1
                if name not in sample_reasons:
                    sample_reasons[name] = f"score={score}"

        if not fail_counter:
            return {"updated": False, "reason": "No failures in history — harness is solid"}

        patterns = [
            {"metric": n, "count": c, "sample_reason": sample_reasons.get(n, "")}
            for n, c in fail_counter.most_common()
        ]
        new_rules = [_RULE_MAP[p["metric"]] for p in patterns if p["metric"] in _RULE_MAP]

        h = load_harness()
        existing = set(h.get("rules", []))
        added = [r for r in new_rules if r not in existing]
        h["rules"] = sorted(existing | set(new_rules))
        h["updated_from_patterns"] = patterns
        h["version"] = h.get("version", 1) + 1
        save_harness(h)

        logger.info("Harness updated to v%d with %d new rules from %d failure patterns",
                     h["version"], len(added), len(patterns))
        return {
            "updated": True,
            "version": h["version"],
            "patterns": patterns,
            "new_rules": added,
            "total_rules": len(h["rules"]),
        }

    except Exception as e:
        logger.warning("improve_from_history failed: %s", e)
        return {"updated": False, "reason": str(e)}


def _search_and_draft(query: str, step_id: int) -> dict:
    """Run a single research step: Gemini grounded search → draft → evaluate."""
    from cruicible.llm import get_client
    from cruicible.tools.draft import draft_section
    from cruicible.tools.evaluate import self_evaluate

    client = get_client()
    try:
        search_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Search the web and provide a detailed factual summary with source URLs for: {query}",
            config={"tools": [{"google_search": {}}]},
        )
        search_results = search_response.text.strip()
    except Exception as e:
        logger.warning("Grounded search failed for '%s': %s — using plain generate", query[:50], e)
        from cruicible.llm import generate
        search_results = generate(
            f"Provide a detailed factual summary about: {query}. Include specific facts, dates, and numbers."
        )

    draft = draft_section(step_id, query, search_results)
    sources_for_eval = [search_results[:2000]] + draft["sources"]
    evaluation = self_evaluate(step_id, draft["content"], sources_for_eval, query)

    return {
        "query": query,
        "passed": evaluation["passed"],
        "avg_score": evaluation["avg_score"],
        "scores": evaluation["scores"],
    }


def run_improvement_experiment(
    queries: list[str] | None = None,
    project: str = "cruicible",
    progress_callback=None,
) -> dict:
    """Run a before/after experiment: baseline (no rules) vs improved (with learned rules).

    1. Backup current harness
    2. Clear harness → run baseline
    3. improve_from_history() → populate harness
    4. Run improved with harness rules
    5. Compute delta
    6. Restore original harness

    Args:
        queries: Test queries. Defaults to EXPERIMENT_QUERIES.
        project: Phoenix project name.
        progress_callback: Optional fn(step, total, message) for UI updates.

    Returns:
        Dict with baseline_rate, improved_rate, delta, details, compare_url.
    """
    test_queries = (queries or EXPERIMENT_QUERIES)[:5]
    total_steps = len(test_queries) * 2 + 2  # queries×2 + improve + finalize
    step = 0

    def _progress(msg):
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(step, total_steps, msg)
        logger.info("[Experiment %d/%d] %s", step, total_steps, msg)

    original_harness = backup_harness()

    try:
        # Phase 1: Baseline (empty harness)
        clear_harness()
        baseline_results = []
        for i, q in enumerate(test_queries):
            _progress(f"Baseline: {q[:40]}...")
            try:
                result = _search_and_draft(q, step_id=i + 1)
                baseline_results.append(result)
            except Exception as e:
                logger.warning("Baseline query failed: %s", e)
                baseline_results.append({"query": q, "passed": False, "avg_score": 0.0, "scores": {}})

        baseline_passed = sum(1 for r in baseline_results if r["passed"])
        baseline_rate = baseline_passed / len(baseline_results) if baseline_results else 0

        # Phase 2: Learn from history
        _progress("Learning from Phoenix history...")
        improve_result = improve_from_history(project)

        if not improve_result.get("updated"):
            logger.warning("No patterns learned — improved run will use same empty harness")

        # Phase 3: Improved (with learned rules)
        improved_results = []
        for i, q in enumerate(test_queries):
            _progress(f"Improved: {q[:40]}...")
            try:
                result = _search_and_draft(q, step_id=100 + i + 1)
                improved_results.append(result)
            except Exception as e:
                logger.warning("Improved query failed: %s", e)
                improved_results.append({"query": q, "passed": False, "avg_score": 0.0, "scores": {}})

        improved_passed = sum(1 for r in improved_results if r["passed"])
        improved_rate = improved_passed / len(improved_results) if improved_results else 0

        _progress("Computing results...")

        baseline_avg = sum(r["avg_score"] for r in baseline_results) / len(baseline_results) if baseline_results else 0
        improved_avg = sum(r["avg_score"] for r in improved_results) / len(improved_results) if improved_results else 0

        result = {
            "ran": True,
            "baseline_pass_rate": round(baseline_rate * 100, 1),
            "improved_pass_rate": round(improved_rate * 100, 1),
            "delta": round((improved_rate - baseline_rate) * 100, 1),
            "baseline_avg_score": round(baseline_avg, 3),
            "improved_avg_score": round(improved_avg, 3),
            "score_delta": round(improved_avg - baseline_avg, 3),
            "queries_used": len(test_queries),
            "rules_learned": improve_result.get("new_rules", []),
            "patterns_found": improve_result.get("patterns", []),
            "baseline_details": baseline_results,
            "improved_details": improved_results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        logger.info(
            "Experiment complete: baseline=%.1f%% → improved=%.1f%% (Δ=%.1f%%)",
            result["baseline_pass_rate"], result["improved_pass_rate"], result["delta"],
        )
        return result

    except Exception as e:
        logger.error("Experiment failed: %s", e)
        return {"ran": False, "reason": str(e)}

    finally:
        restore_harness(original_harness)
