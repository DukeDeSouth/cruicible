"""Re-verification — check previously issued claims against fresh sources.

Uses Gemini grounded search to re-verify each factual claim extracted
from a completed research brief. Returns per-claim verdicts:
CONFIRMED, OUTDATED, UNCONFIRMED, or ERROR.
"""

import json
import logging
import time

from cruicible.llm import generate, get_client

logger = logging.getLogger(__name__)

MAX_CLAIMS = 10


def extract_claims(brief_text: str) -> list[dict]:
    """Use Gemini to extract verifiable factual claims from a brief."""
    prompt = f"""Extract up to {MAX_CLAIMS} specific, verifiable factual claims from this research brief.
Each claim should be a single sentence stating one fact that can be checked.
Skip opinions, hedged statements, and meta-commentary.

Brief:
{brief_text[:3000]}

Return JSON array: [{{"claim": "...", "category": "statistic|date|name|event|other"}}]
Return ONLY the JSON array, no markdown."""

    try:
        raw = generate(prompt)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        claims = json.loads(raw)
        if isinstance(claims, list):
            return claims[:MAX_CLAIMS]
    except Exception as e:
        logger.warning("Claim extraction failed: %s", e)

    return []


def _verify_single_claim(claim_text: str) -> dict:
    """Verify one claim via Gemini grounded search."""
    client = get_client()
    search_prompt = f"Verify this claim with current sources: {claim_text}"

    try:
        from google.genai import types as genai_types

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=search_prompt,
            config=genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                temperature=0.1,
            ),
        )
        search_text = response.text or ""
    except Exception as e:
        logger.warning("Search failed for claim: %s", e)
        return {"status": "error", "reason": str(e)[:200]}

    verdict_prompt = f"""You are a fact-checker. Given a CLAIM and SEARCH RESULTS, determine the verdict.

CLAIM: {claim_text}

SEARCH RESULTS:
{search_text[:2000]}

Respond with exactly one JSON object:
{{"verdict": "CONFIRMED|OUTDATED|UNCONFIRMED", "confidence": 0.0-1.0, "evidence": "one sentence explaining why"}}

- CONFIRMED: search results clearly support the claim
- OUTDATED: claim was once true but newer info contradicts/updates it
- UNCONFIRMED: search results don't clearly support or deny the claim"""

    try:
        raw = generate(verdict_prompt)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning("Verdict parse failed: %s", e)
        return {"verdict": "ERROR", "confidence": 0.0, "evidence": f"Parse error: {e}"}


def reverify_claims(claims: list[dict], progress_cb=None) -> list[dict]:
    """Re-verify a list of claims. Returns enriched claim dicts with verdicts."""
    results = []
    total = len(claims)

    for i, claim in enumerate(claims):
        if progress_cb:
            progress_cb(i + 1, total, f"Verifying: {claim['claim'][:60]}...")

        verdict = _verify_single_claim(claim["claim"])
        results.append({
            **claim,
            **verdict,
        })
        time.sleep(0.5)

    return results
