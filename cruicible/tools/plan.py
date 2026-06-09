"""Plan research — decompose a user query into concrete steps."""

import json
from cruicible.llm import generate


def plan_research(query: str) -> dict:
    """Decompose a research query into 3-5 concrete investigation steps.

    Args:
        query: The user's research question or topic.

    Returns:
        A dict with 'steps' (list of step objects) and 'query' echo.
    """
    prompt = f"""You are a research planner. Given a query, produce a JSON array of exactly 2 steps.
Each step has: "id" (int), "action" (string), "search_query" (string for Google search).
Step 1 should gather core facts. Step 2 should add depth or a second angle.

Query: {query}

Respond ONLY with valid JSON array, no markdown fences."""

    text = generate(prompt)
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        steps = json.loads(text)
        if len(steps) > 2:
            steps = steps[:2]
    except json.JSONDecodeError:
        steps = [
            {"id": 1, "action": f"Research core facts about: {query}", "search_query": query},
            {"id": 2, "action": f"Find additional context", "search_query": f"{query} analysis details"},
        ]

    return {"steps": steps, "query": query}
