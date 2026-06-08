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
    prompt = f"""You are a research planner. Given a query, produce a JSON array of 3-5 steps.
Each step has: "id" (int), "action" (string), "search_query" (string for Google search).

Query: {query}

Respond ONLY with valid JSON array, no markdown fences."""

    text = generate(prompt)
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        steps = json.loads(text)
    except json.JSONDecodeError:
        steps = [
            {"id": 1, "action": "Search for main topic", "search_query": query},
            {"id": 2, "action": "Find supporting data", "search_query": f"{query} statistics data"},
            {"id": 3, "action": "Find expert opinions", "search_query": f"{query} expert analysis"},
        ]

    return {"steps": steps, "query": query}
