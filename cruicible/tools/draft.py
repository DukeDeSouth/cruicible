"""Draft section — write a research section from search results."""

import re
from cruicible.llm import generate


def draft_section(step_id: int, action: str, search_results: str) -> dict:
    """Draft a research section based on search results.

    Args:
        step_id: The step number from the research plan.
        action: What this section should cover.
        search_results: Raw search results text to synthesize.

    Returns:
        A dict with 'step_id', 'content' (markdown), and 'sources' (list of URLs).
    """
    prompt = f"""You are a research writer. Write a concise section for a research brief.

Task: {action}
Search results:
{search_results}

Requirements:
- Write 2-4 paragraphs of clear, factual prose
- Cite sources inline as [Source Title](URL)
- Extract and list all source URLs at the end
- Never fabricate information — only use what's in the search results

Respond with the section content in markdown."""

    content = generate(prompt)

    sources = []
    urls = re.findall(r'\(https?://[^\)]+\)', content)
    for url in urls:
        sources.append(url.strip("()"))

    return {
        "step_id": step_id,
        "content": content,
        "sources": sources,
    }
