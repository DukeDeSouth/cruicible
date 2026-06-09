"""Draft section — write a research section from search results."""

import re
from cruicible.llm import generate
from cruicible.harness import harness_prompt_block


MAX_SEARCH_INPUT = 3000
MAX_CONTENT_OUTPUT = 1500


def draft_section(step_id: int, action: str, search_results: str) -> dict:
    """Draft a research section based on search results.

    Args:
        step_id: The step number from the research plan.
        action: What this section should cover.
        search_results: Raw search results text to synthesize.

    Returns:
        A dict with 'step_id', 'content' (markdown), and 'sources' (list of URLs).
    """
    if len(search_results) > MAX_SEARCH_INPUT:
        search_results = search_results[:MAX_SEARCH_INPUT] + "\n[...truncated]"

    learned = harness_prompt_block()
    prompt = f"""You are a research writer producing a section for an analytical brief.

Task: {action}

Search results to synthesize:
{search_results}

STRICT REQUIREMENTS — your section will be evaluated against these:
1. Write 2-4 paragraphs of clear, factual prose
2. EVERY factual claim MUST cite its source inline as [Source Name](URL)
3. Extract ALL URLs from the search results and use them as citations
4. If a search result has no URL, cite it as (Source: <name of the source>)
5. NEVER make claims without a source reference
6. NEVER use vague phrases like "studies show", "experts say", "it is widely believed"
7. Be specific — use names, dates, numbers from the sources
8. List all source URLs at the end under "Sources:"

If the search results don't contain enough information, say so explicitly rather than fabricating.

{learned}

Respond with the section content in markdown."""

    content = generate(prompt)

    sources = []
    urls = re.findall(r'\(https?://[^\)]+\)', content)
    for url in urls:
        sources.append(url.strip("()"))

    source_tags = re.findall(r'\(Source:\s*([^\)]+)\)', content)
    for tag in source_tags:
        sources.append(tag.strip())

    if len(content) > MAX_CONTENT_OUTPUT:
        content = content[:MAX_CONTENT_OUTPUT] + "\n[...section truncated for context efficiency]"

    return {
        "step_id": step_id,
        "content": content,
        "sources": sources,
    }
