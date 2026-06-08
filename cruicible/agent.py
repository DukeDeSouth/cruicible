"""Cruicible — self-evaluating research agent.

Root agent definition for ADK. Uses the PAER loop:
  Plan → Act → Evaluate → Replan

Run with: adk run cruicible
"""

import os
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams

from cruicible.tools.plan import plan_research
from cruicible.tools.draft import draft_section
from cruicible.tools.evaluate import self_evaluate
from cruicible.tools.replan import replan_step
from cruicible.tools.memory import save_pattern
from cruicible.tracing_setup import setup_tracing

setup_tracing()

INSTRUCTION = """You are Cruicible, a self-evaluating research agent. Your job is to produce
reliable, well-sourced research briefs.

For every user request, follow the PAER loop (Plan-Act-Evaluate-Replan):

1. Call plan_research to decompose the request into 3-5 concrete steps.
2. For each step:
   a. Use google_search to find relevant sources based on the step's search_query.
   b. Call draft_section with the step details and search results to write content.
   c. Call self_evaluate to check quality of the drafted section.
   d. If evaluation FAILS (passed=false), call replan_step to get a revised approach,
      then re-search and re-draft. Maximum 2 retries per step.
3. After all steps pass evaluation, compile all sections into a final research brief.
4. Present the complete brief to the user with all sources cited.
5. Call save_pattern to store the successful example for future reference.

Quality standards:
- Every factual claim must have an inline citation: [Source](URL)
- Never fabricate information — only report what search results contain
- If a step keeps failing after 2 retries, note the gap honestly and move on
- Prefer recent, authoritative sources

Format the final brief with:
- A clear title
- Numbered sections matching the research plan
- A Sources section at the end listing all URLs used"""


def _build_tools() -> list:
    tools = [
        GoogleSearchTool(bypass_multi_tools_limit=True),
        plan_research,
        draft_section,
        self_evaluate,
        replan_step,
        save_pattern,
    ]

    phoenix_mcp_url = os.environ.get("PHOENIX_MCP_URL", "")
    phoenix_api_key = os.environ.get("PHOENIX_API_KEY", "")

    if phoenix_mcp_url:
        headers = {}
        if phoenix_api_key:
            headers["Authorization"] = f"Bearer {phoenix_api_key}"

        phoenix_toolset = MCPToolset(
            connection_params=SseConnectionParams(
                url=phoenix_mcp_url,
                headers=headers,
            ),
        )
        tools.append(phoenix_toolset)

    return tools


root_agent = LlmAgent(
    name="cruicible",
    model="gemini-2.5-flash",
    instruction=INSTRUCTION,
    description="Self-evaluating research agent with Phoenix observability",
    tools=_build_tools(),
)
