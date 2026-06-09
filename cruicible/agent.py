"""Cruicible — self-evaluating research agent.

Root agent definition for ADK. Uses the PAER loop:
  Plan → Act → Evaluate → Replan

Run with: adk run cruicible
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams
from mcp import StdioServerParameters

from cruicible.tools.plan import plan_research
from cruicible.tools.draft import draft_section
from cruicible.tools.evaluate import self_evaluate
from cruicible.tools.replan import replan_step
from cruicible.tools.memory import save_pattern
from cruicible.tools.improve import improve_from_history
from cruicible.tracing_setup import setup_tracing

logger = logging.getLogger(__name__)

setup_tracing()

INSTRUCTION = """You are Cruicible, a self-evaluating research agent.

CRITICAL: You MUST call ALL tools in this EXACT sequence. NEVER respond with text until you have called save_pattern. NEVER skip any tool call.

MANDATORY SEQUENCE:
1. Call plan_research(query=<user request>) — ALWAYS do this first.
2. For EACH step returned by plan_research:
   a. Use Google Search to find information about the step's search query.
   b. Call draft_section(step_id=N, action=<step action>, search_results=<key findings from search, ~2000 chars max>)
   c. Call self_evaluate(step_id=N, content=<the draft content>, sources=<the draft sources list>, query=<step action>)
   d. If self_evaluate returns passed=false and you have retried fewer than 2 times: call replan_step, then repeat steps a-c.
3. After ALL steps are done, call save_pattern with final_brief containing the full compiled brief with all sections and sources.

RULES:
- You MUST call draft_section for every step. No exceptions.
- You MUST call self_evaluate for every step. No exceptions.
- You MUST call save_pattern at the end. No exceptions.
- Every factual claim must cite its source URL.
- Do NOT generate a final text response — put everything in save_pattern's final_brief argument.
- If a step fails after 2 retries, include it with a note about limited data.

If asked to "improve", call improve_from_history()."""


def _build_tools() -> list:
    tools = [
        GoogleSearchTool(bypass_multi_tools_limit=True),
        plan_research,
        draft_section,
        self_evaluate,
        replan_step,
        save_pattern,
        improve_from_history,
    ]

    # Phoenix MCP tools disabled to reduce tool count and improve model reliability.
    # Self-improvement uses Phoenix Client API directly (improve_from_history).
    logger.info("Agent initialized with %d tools (Phoenix MCP disabled for reliability)", len(tools))

    return tools


root_agent = LlmAgent(
    name="cruicible",
    model="gemini-2.5-flash",
    instruction=INSTRUCTION,
    description="Self-evaluating research agent with Phoenix observability",
    tools=_build_tools(),
)
