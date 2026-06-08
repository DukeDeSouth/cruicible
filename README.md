# Cruicible

Self-evaluating research agent built with [Google ADK](https://google.github.io/adk-docs/) and [Arize Phoenix](https://phoenix.arize.com/).

Cruicible follows the **PAER loop** — Plan, Act, Evaluate, Replan — to produce reliable, well-sourced research briefs. Every drafted section goes through external grounded verification: a mix of code-based checks and LLM-as-judge evaluation, all traced to Phoenix for full observability.

## Quick start

```bash
# Clone and set up
git clone https://github.com/DukeDeSouth/cruicible.git
cd cruicible

# Create virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY and PHOENIX_API_KEY

# Run the agent
adk run cruicible
```

## How it works

1. **Plan** — decomposes a research query into 3-5 steps
2. **Act** — searches the web and drafts each section
3. **Evaluate** — checks citations, faithfulness, completeness
4. **Replan** — if a section fails, revises the approach (max 2 retries)

Successful patterns are stored for few-shot reference in future sessions.

## Architecture

```
User → ADK Agent (Gemini 2.5 Flash)
         ├── GoogleSearchTool (web search)
         ├── plan_research (decomposition)
         ├── draft_section (synthesis)
         ├── self_evaluate (mixed rubric)
         ├── replan_step (failure recovery)
         ├── save_pattern (memory)
         └── Phoenix MCP (trace read-back)
                ↓
         Phoenix Cloud (observability)
```

## Requirements

- Python 3.11+
- Gemini API key ([AI Studio](https://aistudio.google.com/))
- Phoenix Cloud account ([app.phoenix.arize.com](https://app.phoenix.arize.com/))

## License

Apache-2.0 — see [LICENSE](LICENSE).
