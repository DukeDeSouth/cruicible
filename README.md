# Cruicible

**An agent that grades its own homework — and catches its own judge lying.**

Built with [Google ADK](https://google.github.io/adk-docs/), [Gemini 2.5 Flash](https://ai.google.dev/), and [Arize Phoenix](https://phoenix.arize.com/) for the Google Cloud Rapid Agent Hackathon (Arize track).

**Live demo:** [cruicible.cc](https://cruicible.cc)

## What it does

Cruicible is a self-evaluating research agent. Give it a question, and it will:

1. **Plan** — break the question into research steps
2. **Search** — query the web via Google Search
3. **Draft** — synthesize findings into a sourced brief
4. **Evaluate** — grade each section with a mixed rubric (code checks + LLM judge)
5. **Replan** — if a section fails evaluation, revise and retry
6. **Calibrate** — detect when the LLM judge inflates scores and override with code evidence

The final brief goes through **human review** before publishing, and all approved briefs are stored in a persistent **CruicibleWiki** knowledge base.

## What makes it different

Every "self-reflecting" agent trusts its own LLM judge. Cruicible doesn't.

When the LLM judge claims high faithfulness but code evaluators find zero real citations, Cruicible fires a **calibration override** — catching its own judge hallucinating source support. This is meta-verification: verifying the verifier.

It also learns from its mistakes. After each session, Cruicible can analyze its past failures from Phoenix traces and update its evaluation rules — measurably improving pass rates across runs (+20% in our experiments).

## Features

- **PAER Loop** — Plan → Act → Evaluate → Replan with full Phoenix tracing
- **Self-Calibrating Judge** — catches LLM-judge self-attribution bias in real-time
- **Cross-Run Self-Improvement** — learns from past failures via Phoenix span annotations
- **Before/After Experiments** — proves improvement with real Google Search, not mocks
- **Human Gate** — approve or reject briefs before they publish
- **Re-Verification** — re-check old briefs against fresh sources (claims can rot)
- **CruicibleWiki** — persistent knowledge base of approved research at `/wiki`

## Quick start

```bash
git clone https://github.com/DukeDeSouth/cruicible.git
cd cruicible
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --port 8080
```

Open [localhost:8080](http://localhost:8080), enter your API keys in Settings, and ask a question.

## API Keys

Cruicible needs two keys, configured through the web UI (Settings button):

- **Google Gemini API Key** — from [aistudio.google.com](https://aistudio.google.com/apikey)
- **Phoenix API Key** — from [app.phoenix.arize.com](https://app.phoenix.arize.com/)

Keys are stored in your browser's localStorage and sent directly to Google/Arize APIs. They never touch our servers.

## Architecture

```
User → Web UI (Tailwind + Alpine.js)
         → FastAPI Server (SSE streaming)
            → ADK Agent (Gemini 2.5 Flash)
               ├── Google Search (grounded web search)
               ├── plan_research (step decomposition)
               ├── draft_section (content synthesis)
               ├── self_evaluate (mixed rubric: code + LLM judge + calibrator)
               ├── replan_step (failure recovery)
               ├── save_pattern (Phoenix datasets)
               └── improve_from_history (cross-run learning)
                      ↓
               Phoenix Cloud (traces, annotations, experiments)
```

## Tech stack

- **Agent framework:** Google ADK (Python)
- **Model:** Gemini 2.5 Flash
- **Observability:** Arize Phoenix (OpenTelemetry traces, span annotations, datasets)
- **Backend:** FastAPI + Uvicorn, SSE streaming
- **Frontend:** Tailwind CSS + Alpine.js (single HTML file, no build step)
- **Hosting:** DigitalOcean + Cloudflare (cruicible.cc)

## License

Apache-2.0 — see [LICENSE](LICENSE).
