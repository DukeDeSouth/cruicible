# Cruicible Demo — 3-Minute Video Script

## Voiceover Script (for AI TTS or narration)

**[0:00–0:15] HOOK — Title card / intro animation**

"Every AI agent today judges its own work. The problem? LLM judges hallucinate just like the agents they evaluate. Nobody's watching the watchmen. Until now."

**[0:15–0:25] INTRO — Show cruicible.cc loading**

"This is Cruicible — a self-evaluating research agent that doesn't trust its own judge."

**[0:25–0:55] PAER LOOP — Type query, show pipeline running**

"Watch what happens when we ask it a research question. The agent plans its approach, searches the web, drafts a section — then evaluates its own work with a mix of code checks and an LLM judge. When a section fails, it replans and tries again. Every step is traced to Arize Phoenix."

**[0:55–1:25] SELF-EVAL — Show eval scores appearing, step failing and retrying**

"The evaluation isn't just vibes. Code evaluators check for real citations and specificity. The LLM judge checks faithfulness and completeness. If anything falls below threshold — the section gets rewritten."

**[1:25–1:55] CALIBRATION OVERRIDE — Show the red override block**

"But here's what no other agent does. When the LLM judge claims high faithfulness — but the code evaluator finds weak citations — Cruicible fires a calibration override. It caught its own judge inflating a score. The original score gets slashed and the evidence is logged. This is meta-verification — verifying the verifier."

**[1:55–2:15] HUMAN GATE — Show approve/reject**

"Before any brief is published, a human reviews it. Approve or reject. The agent works for you, not the other way around."

**[2:15–2:35] SELF-IMPROVEMENT — Show before/after or Learn from History**

"Cruicible also learns across sessions. It reads its past failures from Phoenix, extracts patterns, and updates its evaluation rules. In our experiments: baseline 80% pass rate, improved to 100% — with real Google Search, not synthetic data."

**[2:35–2:50] WIKI + RE-VERIFY — Quick flash of wiki and re-verify**

"Approved briefs are stored in CruicibleWiki — a growing knowledge base. And any old brief can be re-verified against fresh sources, because facts rot over time."

**[2:50–3:00] CLOSING — Final card**

"Cruicible. The agent that catches its own judge lying. Built with Google ADK, Gemini, and Arize Phoenix. Try it at cruicible.cc."

---

## Recording Tips

1. Use a simple, factual query for the main demo (e.g. "What percentage of global electricity is used for Bitcoin mining?")
2. Speed up slow parts (search waiting) to 2-4x, keep money-shot moments at 1x
3. Make sure the browser window is clean — no other tabs visible, no bookmarks bar
4. Record at 1080p or higher
5. For the calibration override to fire, you may need to try a few queries — controversial statistics work best
