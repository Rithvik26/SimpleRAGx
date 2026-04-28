# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 5. LLM Model Selection (SimpleRAG)

> **IMPORTANT — STALE DATA WARNING:** The table below was last researched on **2026-04-29**. LLM pricing and model rankings change frequently. When this section is referenced, **always re-search the web for the latest models, pricing, and benchmarks before making recommendations**, then compare against this baseline.

### Current setup (as of 2026-04-29)
All modes use `gemini/gemini-2.5-flash` with thinking explicitly disabled (`thinkingBudget: 0`). This is a **dev-only** project — no production traffic.

### The $25 incident — never repeat
Flash non-lite + thinking ON during Boeing 10-K indexing = ~$25 in one session. Guard: always pass `thinkingBudget: 0` for all Gemini non-Pro models. Pro mandates thinking and should not be used without an explicit budget decision.

### Best models by mode (researched 2026-04-29 — re-verify before acting)

| Mode | Best model | Why | Input/M | Output/M |
|---|---|---|---|---|
| **PageIndex** | GPT-4.1 | Best agentic function calling (97% reliability), VectifyAI-tier | $2.00 | $8.00 |
| **PageIndex** (budget) | Gemini 2.5 Flash | Current, 64% FinanceBench, no thinking | $0.15 | $0.60 |
| **Normal RAG** | Gemini 2.5 Flash | Synthesis of retrieved chunks, Flash is sufficient | $0.15 | $0.60 |
| **Graph / Neo4j** | GPT-4.1 | More reliable Cypher generation | $2.00 | $8.00 |
| **Graph / Neo4j** (budget) | Gemini 2.5 Flash | Current, acceptable for dev | $0.15 | $0.60 |

### Smart upgrade path (when GPT key is available)
- Swap `GEMINI_MODEL` in `pageindex_service.py` to `gpt-4o-2024-11-20` or `gpt-4.1` — controls both indexing and querying.
- Keep Flash for normal/graph/neo4j modes — those don't need top-tier reasoning.
- Claude Sonnet 4.6 has 90% prompt caching discount ($0.30/M cached input) — worth benchmarking for PageIndex since structure is re-sent across tool calls.

### Estimated costs (dev, 2026-04-29 rates)
| Action | Gemini Flash | GPT-4.1 | Claude Sonnet 4.6 |
|---|---|---|---|
| Index one 190-page 10-K | ~$0.12 | ~$1.20 | ~$1.80 |
| Single PageIndex query | ~$0.005 | ~$0.04 | ~$0.06 |
| 14-question eval run | ~$0.08 | ~$0.74 | ~$1.11 |
