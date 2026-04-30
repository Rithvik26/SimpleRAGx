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

> **IMPORTANT — STALE DATA WARNING:** The table below was last researched on **2026-04-30**. LLM pricing and model rankings change frequently. When this section is referenced, **always re-search the web for the latest models, pricing, and benchmarks before making recommendations**, then compare against this baseline.

### Current setup (as of 2026-04-30)
All modes use `gemini/gemini-2.5-flash` with thinking explicitly disabled (`thinkingBudget: 0`). This is a **dev-only** project — no production traffic.

### The $25 incident — never repeat
Flash non-lite + thinking ON during Boeing 10-K indexing = ~$25 in one session. Guard: always pass `thinkingBudget: 0` for all Gemini non-Pro models. Pro mandates thinking and should not be used without an explicit budget decision.

### Best models by mode (researched 2026-04-30)

| Mode | Stage | Best model | Why | Input/M | Output/M |
|---|---|---|---|---|---|
| **PageIndex** | Agentic tool-calling | Claude Sonnet 4.5/4.6 | tau-bench #1 (0.862 retail, 0.700 airline) — multi-turn stateful tool loops; GPT-4.1 is now legacy-tier | $3.00 | $15.00 |
| **PageIndex** (budget) | Agentic tool-calling | Gemini 2.5 Flash | Current dev setup; ~72% agentic reliability, acceptable for eval | $0.15 | $0.60 |
| **Normal RAG** | Synthesis | Claude Sonnet 4.6 | 94% RAG accuracy, 1.9% hallucination rate (lowest); 90% prompt caching discount on repeated context | $3.00 | $15.00 |
| **Normal RAG** (budget) | Synthesis | Gemini 2.5 Flash | ~86% accuracy, good enough for dev/eval pipelines | $0.15 | $0.60 |
| **Graph / Neo4j** | Synthesis | Claude Sonnet 4.6 | Same as Normal RAG synthesis | $3.00 | $15.00 |
| **Graph / Neo4j** | Cypher generation | Claude Sonnet 4.6 | SWE-bench 43.0 vs GPT-4.1's 21.8; generates modern Cypher without deprecated patterns; Flash errors ~20% on multi-hop | $3.00 | $15.00 |
| **Graph / Neo4j** | Entity extraction | Gemini 2.5 Flash | Extraction is structured JSON output — no deep reasoning needed; cheapest at scale | $0.15 | $0.60 |
| **Embeddings** | Retrieval | Gemini embedding-001 | MTEB 68.32 (English API #1, Jan 2026); text-embedding-3-large is now overpriced (MTEB 64.6, $0.13/M) | ~$0.15/M | — |
| **Embeddings** (self-hosted) | Retrieval | Qwen3-Embedding-8B | MTEB 70.58 (multilingual #1); 32K context; Apache 2.0; free | free | — |
| **Reranking** | Post-retrieval | Voyage AI Rerank 2.5 | ELO 1544, highest nDCG@10 (0.110), $0.05/M, commercial-safe; Cohere 3.5 is now #10 — don't use | $0.05/M | — |

### Smart upgrade path
- **PageIndex:** Switch to Claude Sonnet 4.5/4.6 — it's now the clear agentic winner on tau-bench. GPT-4.1 is no longer the top agentic model.
- **Normal RAG + Graph synthesis:** Claude Sonnet 4.6 with prompt caching — the cached input rate ($0.30/M) makes it competitive vs Flash when context is re-sent.
- **Cypher generation:** Switch from Gemini Flash to Claude Sonnet 4.6 — Cypher errors are silent and Flash fails ~20% of multi-hop queries.
- **Embeddings:** Move from Gemini text-embedding-004 → Gemini embedding-001 (Jan 2026 release, MTEB +4 points, same API surface).
- **Reranker:** If not using one, add Voyage AI Rerank 2.5 ($0.05/M) — typically +8-15% accuracy on retrieval-heavy workloads.

### Estimated costs (dev, 2026-04-30 rates)
| Action | Gemini Flash | Claude Sonnet 4.6 | Claude Sonnet 4.6 (cached) |
|---|---|---|---|
| Index one 190-page 10-K | ~$0.12 | ~$1.80 | ~$0.55 (70% cache hit) |
| Single PageIndex query | ~$0.005 | ~$0.06 | ~$0.02 |
| 14-question eval run | ~$0.08 | ~$1.11 | ~$0.35 |
