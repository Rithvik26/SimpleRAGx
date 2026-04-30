# Eval Observations — SimpleRAG

All observations recorded on **2026-04-30**. Results are only meaningful relative to the setup documented below.

---

## Setup

| Component | Value |
|---|---|
| LLM | `gemini/gemini-2.5-flash`, `thinkingBudget: 0` (thinking disabled) |
| Embeddings | Gemini `embedding-001`, 768-dim |
| Reranker | `gemini/gemini-2.5-flash` (LLM-based, not cross-encoder) |
| Vector DB | Qdrant Cloud |
| Graph DB | Neo4j AuraDB |
| Entity extraction | GLiNER `urchade/gliner_small-v2.1` (indexing only, via Modal) |
| top_k | 10 |
| Eval judge | `gemini/gemini-2.5-flash`, LLM-as-judge (semantic equivalence) |

**Collections:**
- `simple_rag_normal` / `simple_rag_graph` — AMD + Boeing 2022 10-K (real app collections)
- `multihop_graph_docs` / `multihop_graph` — MultiHop-RAG benchmark, 609 news docs (isolated)

---

## MultiHop-RAG Corpus — What Was Built (609 docs)

**Source:** HuggingFace `yixuantt/MultiHopRAG` (COLM 2024 paper dataset)

**Corpus:** 609 news articles across tech, finance, sports, entertainment topics.

**Collections built (do not re-index):**

| Collection | Contents | Mode |
|---|---|---|
| `multihop_graph_docs` | Dense chunks from all 609 docs | Normal retrieval |
| `multihop_graph` | Entity/relationship chunks extracted by GLiNER + LLM | Graph retrieval |

**How indexed:**
- GLiNER `urchade/gliner_small-v2.1` (via Modal) extracted named entities per chunk
- LLM (Gemini Flash) extracted relationships between entities
- Entities + relationships stored as nodes/edges in Neo4j AuraDB AND as searchable chunks in `multihop_graph`
- Dense chunks stored in `multihop_graph_docs` for normal retrieval path

**QA set:** 2,556 questions total. We sample 50 per run using `seed=42` (fixed — same 50 every run). Questions have 4 types: `inference_query`, `temporal_query`, `null_query`, `comparison_query`.

**Key constraint:** The multihop collections are **isolated** from the real app collections (`simple_rag_normal`, `simple_rag_graph`). Eval queries go to multihop collections; app queries go to real collections. Never cross-reference them.

---

## MultiHop-RAG — Graph Mode (n=50, seed=42, full 609-doc corpus)

### Accuracy progression

| Version | Accuracy | Change from prev | Key change |
|---|---|---|---|
| v1 | 52% | baseline | — |
| v2 | 54% | +2% | Query planner in graph mode + retrieval budget fix (top_k full for both doc + graph) + traversal seed/limit increase (seed[:10], limit=max(top_k×5, 50)) |
| v3 | 78% | +24% | Yes/No prompt fix + fairer judge (ignores trailing disclaimers, handles partial name matches) |
| v4 | 72% | −6% | REGRESSION — source boosting fired on temporal questions |
| v5 | 74% | +2% | Source boosting reverted + comparison prompt added — functionally same as v3 within noise |

**v3 is the effective best** — v5's 74% vs v3's 78% is within the ±5% LLM non-determinism noise (7 regressions vs 5 improvements, net −2 questions, all due to HyDE generating different hypothetical docs on re-run).

### v3 question-type breakdown

| Type | N | Accuracy | Notes |
|---|---|---|---|
| inference_query | 16 | 93.8% | Strongest category |
| null_query | 12 | 91.7% | Near-perfect — "Insufficient information" detection works well |
| temporal_query | 13 | 69.2% | Retrieval misses on source-named questions ("After The Verge reported...") |
| comparison_query | 9 | 44.4% | Main remaining gap |

### Comparison query failure analysis (5/9 failing)

| Root cause | Count | Fix status |
|---|---|---|
| LLM reasoning fail (ev_recall=1.0, LLM said "docs don't contain") | 2 | Partially fixed by comparison prompt in v5 — 1 resolved |
| Partial retrieval (ev_recall=0.33–0.5, source name not surfaced) | 2 | Attempted via source boosting → caused v4 regression; reverted |
| Full retrieval miss (ev_recall=0.0) | 1 | Unaddressed |

**Source boosting lesson:** `_extract_named_sources()` is defined in `simple_rag.py` but NOT called anywhere. It fires too broadly — hit temporal questions ("After Sporting News reported...") and destroyed ev_recall on those. Correct gate: **only boost when question is BOTH comparison-type AND mentions a source name.** Cap source-filtered results at top_k=3.

### v4 regression root cause (documented for posterity)
`_extract_named_sources()` was called in the RRF merge path. It parsed source names from temporal questions like "After The Verge reported..." and added source-filtered Qdrant searches. These pushed actual evidence chunks out of the top-k. ev_recall went 1.0 → 0.0 on 3 temporal questions.

### Published SOTA comparison (MultiHop-RAG, COLM 2024)

| System | Answer accuracy | Eval method |
|---|---|---|
| GPT-4 (perfect retrieval upper bound) | 89% | Exact match |
| Multi-Meta-RAG + GPT-4 (best published) | 60.6% | Exact match |
| GPT-4 naive RAG | 56% | Exact match |
| **SimpleRAG graph v3** | **78%** | **LLM-as-judge** |
| Claude 2.1 | 52% | Exact match |

**Caveat:** our 78% uses LLM-as-judge (semantic equivalence), published numbers use exact match. Not directly comparable — LLM-as-judge is more lenient. The gap vs published systems is partially explained by eval method.

---

## FinanceBench — Normal vs Graph (AMD + Boeing 2022 10-K, 14 questions)

| Mode | Accuracy | Avg latency |
|---|---|---|
| **Normal** | **78.6% (11/14)** | 11.1s |
| Graph | 71.4% (10/14) | 10.7s |

### Failures

| Question | Normal | Graph | Root cause |
|---|---|---|---|
| amd_05 — quick ratio / liquidity profile | ✗ | ✓ | Calculation not performed (retrieves raw numbers, no arithmetic) |
| amd_07 — customer concentration % | ✓ | ✗ | Graph traversal misses the specific % figure, normal chunk retrieval finds it |
| boeing_03 — improving gross margin | ✗ | ✗ | Needs sign detection on negative margin numbers |
| boeing_04 — primary customers | ✓ | ✗ | Graph mode retrieves entity nodes (names) but misses the % breakdown |
| boeing_07 — effective tax rate comparison | ✗ | ✗ | Arithmetic: 0.62% vs −14.76% requires calculating from raw tax/income figures |

### Observations

1. **Normal mode > Graph mode for financial 10-K QA.** Graph shines on multi-hop entity-linked reasoning (news). For table/number lookups in 10-Ks, dense chunk retrieval is better.

2. **Arithmetic is the main remaining gap.** boeing_07 (tax rate) and amd_05 (quick ratio) fail in both modes — the evidence is retrieved but no calculation step runs. A `_try_calculate` pass (already in pageindex_service.py) would close these.

3. **Graph mode loses on precise % figures.** Graph entities store names and relationships; the exact percentage from a table cell often lives in the raw chunk, not in a graph node. Normal mode retrieves the full chunk.

4. **Shared failure boeing_03 (improving margins):** Both modes get the sign wrong. Boeing had worsening margins — the LLM answers "No, margins decreased" correctly but the judge fails it. Possible judge prompt issue for negative-value comparisons.

---

## General Observations

- **n=50 is noisy.** At this sample size, ±5% swing is expected from LLM non-determinism in HyDE query expansion alone. Don't treat <5% accuracy changes as signal.
- **Judge quality matters more than model quality at this scale.** The jump from v2→v3 (54%→78%) came entirely from fixing the judge, not the RAG pipeline.
- **Retrieval (ev_recall) is the ceiling.** If ev_recall is 0, no prompt fix helps. For comparison_query failures, the retrieval problem must be solved first.
- **Latency is ~9–11s per query** across both benchmarks at top_k=10 with Gemini Flash.
