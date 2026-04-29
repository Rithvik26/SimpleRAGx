# SimpleRAG → LLM-Wiki Compiler: Audit & Upgrade Plan

**Document owner:** Rithvik
**Last updated:** 2026-04-23
**Status:** Phase 1 ready to execute
**Purpose:** Single source of truth for the SimpleRAG architectural upgrade. A fresh Claude Code session should be able to execute any phase of this plan by reading this file.

---

## 0. TL;DR

SimpleRAG is a seven-mode retrieval orchestrator (`normal`, `graph`, `neo4j`, `hybrid_neo4j`, `pageindex`, `agentic`, `parallel`) that retrieves on every query. 2026 research (ICLR'26 GraphRAG-Bench, HippoRAG-2, Karpathy's LLM-Wiki pattern, Jina late-chunking) has moved toward compiling a persistent, interlinked, self-healing knowledge artifact and querying that artifact instead.

The highest-leverage move is **not** tuning the seven modes in place — it is **collapsing them into a compiler + lint + retrieval stack**, where the existing modes become retrieval *strategies over the wiki*, not separate kingdoms of truth. This plan executes that migration in four phases. Phase 1 stabilizes the current system; Phases 2–4 build the LLM-Wiki substrate on top of it.

---

## 1. Current-state audit (ranked by leverage)

All citations below reference files in the project root.

### Top 10 bottlenecks

| # | Bottleneck | Where | Why it matters | Fix |
|---|---|---|---|---|
| 1 | Entity IDs not stable across re-ingests | `graph_rag_service.py:101`, `neo4j_service.py:108` | Re-ingesting doubles the graph. Similarity threshold 0.8 (`config.py:33`) cannot compensate. Wiki would drift on every refresh. | Canonical ID = `slug(normalized_name) + "::" + type`. MERGE on that key. Maintain `aliases:` list. |
| 2 | No reranking in any mode | `simple_rag.py:789`, `graph_rag_service.py:843` | Gemini 768-d dense top-k is noisy. GraphRAG-Bench shows reranking is a bigger win than graph structure on Level-1 queries. | Add Jina ColBERT-v2 or `BAAI/bge-reranker-v2-m3` rerank step. ~60 LOC. |
| 3 | Chunking is document-structure-blind | `document_processor.py:314-376` | 1000-char sentence-sliding windows ignore headings, tables, lists, code. Late chunking beats this 2–18pp. | Structure-aware splitter (unstructured.io or `MarkdownNodeParser`) + late chunking with `jina-embeddings-v3`. Keep heading path in metadata. |
| 4 | Relationship extraction: single-pass, greedy, truncated | `graph_extractor.py:61-91` | Only first 1500 chars/chunk sent to LLM. Long chunks silently drop relations. Everything becomes `RELATES_TO`. | Multi-pass extraction + closed relation-type schema per document class. Validator LLM pass. Canonicalize triples before write. |
| 5 | `hybrid_neo4j` triple-stores and doesn't dedup on merge | `simple_rag.py:1213-1267` | 3× storage, 3× write, same fact cited from three stores. Neo4j Cypher scores faked (0.95/0.90/0.85). | Replace with one retrieval stage that composes signals. Drop `hybrid_neo4j` as a user-facing mode. |
| 6 | No query decomposition, no HyDE, no multi-query | `llm_service.py` throughout | "List all CEOs and their startups' markets" → one embedding. Agentic-RAG SoK (2026) shows decomposition lifts multi-hop 34% → 78%. | Planner pass (Haiku): decompose → retrieve per sub-query → fuse. ~150 LOC. |
| 7 | Cypher is LLM-generated, un-validated; errors silently empty | `neo4j_service.py:1143-1145` | User sees "no results" for broken queries. Confidence collapses. | Validate with `EXPLAIN` before `RUN`. One-shot self-correct on parse error. Surface final error. |
| 8 | `benchmark_modes.py` hard-codes API keys | `benchmark_modes.py:15-20` | Secrets in git. Five hardcoded TechCorp questions is a smoke test, not an eval. | Move to `.env`, rotate keys, integrate RAGAS + DeepEval in CI. |
| 9 | `ProgressTracker` leaks forever | `extensions.py` | No TTL. Multi-user or long-running instances drift unbounded. | TTL eviction (1h) + `WeakValueDictionary`. |
| 10 | Agentic mode has no iteration cap; LangChain import sprawl | `agentic_service.py:11-21`, `:150-200` | Potential infinite loops, fragile to LangChain churn. | Cap to 5 iterations (mirror `pageindex_service.py` / `config.py:54`). Or rip LangChain out; replace with ~80-LOC ReAct loop. |

### Additional issues (lower leverage)

- No embedding-response cache (`embedding_service.py`)
- Sentence splitter regex fails on abbreviations (`document_processor.py:323`)
- `EmbeddingCache` has no TTL or size bound (`embedding_service.py:23-27`)
- `agentic_stats` can crash on unset attribute (`agentic_service.py:248`)
- `completed_lock` protects count but not tracker state (`app.py:187, 235-240`)
- No distributed tracing or request IDs
- No LLM response cache — identical queries re-call LLM every time

### Per-mode summary

| Aspect | Normal | Graph | Neo4j | Hybrid | PageIndex | Agentic | Parallel |
|---|---|---|---|---|---|---|---|
| Ingest latency | ~0.5s | +3–5s | +3–5s | +6–10s | +5–10s | ~0.5s | N/A |
| Query latency | ~0.5s | ~0.6s | ~0.8s | ~1.5s | ~5–10s | ~2s | max(modes)+0.2s |
| Reranking | None | None | None | None | None | None | None |
| Dedup | N/A | Naive merge | Naive + Neo4j MERGE | Triple-store w/ dupes | N/A | N/A | None |
| Chunk awareness | Blind | Blind | Blind | Blind | Hierarchical | Blind | N/A |
| Stability | Stable | IDs regen | IDs regen | Both | Stable | LangChain risk | Thread-safe |

---

## 2. Why the LLM-Wiki architecture (research grounding)

Three 2026 signals converge:

**Karpathy's LLM-Wiki.** LLM is a compiler: `raw/` is source code, `wiki/` is the compiled artifact, `lint` is tests, `query` is runtime. Retrieval happens over the compiled wiki, which has already absorbed cross-document synthesis, contradiction resolution, and entity canonicalization. Pay compilation cost once instead of re-deriving on every query.

**ICLR'26 GraphRAG-Bench.** Graph structure hurts on simple lookup (RAG wins 60.9% vs 60.1% on Novel fact retrieval) and helps on multi-hop (HippoRAG-2: +10pp, MS-GraphRAG: +13pp). Right architecture isn't "graph vs vector" — it's **pick the retrieval signal by query class**, which requires a shared substrate.

**HippoRAG-2.** Dual-node KG (passages + phrases) + Personalized PageRank beats pure dense by +7 F1 on associative tasks. MuSiQue recall@5: 69.7 → 74.7. 2Wiki: 76.5 → 90.4. The specific retrieval method to run over the wiki.

**Jina late chunking.** Embedding after structure-aware splitting +2–18pp vs splitting-then-embedding. Recursive 512-token + 50–100 overlap is still the best no-model baseline (Vecta Feb'26: 69% vs 54% for semantic chunking).

**Agentic-RAG SoK (March 2026, arXiv 2603.07379).** Planner + Retriever + Grader + Tool-Executor loop beats static pipelines 34% → 78% on multi-hop. Maps 1:1 onto the `lint` + `query` agents in the target architecture.

---

## 3. Target architecture

Three stages, matching the user's provided diagram.

### Stage 1 — Ingestion (`raw/` append-only)

```
raw/
  pitch-deck-2026-01-17.pdf
  market-report-insurtech.md
  cap-table-acme-q1.csv
  _meta.yaml                # ingest log: source hashes, timestamps
```

- `raw/` is immutable, git-tracked. LLM never edits it.
- Ingest = copy source + content hash + append `_meta.yaml` row with absolute date.
- Replace current "ingest triggers embedding + graph extraction" path with "ingest only writes to `raw/`". Compilation is a separate, idempotent step. Eliminates duplicate-entity bugs.

### Stage 2 — LLM Compilation

Three components:

**(a) Document classifier.** Haiku call per new `raw/` file. Picks schema: `pitch_deck | market_report | cap_table | press | filing | transcript | other`. Each schema defines expected entity types and relation vocabulary. Branches chunking strategy.

**(b) Active compiler (Sonnet/Opus).** Per document, one context-window pass producing structured markdown nodes:

```
wiki/
  startups/acme.md
  founders/alice_chen.md
  markets/insurtech_2025.md
  concepts/                 # cross-topic syntheses (3+ overlapping sources)
    vertical_ai_insurance.md
  index.md                  # auto-regenerated catalog
  log.md                    # append-only: [2026-04-23] compile | acme.pdf → startup_acme
  schema.md                 # frontmatter contract, relation vocab, coverage thresholds
```

Node shape (Obsidian-compatible):

```markdown
---
id: startup_acme
type: startup
aliases: [Acme Inc, Acme Corp]
founders: [alice_chen, bob_ng]
stage: Series A
sources: [raw/pitch-deck-2026-01-17.pdf]
last_compiled: 2026-04-23
---

# Acme

## Summary [coverage: high -- 15 sources]
[[alice_chen]] founded Acme in 2024 to...

## Sources
- [[pitch-deck-2026-01-17]]
```

Key properties:
- Obsidian `[[wikilink]]` cross-refs — not Neo4j edges. Graph is implicit in link structure. Free KG, human-browsable.
- YAML frontmatter = canonical store for ID, type, aliases, sources. Replaces Neo4j `:Entity` as source of truth. Neo4j becomes a derived index.
- Coverage tags surface evidential strength.
- Incremental compilation: hash each `raw/` file; only recompile nodes whose source hash changed. `index.md` always regenerates.

**(c) Fan-out writer.** Applies node diffs atomically (temp + rename — generalize the pattern from `pageindex_service.py:150-180`).

### Stage 3 — Autonomous Lint Pass

Run as background job (cron or on-demand), not per-query. Four checks:

1. **Contradiction detection.** For each pair of nodes sharing ≥2 tags, LLM grades claims. Conflicts get `⚠️ contradiction` marker with both sources and resolution proposal (prefer newer source, note shift, preserve historical context).
2. **Duplicate merge.** Candidates = `cosine(node_embeddings) > 0.92 OR alias overlap`. LLM writes merge proposal. Preserves backlinks.
3. **Orphan linking.** Nodes with zero inbound `[[wikilinks]]` get a targeted "find 3 most related" pass.
4. **Network map refresh.** Recompute implicit graph, detect communities, surface new `concepts/` candidates.

Emits diffs, writes back. No human unless `--interactive`.

### Retrieval on top of the wiki

Existing modes become strategies, not kingdoms:

- **`normal`** → dense on wiki-node embeddings (late-chunked per section). Strong for Level-1 fact queries.
- **`graph`** → parse `[[wikilink]]` graph, run Personalized PageRank (HippoRAG-2 style) seeded by dense matches. Strong for multi-hop.
- **`pageindex`** → keep for long-doc agentic browsing; trigger when query names a doc.
- **`neo4j` / `hybrid_neo4j`** → deprecate. Neo4j becomes a derived cache if Cypher power-user access is needed.
- **`parallel`** → becomes a **router**, not fan-out. Planner picks strategies with grader-based CRAG self-correction.
- **`agentic`** → collapse into planner-grader-executor loop. Rip out LangChain.

---

## 4. Phase roadmap

### Phase 1 — Stop bleeding (1 week) ← **CURRENT**

Fix audit items #1, #7, #8, #9, #10. Purely local. No architectural change. Outcome: existing modes become trustworthy. See §5 for full execution plan.

### Phase 2 — Introduce `wiki/` without removing modes (2 weeks)

- Add `raw/` + `wiki/` directories on disk
- `/compile` endpoint: classifier → compiler → writer for one document
- `/lint` endpoint (basic: contradictions + orphans)
- Wiki viewer (Obsidian works; or mount in Flask UI)
- Side-by-side: "answers via wiki retrieval" vs existing modes
- RAGAS in CI (faithfulness, context precision, answer relevance)

Existing Qdrant/Neo4j stores remain. Wiki is **additional**, not replacement.

### Phase 3 — Route retrieval through the wiki (2 weeks)

- Dense + PPR retrieval strategies over the wiki
- Planner-grader-router
- Deprecate `hybrid_neo4j`
- Keep `normal` as fallback during A/B
- Qdrant stores wiki-chunk embeddings only at end of phase

### Phase 4 — Compound & polish (ongoing)

- Scheduled `/lint` cron
- `concepts/` auto-synthesis
- Late chunking (Jina v3 or bge-m3)
- ColBERT rerank
- Claude Code slash commands: `/wiki-compile`, `/wiki-query`, `/wiki-ingest`, `/wiki-lint`

### Cost estimate (≤1000 docs, ≤100 wiki nodes)

- Full-corpus compile: ~$5–15 (Sonnet)
- Per query: ~$0.01 (Haiku planner + small-model synthesis)
- Lint pass: $2–5 per run; cron'd, not per-query

---

## 5. Phase 1 execution plan (detailed)

**Goal:** Stabilize existing codebase before building the wiki layer on top. Every task below is independently shippable.

**Prerequisites before starting:**
- `git status` clean on master
- `dev_config.json` loaded (verify Qdrant + Neo4j + Gemini keys work)
- Run `python benchmark_modes.py` once and save output as baseline

### Task 1.1 — Rotate and externalize API keys (audit #8)

**Files:** `benchmark_modes.py`, `dev_config.json`, `.gitignore`, new `.env.example`

**Changes:**
1. Create `.env.example` at root listing all required env vars (no values):
   ```
   GEMINI_API_KEY=
   ANTHROPIC_API_KEY=
   QDRANT_URL=
   QDRANT_API_KEY=
   NEO4J_URI=
   NEO4J_USER=
   NEO4J_PASSWORD=
   ```
2. Add `.env` to `.gitignore` (verify not already there)
3. `benchmark_modes.py:15-20` — replace hard-coded keys with `os.environ["..."]` reads, fail-fast with clear error if missing
4. **Rotate the exposed keys** (Gemini, Anthropic, Qdrant, Neo4j) — they've been in git history; treat as compromised
5. Commit message: `security: remove hardcoded API keys from benchmark_modes.py`

**Verification:** `grep -rn "AIza\|sk-ant\|neo4j+s" --include="*.py"` returns nothing. `python benchmark_modes.py` still runs when env vars set.

**⚠️ Do not skip the rotation step.** The keys are in git history even after this commit.

### Task 1.2 — Stable entity IDs (audit #1)

**Files:** `graph_extractor.py`, `graph_rag_service.py`, `neo4j_service.py`, new `entity_canonicalizer.py`

**Changes:**
1. Create `entity_canonicalizer.py`:
   ```python
   import re, unicodedata

   def canonical_id(name: str, entity_type: str) -> str:
       # NFKC normalize, lowercase, strip punct, collapse whitespace, slugify
       n = unicodedata.normalize("NFKC", name).lower()
       n = re.sub(r"[^\w\s-]", "", n)
       n = re.sub(r"\s+", "_", n.strip())
       return f"{entity_type.lower()}::{n}"
   ```
2. `graph_extractor.py` — after LLM extraction, compute `canonical_id` for every entity. Include it in the returned dict as `"id"`.
3. `graph_rag_service.py:101` — merge logic uses `id` equality instead of string-similarity. Keep similarity merge only for proposing *alias* additions. Maintain `aliases: [...]` list on each canonical entity.
4. `neo4j_service.py:108` — `MERGE (e:Entity {id: $id})` on canonical ID. Set `name` and `aliases` as properties. Existing queries that matched on `name` become queries that match on `id` OR `name IN aliases`.
5. Migration script `scripts/migrate_entity_ids.py`: read existing Neo4j, compute canonical IDs, merge duplicates, write back. Dry-run by default, `--apply` to execute.

**Verification:** Ingest the same document twice. Neo4j node count is identical after run 2. Graph stats (`graph_rag_service.get_graph_stats()`) show stable entity count.

### Task 1.3 — Cypher validation + self-correct (audit #7)

**Files:** `neo4j_service.py`

**Changes:**
1. Add `_validate_cypher(cypher: str)` method that runs `EXPLAIN <cypher>` in a read-only tx. Returns `(valid: bool, error: str | None)`.
2. In `generate_cypher_from_question` (or wherever the LLM-produced Cypher is executed — likely `execute_cypher_query`):
   - Validate before execution
   - On invalid: one retry with prompt `"The Cypher query you generated failed with error: {err}. Return ONLY a corrected Cypher query, no prose."`
   - If second attempt fails: surface error to caller as a typed exception `CypherGenerationError`, don't return empty results
3. `simple_rag.py:1143-1145` and `app.py` routes that call Neo4j — catch `CypherGenerationError` and return a structured error to the user (not silent empty).

**Verification:** Ask an impossible question that forces bad Cypher (e.g., "return the gradient of the founder graph"). Should return explicit error, not empty list.

### Task 1.4 — Cap agentic iteration, reduce LangChain surface (audit #10)

**Files:** `agentic_service.py`, `config.py`

**Changes:**
1. `config.py` — add `AGENTIC_MAX_ITERATIONS = 5` (mirror `PAGEINDEX_MAX_TOOL_ROUNDS`)
2. `agentic_service.py:150-200` — explicit iteration counter; break and return best-effort answer when hit. Log the cap event at `WARNING`.
3. `agentic_service.py:248` — guard `getattr(self, "agentic_stats", {})` so stats access never crashes when init partially failed.
4. `agentic_service.py:11-21` — consolidate the three fallback import paths into one helper function `_import_langchain_agent()` returning `(AgentExecutor, create_react_agent)` or raising a clear `ImportError` with install hint.

**Don't** rip out LangChain in Phase 1 — that's Phase 2. Just contain the blast radius.

**Verification:** Force an infinite-loop scenario (tool returns constant "need more info"). Agent exits at iteration 5, returns answer with note about cap.

### Task 1.5 — ProgressTracker TTL + cleanup (audit #9)

**Files:** `extensions.py`

**Changes:**
1. Store `created_at: datetime` on each `ProgressTracker` instance
2. Add `PROGRESS_TTL_SECONDS = 3600` constant (1 hour)
3. `ProgressTracker.get_tracker(session_id)` — before returning, evict entries older than TTL. Batch eviction: only scan if last eviction was >60s ago (avoid per-call O(n) scan).
4. Alternative lower-effort option: wrap the module-level tracker dict in `WeakValueDictionary` so Python GCs trackers when references drop. TTL is more predictable; pick one.
5. Add `/api/progress/cleanup` admin endpoint (optional) that force-runs eviction and returns count.

**Verification:** Simulate 1000 session IDs via a quick script. Sleep past TTL. Tracker dict size returns to baseline.

### Task 1.6 — RAGAS eval harness in CI (audit #8 follow-on)

**Files:** new `eval/ragas_harness.py`, new `eval/ground_truth.yaml`, `.github/workflows/eval.yml` (if GH Actions), `requirements.txt`

**Changes:**
1. Add `ragas` and `datasets` to `requirements.txt`
2. `eval/ground_truth.yaml` — start with the 5 TechCorp questions already in `benchmark_modes.py` plus ~15 more covering: simple fact retrieval, multi-hop, summarization, contradiction, out-of-scope. Each entry has `question`, `expected_answer`, `expected_contexts` (list of document snippets).
3. `eval/ragas_harness.py` — load ground truth, run each question through each mode, compute RAGAS metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`. Export JSON per-mode per-metric.
4. CI job: run harness on PR, fail if any metric drops >5pp vs main.

**Verification:** Push a dummy branch that deliberately breaks retrieval. CI fails with a clear metric delta.

### Task 1.7 — Baseline benchmark and commit

After 1.1–1.6 are merged:
1. Re-run `python benchmark_modes.py` with the now-clean secrets path
2. Run the RAGAS harness from 1.6 and save `eval/baseline_phase1.json`
3. Commit `eval/baseline_phase1.json` as the reference point Phase 2 must beat

### Phase 1 exit criteria

- [ ] No hardcoded secrets in repo; keys rotated
- [ ] Re-ingesting the same document does not grow the graph
- [ ] Bad Cypher surfaces errors instead of silently returning empty
- [ ] Agentic mode has hard iteration cap
- [ ] `ProgressTracker` memory does not grow unboundedly
- [ ] RAGAS baseline committed; CI blocks regressions
- [ ] All existing modes still pass their smoke tests
- [ ] `eval/baseline_phase1.json` saved

### Estimated effort

~5–8 focused days for one engineer. Tasks 1.1 and 1.2 are prerequisites for everything else; 1.3–1.6 parallelize after that.

---

## 6. Things to explicitly NOT do

- **Don't unify Neo4j and wiki as co-equal stores.** One source of truth (wiki), derive the rest. `hybrid_neo4j` is the counterexample.
- **Don't abstract over Qdrant.** `vector_db_service.py` is fine; keep it.
- **Don't add LangGraph.** Whole retrieval loop is ≤300 LOC. Framework overhead exceeds the logic. ~80-LOC hand-rolled ReAct wins.
- **Don't ship semantic chunking right away.** Vecta Feb'26 benchmark: recursive 512+overlap still wins on average. Add late chunking and structure-awareness first; A/B semantic later.
- **Don't build the wiki viewer from scratch.** Obsidian points at any `wiki/` directory.

---

## 7. How to resume in a new chat

A fresh Claude Code session should be able to pick this up by:

1. Reading this file (`UPGRADE_PLAN.md`)
2. Running `git log --oneline -20` to see what's already shipped
3. Checking `eval/baseline_phase1.json` presence to see if Phase 1 is complete
4. Picking the next unchecked task from §5 exit criteria

**Prompt template to paste in a new chat:**

> Read `UPGRADE_PLAN.md` at the project root. We are executing Phase 1 §5 Task 1.X. Before making changes, run `git status` and `git log --oneline -5` so you're grounded. Follow the exact file changes and verification step listed for that task. Ship one task per PR.

---

## 8. Production Infrastructure Plan (as of 2026-04-30)

### What we fixed (shipped)

| Fix | File | Impact |
|---|---|---|
| Batch graph embedding | `graph_rag_service.py` | 10-18x faster (297 elements: 2:48 → 9s) |
| Doc-level skip cache | `simple_rag.py` | Re-index same doc = instant, $0 LLM cost |
| Parallel doc indexing (eval) | `eval/multihop_rag_real.py` | 4.5x faster (25 docs: 14 min → 3 min) |

### Current bottlenecks (ranked)

1. **Graph extraction** — 1 LLM call per chunk, dominant cost. Only fix is faster model or parallel docs.
2. **GLiNER cold start** — ~35s model load every fresh Python process. Fixed when server stays warm (Render deploy).
3. **Sequential docs in server** — upload page does one doc at a time by design (single-file upload). Fine for now.

### Infra tier plan

**Tier 0 — Dev (now, $0/month)**
- Qdrant Cloud free (1GB RAM, 4GB disk, ~500k vectors)
- Neo4j AuraDB free (200k nodes)
- `ThreadPoolExecutor(4)` for parallel indexing in eval harness
- GLiNER on local CPU / Render web dyno

**Tier 1 — First real users (~$80-130/month)**
- Qdrant Cloud Standard — $50-80/mo, 4GB dedicated, 99.9% SLA
- Neo4j AuraDB Professional — $65/mo, no node limit
- Render web service (existing) stays free
- Modal Labs for GLiNER — $30/mo free credits, GPU warm serving

**Tier 2 — Multiple simultaneous users (~$300-500/month)**
- Celery + Upstash Redis — async job queue, retries, job history
- Upstash Redis free tier: 500k commands/month, 256MB (sufficient for dev)
- Render background worker — $7/mo for dedicated Celery worker
- Qdrant 8GB cluster — $150-200/mo

**Tier 3 — Production scale ($1000+/month)**
- Ray or Celery + RabbitMQ for distributed workers
- Qdrant Hybrid Cloud (your own infra, Qdrant manages ops)
- Neo4j self-hosted on EC2
- GPU workers for GLiNER (NVIDIA L4: 8,771 pages/hour)

### Technology decisions

**Celery vs Ray:**
- **Celery** — winner for SimpleRAG. Background job queue for Flask web app. pip install, works with existing code.
- **Ray** — for serving your own trained ML models at scale (10k concurrent users). Overkill here.
- **AWS Lambda** — 15-min hard limit kills it for indexing (Boeing 10-K = 20 min). Good for query endpoint only.
- **AWS SQS** — AWS-native Celery equivalent. More complexity, same result. Use when already on AWS.

**Free options available right now:**
- Upstash Redis: 500k commands/month free, no credit card
- Modal Labs: $30/month free compute credits for GLiNER GPU serving
- Render background workers: NOT free ($7/mo minimum)
- Fly.io / Railway / Koyeb: all killed free worker tiers in 2025-2026

### Cost benchmarks (observed Apr 2026)

| Action | Cost |
|---|---|
| Index 25 news articles (graph mode) | ~$0.05 |
| Index full 609-doc MultiHop-RAG corpus | ~$0.80-1.50 (one-time) |
| Re-index same corpus (with skip cache) | ~$0.00 |
| Single query + judge | ~$0.005 |
| n=50 eval run | ~$0.10 |

### Next infra steps (priority order)

1. **Modal for GLiNER** — free, removes 35s cold start, GPU inference
2. **Upstash Redis Streams + Celery** — when first concurrent users arrive
3. **Render background worker ($7/mo)** — proper Celery worker separation
4. **Qdrant Standard** — when free tier hits limits (~500k vectors)

---

### Message queue decision — Redis Streams vs RabbitMQ vs AWS SQS

**Winner: Redis Streams (Upstash free tier)**

| | Redis Streams | RabbitMQ | AWS SQS |
|---|---|---|---|
| **Delivery** | At-least-once (AOF) | At-least-once ✅ | At-least-once ✅ |
| **Persistence** | AOF/RDB config needed | Durable queues built-in | Always persistent ✅ |
| **Message replay** | ✅ Yes (log-based) | ❌ No | ❌ No |
| **Fan-out** | ✅ Consumer groups | ✅ Exchanges | ❌ Needs SNS |
| **Max message size** | 512MB | 128MB | **256KB ❌** |
| **Dead letter queue** | Manual | Built-in ✅ | Built-in ✅ |
| **Free tier** | Upstash 500k/mo | CloudAMQP 1M/mo | 1M requests/mo forever |
| **Vendor lock-in** | Redis protocol (open) | AMQP (open) | AWS only ❌ |
| **Ops burden** | Low | Low | Zero |

**Why Redis Streams wins for SimpleRAG:**
- Already planned Upstash Redis — no extra service
- Message replay = re-process failed indexing jobs without re-upload
- 512MB message size — no issues passing large doc metadata
- Celery supports it natively via kombu
- Free on Upstash (500k commands/month)

**Why NOT Redis Pub/Sub:**
- Fire and forget — if worker is offline, message gone forever
- No persistence, no acknowledgement, no retry
- Built for live notifications (chat, dashboards), not background jobs

**Why NOT Kafka:**
- Built for 10k+ events/second, event streaming, financial transactions
- Minimum 6GB RAM to self-host (JVM + ZooKeeper)
- No meaningful free tier (Confluent free expires after 30 days)
- 18-wheel truck for moving a pizza box

**When to switch:**
- → RabbitMQ: need complex routing (different queue per doc type) or proper AMQP dead letter queues
- → AWS SQS: already deep in AWS (Lambda, EC2, IAM all wired), messages stay under 256KB
- → Kafka: 10k+ indexing jobs/day, need event replay at scale, multiple microservices consuming same stream

---

## 9. Sources

- [Karpathy, LLM Wiki gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- [VentureBeat — Karpathy LLM Knowledge Base bypasses RAG](https://venturebeat.com/data/karpathy-shares-llm-knowledge-base-architecture-that-bypasses-rag-with-an)
- [llm-wiki-compiler (reference impl)](https://github.com/ussumant/llm-wiki-compiler)
- [GraphRAG-Bench (ICLR'26) — When to use Graphs in RAG](https://arxiv.org/html/2506.05690v3) / [repo](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
- [HippoRAG-2 / From RAG to Memory](https://arxiv.org/abs/2502.14802)
- [Jina Late Chunking](https://arxiv.org/pdf/2409.04701)
- [Jina ColBERT-v2](https://jina.ai/news/jina-colbert-v2-multilingual-late-interaction-retriever-for-embedding-and-reranking/)
- [Jina reranker-v3 (LBNL)](https://arxiv.org/pdf/2509.25085)
- [SoK: Agentic RAG, March 2026](https://arxiv.org/html/2603.07379v1)
- [RAGAS faithfulness docs](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/)
- [PreMAI 2026 RAG eval roundup](https://blog.premai.io/rag-evaluation-metrics-frameworks-testing-2026/)
- [PreMAI 2026 chunking benchmark](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/)
