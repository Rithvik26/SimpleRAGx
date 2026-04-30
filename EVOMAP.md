# Evomap — Deal Intelligence Infrastructure for Venture Firms

**Document owner:** Rithvik
**Last updated:** 2026-04-30
**Status:** Architecture designed, not yet built
**Purpose:** Single source of truth for Evomap product, architecture, and engineering decisions.

---

## 1. What Evomap Is

Evomap is deal intelligence infrastructure for venture firms.

VCs spend 80% of their time on work that doesn't require a partner's judgment — reading pitch decks, verifying claims, monitoring portfolio companies, writing LP reports, drafting cold outreach. Evomap automates all of that.

**The short pitch:** Evomap is the AI analyst layer that sits between a VC firm's deal flow and their partners' time. Everything that doesn't require a partner's judgment gets automated. Everything that does gets handed to them pre-researched.

**What it is not:** A matching engine. Matching is table stakes — Crunchbase, Signal, Dealroom already do it with structured data. You don't win on matching alone. Matching is one output of the intelligence infrastructure, not the whole product.

---

## 2. The Seven Products

### 2.1 Deal Matching
Ingest any document — pitch deck, fund mandate, market report, cap table — extract structured intelligence automatically, match startups to VCs continuously offline. Multi-dimensional similarity across thesis, stage, sector, geo, check size, and portfolio fit. Conflict check against portfolio graph before anything surfaces. Results in 250ms. Rationale for top-5 streams async.

### 2.2 Due Diligence Automation
A partner clicks DD on any startup and gets back a structured memo in 10 minutes instead of 2 weeks. Every claim in the pitch deck verified against public data. Founder background checked. Competitive landscape mapped. Red flags flagged with evidence and confidence scores. Junior analyst work, automated.

### 2.3 Market Mapping
"Show me the AI infrastructure landscape" — Evomap pulls from funding data, news, patents, job postings, and academic papers and produces a living competitive map updated weekly. Identifies white spaces. Used for thesis validation and sourcing.

### 2.4 Portfolio Monitoring
Key person departures, competitor funding rounds, regulatory changes, revenue proxy signals (SimilarWeb, app store rankings, GitHub activity). Daily digest, prioritized by urgency: Red (action needed), Yellow (watch), Green (positive signal).

### 2.5 Outreach Personalization
Reads the startup's recent activity, finds the specific hook, writes cold outreach in the partner's voice calibrated from their past emails. Personalized cold outreach has 3-5x higher response rates than templated.

### 2.6 LP Report Generation
Pulls portfolio data quarterly, writes the narrative, formats it. Ops team reviews and approves instead of building from scratch. Saves 2-3 weeks of ops time per quarter.

### 2.7 Profile Enrichment (background, continuous)
Takes every structured profile and continuously enriches it from external sources: Crunchbase, LinkedIn public, USPTO patents, SEC EDGAR, GitHub. Updates nightly. This is the data moat — competitors have the same AI, you have better data.

---

## 3. Architecture

### Core Insight: Offline vs Online Split

**Never run LLM agents in the online hot path.** Agents are for offline ETL and on-demand workflows. Online serving is pure math — SQL, graph traversal, vector ANN, reranking. No LLM until the final rationale generation, and even that is async.

The O(n×m) matching problem (500 VCs × 2000 startups = 1M pairs) makes sequential agents in the hot path architecturally impossible at scale.

### Three Layers

```
┌─────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                      │
│  S3 → Docling → GLiNER → Haiku extraction → Postgres   │
│  Triggered by: upload, scheduled scrape, API webhook    │
│  Orchestrated by: Temporal (durable, retryable)         │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                 INTELLIGENCE LAYER                      │
│  Enrichment Agent | DD Agent | Market Mapper            │
│  Outreach Agent   | LP Report Agent                     │
│  Runs OFFLINE or ON-DEMAND — never in matching hot path │
│  Data stores: Postgres+pgvector, Neo4j                  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   SERVING LAYER                         │
│  Hard filter (SQL 5ms) →                                │
│  Conflict check (Neo4j Cypher 10ms) →                   │
│  Multi-vector ANN (pgvector 20ms) →                     │
│  Cohere Rerank (200ms) →                                │
│  Return top-20 ranked matches with scores               │
│  ASYNC: Sonnet rationale for top-5 only                 │
│  Total: <250ms to scores, rationale streams after       │
└─────────────────────────────────────────────────────────┘
```

### Data Model

Single Postgres instance with pgvector. No Qdrant. No in-memory graph. Two stores total.

```sql
-- Postgres + pgvector
vc_profiles        (id, name, firm, check_min, check_max, stages[], sectors[], geo[], mandate_text, last_indexed_at)
startup_profiles   (id, name, arr, runway_months, stage, sectors[], geo, team_size, ask, last_indexed_at)
vc_embeddings      (vc_id, aspect ENUM, embedding vector(1536))   -- 5 rows per VC: thesis/sector/stage/geo/team
startup_embeddings (startup_id, aspect ENUM, embedding vector(1536))
match_candidates   (vc_id, startup_id, compat_score, hard_filtered_at, ranked_at)
match_rationale    (vc_id, startup_id, rationale_text, generated_at)
```

```cypher
// Neo4j — portfolio graph only
(vc:VC)-[:INVESTED_IN]->(startup:Startup)
(startup:Startup)-[:COMPETES_WITH]->(startup:Startup)
(startup:Startup)-[:OPERATES_IN]->(sector:Sector)
```

---

## 4. The Agents

### Agent 1: Document Ingestor
**Type:** Deterministic ETL pipeline (not truly agentic — no conditional branching)
**Trigger:** New document upload to S3
**Flow:** Docling (PDF → structured markdown, table-aware) → GLiNER (NER tagging) → Claude Haiku structured extraction with Pydantic schema → Postgres row + pgvector embeddings + Neo4j relationships
**Why Haiku:** Better structured output adherence than Gemini Flash for typed schemas. With 90% prompt caching on the extraction system prompt, effective cost per doc is competitive.
**Not in hot path.**

### Agent 2: Profile Enricher
**Type:** Genuinely agentic (decides what to search based on what's missing)
**Trigger:** Nightly cron, runs on all profiles updated in last 24h
**Flow:** Read profile → identify missing/stale fields → decide what to look up → call tools → update Postgres with source citations and confidence scores
**Tools:** Crunchbase API, Tavily web search, USPTO API, SEC EDGAR, GitHub API
**Note:** GLiNER runs at ingest time (Agent 1), not here. Results cached in Postgres.

### Agent 3: Due Diligence Agent
**Type:** Genuinely agentic (number of verification steps depends on pitch deck content)
**Trigger:** On-demand, partner clicks "Run DD"
**Flow:**
1. Extract all factual claims from pitch deck
2. For each claim, spawn verification sub-task (web search, competitor check, market size cross-reference)
3. Cross-reference founder team (past companies, exits, education, GitHub, media)
4. Check competitive landscape (recent funding rounds, incumbent players)
5. Flag red flags (litigation, failed startups, inflated metrics)
6. Synthesize into structured memo in VC's house style

**Where Normal RAG earns its place:** RAG over the VC's own past deal memos to calibrate output style and surface relevant portfolio comparables.
**Output:** Structured DD memo with claim → evidence → confidence → flag format.

### Agent 4: Market Mapper
**Type:** Genuinely agentic (judgment calls on categorization and white space identification)
**Trigger:** Weekly cron per tracked sector
**Flow:** Pull from Crunchbase, arxiv (deep tech), USPTO, job postings, news → categorize landscape → identify white spaces → generate structured brief
**Output:** Living competitive landscape doc, updated weekly. Used for thesis validation and sourcing.

### Agent 5: Portfolio Monitor
**Type:** Monitoring pipeline with alert generation (not agentic in the tool-use sense)
**Trigger:** Continuous, daily digest generation
**Signals:** News mentions, LinkedIn change detection, competitor funding, regulatory changes, job posting changes, GitHub activity, SimilarWeb traffic
**Output:** Daily digest prioritized by urgency. Red/Yellow/Green.
**Implementation:** Celery cron jobs + Sonnet summarization. No LangGraph needed here.

### Agent 6: LP Report Generator
**Type:** Agentic document generation (pulls from multiple sources, writes narrative)
**Trigger:** Quarterly schedule
**Flow:** Pull all portfolio data from Postgres → enriched profiles → valuation updates → milestone tracking → generate structured report in fund's house style → human reviews and approves
**Note:** Human always in the loop for final approval. Agent handles 80% of the work.

### Agent 7: Outreach Personalizer
**Type:** On-demand generation
**Trigger:** VC selects a startup to reach out to cold
**Flow:** Read startup's recent news, blog posts, product launches → identify specific hook → RAG over VC's past sent emails for tone calibration → generate personalized intro
**Where Normal RAG earns its place:** Retrieving relevant portfolio examples and past deals in adjacent spaces to personalize the hook.

---

## 5. Technology Stack

| Layer | Tool | Why |
|---|---|---|
| PDF parsing | Docling | Table-aware, open source, IBM-backed. Better than raw PDF → LLM. |
| NER | GLiNER on Modal | Free GPU credits ($30/mo), warm serving, removes 35s cold start |
| Extraction LLM | Claude Haiku 4.5 | Structured output + prompt caching on system prompt |
| Rationale / DD LLM | Claude Sonnet 4.6 | Prompt caching on VC profile (90% discount on repeated context) |
| Vector + structured | Postgres + pgvector | One system. Consistent. No 3-store sync problem. |
| Graph | Neo4j | Portfolio conflict only. Single-hop Cypher traversal. |
| Re-ranking | Cohere Rerank | 200ms, better than LLM-as-judge, not a retrieval problem |
| Live market signals | Tavily / Perplexity API | Static corpus is baseline, live search is delta |
| Workflow orchestration | Temporal | Durable, versionable, retryable. LangGraph is a prototype tool. |
| Job queue | Celery + Redis Streams | Simple async tasks outside Temporal |
| Serving | FastAPI + Uvicorn | Standard |
| Infra (now) | Render + Modal + Upstash | Free tiers cover dev |
| Infra (scale) | AWS ECS + RDS + managed Neo4j | Incremental migration when needed |

### What Gets Cut vs Current SimpleRAG Design

| Removed | Replaced by | Why |
|---|---|---|
| Qdrant | pgvector in Postgres | One consistent system, no sync |
| In-memory Graph RAG | Neo4j | Persists, scales, multi-worker |
| LangGraph | Temporal | Durability, versioning, retries |
| 5-agent sequential hot path | Offline ranking, online math | O(n×m) scale requirement |
| LLM-as-judge for scoring | Cohere Rerank + XGBoost ranker | LLM score is a feature, not the answer |
| hybrid_neo4j mode | Not needed | LangGraph orchestration replaced it |
| Normal RAG as matching mode | pgvector ANN | Strictly better for this use case |

---

## 6. Where Each SimpleRAG Mode Lives in Evomap

| Mode | Where used | Why |
|---|---|---|
| **PageIndex** | Agent 1 (ingestion) | PDF extraction for pitch decks and fund docs. Page-level tool calling for structured fields. |
| **Normal RAG** | Agent 3 (DD) + Agent 7 (outreach) | RAG over VC's own deal archive for style calibration and portfolio comparables. |
| **Neo4j** | Serving layer hard filter | Portfolio conflict check — deterministic graph traversal, not a retrieval problem. |
| **pgvector ANN** | Serving layer matching | Replaces Qdrant. Multi-aspect embeddings (5 per entity), Cohere rerank on top-50. |
| **In-memory Graph RAG** | Deleted | Never again in production. |
| **hybrid_neo4j** | Deprecated | No role in Evomap. |

---

## 7. The Matching Pipeline in Detail (Online)

```
Request: "matches for VC X"
    ↓
Hard filter: SQL (stage, geo, check size) — 5ms
    ↓
Hard filter: Neo4j conflict check — 10ms
    (single Cypher hop: VC → invested_in → Startup → sector ← candidate)
    ↓
Multi-vector ANN: pgvector cosine across 5 aspects — 20ms
    (thesis, sector, stage, geo, team — separate embeddings per aspect)
    ↓
Cohere Rerank on top-50 candidates — 200ms
    ↓
Return top-20 ranked matches with scores — <250ms total
    ↓
ASYNC: Sonnet generates rationale for top-5 only
    (VC profile is prompt-cached — only startup diff is new tokens)
    Rationale streams to UI 2-3 seconds after scores appear
```

---

## 8. What Actually Determines If This Works

**None of the above matters without ground truth match data.**

The ranker is only as good as its training signal. You need behavioral outcomes: did the VC take a meeting? Did they invest? Did they pass and why?

Without that feedback loop, Evomap is an expensive similarity search engine dressed up as a matching platform. With it, the system improves every week.

**Cold start strategy:** Don't try to match all VCs to all startups on day one. Target a small, specific VC community (10-20 funds), go extremely deep on their specific needs, collect real outcome data. Validate the ranker against that data before expanding. The data flywheel is the moat, not the AI.

**The defensible position:** Every DD memo run teaches the system what VCs care about. Every portfolio event tracked improves the monitoring model. Every funded match validates the ranker. Competitors have access to the same LLMs. You have proprietary behavioral data about how VCs actually make decisions.

---

## 9. Honest Assessment

| Product | Score | Reasoning |
|---|---|---|
| Matching alone | 3/10 | Table stakes. Crunchbase exists. |
| Due Diligence automation | 9/10 | High pain, high value, hard to replicate without the data pipeline |
| Market Mapping | 7/10 | Genuinely useful, but content quality depends on corpus coverage |
| Portfolio Monitoring | 8/10 | Pure time savings, clear ROI, easy sell to portfolio ops |
| Outreach Personalization | 7/10 | High response rate lift, measurable |
| LP Reporting | 8/10 | Different buyer (COO/CFO), clear time savings, easy to demo |
| Full platform | 8/10 | Data flywheel is real. Network effect is real. Moat builds over time. |

---

## 10. Production LLM Stack

*Last researched: 2026-04-30. Re-verify pricing before large runs — this table decays fast.*

### By RAG mode: dev vs production

#### Normal RAG (vector retrieval + synthesis)

| Tier | Model | RAG accuracy | Hallucination rate | Input/M | Output/M |
|---|---|---|---|---|---|
| **Production** | Claude Sonnet 4.6 | 94% | 1.9% (lowest) | $3.00 | $15.00 |
| Premium alt | GPT-5.4 | 91% | 3.2% | $2.50 | $10.00 |
| Budget prod | Gemini 2.5 Pro | 89% | 4.1% | $1.25 | $10.00 |
| **Dev (current)** | Gemini 2.5 Flash | ~86% | ~5% | $0.15 | $0.60 |

Claude Sonnet 4.6 has 90% prompt caching discount ($0.30/M cached input) — for Evomap DD memos where the fund mandate and doc structure are re-sent on every call, prompt caching offsets the $3.00 base rate significantly.

#### Graph RAG

Three distinct sub-tasks, each with the right model:

| Sub-task | Production model | Why | Input/M | Output/M |
|---|---|---|---|---|
| **Synthesis** (multi-hop QA) | Claude Sonnet 4.6 | 94% accuracy, best multi-doc reasoning, 1.9% hallucination | $3.00 | $15.00 |
| **Cypher generation** | Claude Sonnet 4.6 | SWE-bench 43.0 (vs GPT-4.1's 21.8); correctly generates modern Cypher without deprecated patterns | $3.00 | $15.00 |
| **Entity/rel extraction** | Gemini 2.5 Flash | Extraction is JSON-structured output, doesn't need top-tier reasoning; cost-efficient at scale | $0.15 | $0.60 |
| **Extraction (quality)** | GPT-4.1 | If extraction errors are propagating to graph quality, GPT-4.1 structured output is more reliable | $2.00 | $8.00 |

Cypher generation errors are silent (return empty result set, no exception) — model quality here directly limits graph traversal accuracy. Flash generates invalid Cypher ~20% of the time on multi-hop queries. Claude Sonnet 4.6's higher coding benchmark score closes this gap.

#### PageIndex / Agentic RAG

The right benchmark here is **tau-bench** (multi-turn, stateful tool-calling loops) not BFCL (single-turn format precision). tau-bench is what PageIndex-style agentic RAG actually resembles.

| Tier | Model | tau-bench retail | tau-bench airline | Input/M | Output/M |
|---|---|---|---|---|---|
| **Production** | Claude Sonnet 4.5/4.6 | **0.862** (#1) | **0.700** (#1) | $3.00 | $15.00 |
| Alt | GPT-5.4 | ~0.81 | — | $2.50 | $10.00 |
| **Dev (current)** | Gemini 2.5 Flash | ~0.72 | — | $0.15 | $0.60 |

GPT-4.1 was the prior recommendation — GPT-5.4 is now OpenAI's current frontier model.

#### Embeddings (retrieval quality)

| Model | MTEB score | Context | Cost/1M tokens | Notes |
|---|---|---|---|---|
| **Qwen3-Embedding-8B** | **70.58** (#1 multilingual) | 32K | Free (self-hosted) | Apache 2.0; best for self-hosted at scale |
| **Gemini embedding-001** | **68.32** (#1 API model) | 2,048 | ~$0.15 | Jan 2026 release; default API choice |
| Voyage-3.1-large | ~67.7 | 32K | **$0.05** | Best commercial price-quality |
| Cohere embed-v4 | 65.2 | **128K** | $0.10 | Only model with 128K context; best for long docs |
| text-embedding-3-large | 64.6 | 8K | $0.13 | **Now overpriced** — 3.7 MTEB points below Gemini embedding-001 at same price |

Evomap recommendation: Gemini embedding-001 for API usage (no infra overhead, MTEB #1 API). Voyage-3.1-large if budget is the constraint. Drop text-embedding-3-large — it's no longer competitive.

#### Reranking

| Model | ELO | nDCG@10 | Latency | Price/1M | License |
|---|---|---|---|---|---|
| **Zerank 2** | **1638** (#1) | 0.079 | 265ms | $0.025 | CC-BY-NC-4.0 — verify commercial use |
| Cohere Rerank 4 Pro | 1629 | 0.095 | 614ms | $2.50 | Commercial |
| **Voyage AI Rerank 2.5** | 1544 | **0.110** (best precision) | 613ms | **$0.05** | Commercial |
| Voyage AI Rerank 2.5 Lite | 1520 | 0.103 | 616ms | $0.02 | Commercial |
| Cohere Rerank 3.5 | 1451 (#10) | 0.080 | 392ms | $2.00/1K searches | Commercial |

Evomap recommendation: **Voyage AI Rerank 2.5** ($0.05/M, commercial-safe, highest nDCG@10 precision). Cohere Rerank 3.5 is now #10 — don't use for new deployments.

### Current accuracy gap

Dev system: **54% on multi-hop financial QA** (FinanceBench-style). Industry production: **72-83%**. Gap breakdown:

| Gap source | Delta | Fix |
|---|---|---|
| Synthesis model quality | ~10% | Claude Sonnet 4.6 vs Flash |
| Embedding quality | ~8% | Gemini embedding-001 vs Gemini text-embedding-004 |
| Cypher reliability | ~6% | Claude Sonnet 4.6 vs Flash for Cypher |
| Retrieval failures (keyword boosting) | ~4% | Not yet implemented |

### The $25 guard — still applies

Never enable thinking on Gemini Flash. `thinkingBudget: 0` must be passed for all non-Pro Gemini models. Gemini Pro mandates thinking — do not use without an explicit token budget decision.

---

## 11. Build Order (if starting from scratch)

1. **Ingestion pipeline** — Docling + Haiku extraction → Postgres. Foundation everything else sits on.
2. **Profile Enricher** — nightly enrichment from external APIs. Data moat starts accumulating.
3. **Matching serving layer** — hard filter + pgvector ANN + Cohere rerank. First user-facing feature.
4. **Due Diligence Agent** — highest value, $50/seat feature. Closes enterprise deals.
5. **Portfolio Monitor** — recurring value, drives retention.
6. **Outreach Personalizer + LP Reports** — expansion revenue, different buyers within the firm.
7. **Market Mapper** — thesis-level product, targets partners not analysts.
