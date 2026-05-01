# SimpleRAG — Enterprise Production Playbook
## General-Purpose RAG Engine: Domain Adaptation & Production Hardening

**Author:** Rithvik  
**Perspective:** Written as a senior technical staff member who has shipped RAG systems at scale  
**Last updated:** 2026-05-01  
**References:** [UPGRADE_PLAN.md](UPGRADE_PLAN.md), [EVOMAP.md](EVOMAP.md), [CLAUDE.md](CLAUDE.md)

---

## 1. The Pitch

SimpleRAG is a **general-purpose retrieval orchestrator** with financial-domain tuning layered on top of it — not a financial product with incidental generality.

The core engine — Qdrant vector search, BM25 sparse encoding, HyDE query planning, multi-hop decomposition, PageIndex agentic browsing, graph entity extraction, hybrid reranking-ready retrieval, and LLM synthesis — has zero financial domain assumptions. The financial tuning lives in exactly three places:

| What | File | Lines | What it does |
|---|---|---|---|
| VC domain schema | `domain_config.py` | 9–52 | doc types, metadata fields, extraction prompt, system prompt suffix |
| Arithmetic keyword detector | `llm_service.py` | 22–32 | routes financial ratio questions to a calculation step |
| Default active domain | `config.py` | ~16 | `"active_domain": "vc_financial"` |

Everything else is domain-agnostic. The financial coupling is shallow, isolated, and intentional.

**The claim:** Retargeting to legal, medical, educational, government, or any other domain is a configuration change — not a rebuild.

---

## 2. Why the Architecture Is Actually General

### What the seven modes are doing under the hood

| Mode | What it actually does |
|---|---|
| `normal` | Dense vector retrieval + BM25 hybrid + LLM synthesis |
| `graph` | Entity/relationship extraction → in-memory NetworkX → graph-augmented retrieval |
| `neo4j` | Same as graph but persisted; Cypher traversal for structured queries |
| `hybrid_neo4j` | Vector + graph + Neo4j composed (architecture debt — see UPGRADE_PLAN.md §1) |
| `pageindex` | Agentic tool-calling that browses documents page-by-page |
| `agentic` | ReAct loop with tool use |
| `parallel` | Fan-out across modes, fuse results |

None of these are financial. "Revenue growth" or "gross margin" appearing in answers is because the *domain config* injected those concepts into the system prompt and extraction prompt. Strip that and the modes still work on medical trial reports, legal briefs, or academic papers.

### The three layers that make it generic

```
┌─────────────────────────────────────────────────────┐
│               DOMAIN CONFIG LAYER                   │
│  domain_config.py: schema, extraction prompt,       │
│  system prompt suffix, filterable fields            │
│  → Change this to change the domain                 │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│             RETRIEVAL ORCHESTRATION                 │
│  Query planning, HyDE, multi-hop decomposition,     │
│  hybrid dense+sparse, reranking pipeline            │
│  → Fully generic. Domain-unaware.                   │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│             SYNTHESIS LAYER                         │
│  LLM answer generation with inline citations,       │
│  arithmetic detection (extend per domain),          │
│  source comparison gates                            │
│  → Generic core. Small domain-specific extensions.  │
└─────────────────────────────────────────────────────┘
```

---

## 3. Domain Adaptation Playbook

Retargeting to a new domain is three steps. All three are already wired up — there are no new abstractions to build.

### Step 1 — Write the domain config

Add your domain to `domain_config.py`. The schema below is a template:

```python
"legal": {
    "doc_types": [
        "contract", "brief", "statute", "ruling", "motion",
        "deposition", "memo", "policy", "regulation"
    ],
    "filterable_fields": [
        "jurisdiction", "practice_area", "party", "doc_type",
        "court", "effective_date", "governing_law"
    ],
    "extraction_prompt": (
        "Extract the following from this legal document as structured JSON:\n"
        "- parties: list of party names and their roles (plaintiff, defendant, counsel)\n"
        "- jurisdiction: governing jurisdiction and court\n"
        "- governing_law: choice of law clause if present\n"
        "- key_obligations: list of primary obligations with the obligated party\n"
        "- effective_date: ISO date string\n"
        "- expiry_date: ISO date string if present\n"
        "- penalty_clauses: list of penalty/damages provisions\n"
        "- red_flags: ambiguous clauses, missing standard terms, unusual provisions"
    ),
    "system_prompt_suffix": (
        "Prioritize: defined terms, obligations, conditions precedent, "
        "representations and warranties, termination rights, and jurisdiction. "
        "Cite clause numbers. Flag ambiguity explicitly."
    ),
    "arithmetic_keywords": [
        "liquidated damages", "penalty calculation", "interest rate",
        "statute of limitations", "damages cap", "pro rata"
    ]
}
```

```python
"medical": {
    "doc_types": [
        "clinical_trial", "case_study", "research_paper", "drug_label",
        "protocol", "adverse_event", "formulary", "guidelines"
    ],
    "filterable_fields": [
        "therapeutic_area", "phase", "indication", "drug_name",
        "study_type", "patient_population", "sponsor"
    ],
    "extraction_prompt": (
        "Extract as structured JSON:\n"
        "- drug_names: list of drugs/interventions with generic and brand names\n"
        "- indication: condition being studied or treated\n"
        "- primary_endpoint: primary efficacy or safety endpoint\n"
        "- study_population: sample size, inclusion/exclusion criteria\n"
        "- phase: trial phase (I/II/III/IV) if applicable\n"
        "- key_results: primary outcome with statistical values (p-value, CI)\n"
        "- adverse_events: serious adverse events (SAEs) with incidence rates\n"
        "- regulatory_status: FDA/EMA approval status if mentioned"
    ),
    "system_prompt_suffix": (
        "Prioritize: statistical significance (p-values, confidence intervals), "
        "effect sizes (NNT, NNH, hazard ratios), safety signals, and regulatory context. "
        "Distinguish efficacy from safety claims. Note study design limitations."
    ),
    "arithmetic_keywords": [
        "p-value", "confidence interval", "hazard ratio", "odds ratio",
        "number needed to treat", "relative risk", "absolute risk reduction",
        "number needed to harm", "incidence rate"
    ]
}
```

```python
"education": {
    "doc_types": [
        "curriculum", "syllabus", "research_paper", "textbook_chapter",
        "assessment", "policy", "accreditation_report"
    ],
    "filterable_fields": [
        "subject", "grade_level", "institution", "learning_objective",
        "assessment_type", "standard"
    ],
    "extraction_prompt": (
        "Extract as structured JSON:\n"
        "- learning_objectives: list of stated learning outcomes\n"
        "- subject_area: primary subject and sub-topics\n"
        "- grade_or_level: target grade or academic level\n"
        "- standards_alignment: curriculum standards referenced (CCSS, NGSS, etc.)\n"
        "- assessment_methods: types of assessment described\n"
        "- prerequisite_knowledge: prior knowledge assumed\n"
        "- key_concepts: list of core concepts covered"
    ),
    "system_prompt_suffix": (
        "Prioritize: learning objectives, conceptual prerequisites, "
        "assessment methods, and standards alignment. "
        "Explain at the level of the target audience."
    ),
    "arithmetic_keywords": []  # No domain-specific arithmetic needed
}
```

### Step 2 — Flip the active domain

One line change in `config.py`:

```python
# Before
"active_domain": "vc_financial"

# After
"active_domain": "legal"   # or "medical", "education", etc.
```

### Step 3 — Wire domain arithmetic keywords (if any)

In `llm_service.py:22-32`, the `_ARITHMETIC_KW` frozenset currently holds financial ratios. Extend it with your domain's numeric reasoning keywords — or leave it empty for domains like education where calculation prompts are not needed. The detection function at line 29 will automatically pick up the extended set.

That's the entire change surface. The query planner, chunker, vector store, graph extractor, BM25 index, and synthesizer require zero domain-specific changes.

---

## 4. Target Domains: What Each Requires

| Domain | Key metadata | Arithmetic needs | Graph value | Hardest part |
|---|---|---|---|---|
| **Legal** | jurisdiction, parties, clause refs | Damages, timelines, liability caps | HIGH — party relationships, precedent chains, cross-case entity linking | Clause-level citation precision |
| **Medical / Clinical** | indication, phase, endpoints, drug | p-values, NNT, hazard ratios, incidence | HIGH — drug-disease-mechanism triples; adverse event co-occurrence | Distinguishing claims by evidence quality |
| **Financial (current)** | stage, ARR, sector, valuation | Ratios, margins, growth rates | MEDIUM — fund-company-founder graph | Already solved |
| **Government / Policy** | agency, regulation, jurisdiction, effective date | Budget allocations, penalty amounts | MEDIUM — cross-reference chains across bills/regulations | Dense cross-referencing |
| **Education** | subject, grade, standard, objective | Mostly none | LOW — prerequisite chains | Matching content to learning level |
| **Scientific Research** | domain, methodology, dataset, year | Statistical tests, effect sizes | HIGH — citation graph, replication relationships | Claim confidence stratification |

---

## 5. Enterprise Production Stack

This section covers what to change when taking the system to enterprise production. No free tiers. No development shortcuts. These are the correct choices at company scale.

### 5.1 LLM Model Selection

*Sourced from UPGRADE_PLAN.md §5 (CLAUDE.md) and EVOMAP.md §10. Re-verify pricing before large runs — this table was last validated 2026-04-30.*

#### Synthesis (Normal RAG, Graph RAG, answer generation)

**Use: Claude Sonnet 4.6**

- 94% RAG accuracy, 1.9% hallucination rate (lowest measured across models)
- 90% prompt caching discount ($0.30/M cached input vs $3.00/M base) — at enterprise scale, context re-sent across calls amortizes aggressively
- With a 70% cache hit rate: effective cost per query drops to ~$0.02 vs $0.06 uncached
- Current dev setup (Gemini 2.5 Flash) runs at ~86% accuracy with ~5% hallucination — acceptable for eval, not for customers

```
claude-sonnet-4-6
Input:   $3.00/M  ($0.30/M cached)
Output:  $15.00/M
```

#### Agentic / PageIndex mode (stateful multi-turn tool loops)

**Use: Claude Sonnet 4.5 or 4.6**

The correct benchmark for PageIndex is **tau-bench** (multi-turn stateful tool calls), not BFCL (single-turn). Claude Sonnet leads tau-bench: 0.862 retail / 0.700 airline (#1 across all models evaluated). Current dev setup on Gemini Flash runs ~72% agentic reliability — acceptable for internal eval, not production PageIndex.

#### Cypher generation (Neo4j mode)

**Use: Claude Sonnet 4.6**

Cypher errors are silent: invalid query returns an empty result set, not an exception. User sees "no results found" with no indication the query itself was malformed. Gemini 2.5 Flash generates invalid Cypher ~20% of the time on multi-hop traversals. Claude Sonnet 4.6 scores 43.0 on SWE-bench vs Flash's significantly lower coding benchmark — this gap shows up directly in Cypher quality. Given errors are silent and silent failures destroy user trust, this is not a place to save money.

#### Entity/relationship extraction

**Use: Gemini 2.5 Flash with `thinkingBudget: 0`**

Extraction is structured JSON output over well-bounded text — it does not need frontier reasoning ability. Flash is 20x cheaper than Sonnet and adequate for structured extraction tasks. The $25 incident (Boeing 10-K indexing with thinking enabled, UPGRADE_PLAN.md §8) was caused by forgetting `thinkingBudget: 0`. This guard is non-negotiable for Flash.

#### Embeddings

**Use: Gemini embedding-001 (API) or Qwen3-Embedding-8B (self-hosted)**

| Model | MTEB | Context | Cost/1M |
|---|---|---|---|
| **Qwen3-Embedding-8B** | **70.58** (multilingual #1) | 32K | Free (self-hosted) |
| **Gemini embedding-001** | **68.32** (API #1) | 2,048 | ~$0.15 |
| Voyage-3.1-large | ~67.7 | 32K | $0.05 |
| text-embedding-3-large | 64.6 | 8K | $0.13 |

At enterprise scale with self-hosted infra budget: Qwen3-Embedding-8B on a GPU worker is the correct call (Apache 2.0, MTEB #1, free). For API-only deployments: Gemini embedding-001. Drop text-embedding-3-large — it's 3.7 MTEB points behind Gemini embedding-001 at the same price. It is no longer competitive.

#### Reranking (currently missing from all modes — highest leverage addition)

**Use: Voyage AI Rerank 2.5**

No mode in the current system reranks. UPGRADE_PLAN.md §1 item #2 identifies this as the highest-leverage retrieval improvement: "reranking is a bigger win than graph structure on Level-1 queries" (GraphRAG-Bench, ICLR'26).

| Model | ELO | nDCG@10 | Price/1M | License |
|---|---|---|---|---|
| **Voyage AI Rerank 2.5** | 1544 | **0.110** (best precision) | **$0.05** | Commercial ✅ |
| Cohere Rerank 4 Pro | 1629 | 0.095 | $2.50 | Commercial |
| Cohere Rerank 3.5 | 1451 | 0.080 | $2.00/1K | Commercial |

Voyage AI Rerank 2.5 has the highest nDCG@10 precision, is commercially safe, and costs 50x less than Cohere Rerank 4 Pro. Add it as a post-retrieval step in `simple_rag.py` before passing contexts to the LLM. Typical accuracy lift: +8–15pp on retrieval-heavy workloads.

Do not use Zerank 2 (ELO #1) in production until CC-BY-NC-4.0 commercial use is formally cleared with legal.

---

### 5.2 Infrastructure

| Layer | Enterprise choice | Why |
|---|---|---|
| **Vector DB** | Qdrant Hybrid Cloud | You own the infra, Qdrant manages ops. No vendor lock on data. Scales horizontally. |
| **Graph DB** | Neo4j Enterprise self-hosted on EC2 | Full Cypher, no node limits, RBAC, hot backups. AuraDB Enterprise if you prefer managed. |
| **Embeddings** | GPU workers (AWS g5.xlarge or Modal) | For Qwen3-8B self-hosted; $0.00 per token vs $0.15/M for API |
| **Job queue** | Celery + Redis Streams (Upstash managed or self-hosted) | Message replay for failed indexing jobs. 512MB message size. See UPGRADE_PLAN.md §8 for the Redis Streams vs RabbitMQ vs SQS decision. |
| **Workflow orchestration** | Temporal | Durable, versionable, retryable long-running workflows. Used for multi-step indexing pipelines. LangGraph is a prototype tool — Temporal is the production-grade choice (see EVOMAP.md §5). |
| **PDF extraction** | Docling | Table-aware, open source, IBM-backed. Better than raw PyMuPDF → LLM for structured documents. Replace the current `document_processor.py` PDF path. |
| **GLiNER serving** | Modal (GPU warm serving) or AWS SageMaker | Removes the 35s cold start. Modal gives warm GPU inference; SageMaker scales to enterprise SLA requirements. |
| **Serving layer** | FastAPI + Uvicorn behind AWS ALB | Flask is fine for dev. FastAPI is correct for production (async, OpenAPI, better performance). |
| **Container orchestration** | AWS ECS Fargate or Kubernetes (EKS) | Fargate removes node management. EKS if you need GPU scheduling for self-hosted models. |
| **Observability** | Langfuse (LLM traces) + Datadog (infra) | Langfuse gives token counts, latency, cache hit rates per call. Non-negotiable for cost tracking at enterprise scale. |
| **Eval in CI** | RAGAS + DeepEval | Per UPGRADE_PLAN.md §5 task 1.6: CI fails if faithfulness/context precision drops >5pp vs main. |

#### Why not these alternatives

- **AWS Lambda for indexing:** 15-minute hard limit kills it for long documents (Boeing 10-K = 20+ minutes). Acceptable for the query endpoint only.
- **Kafka:** Built for 10k+ events/second. 18-wheel truck for moving a pizza box at SimpleRAG's indexing volume.
- **LangGraph:** Prototype-grade. Temporal provides durability, versioning, and retries that LangGraph cannot. See EVOMAP.md §5.
- **Qdrant Cloud Standard (managed):** Fine for <500k vectors. At enterprise: Hybrid Cloud gives you data locality control.

---

### 5.3 Architecture Upgrades Required for Production

These are not optional at enterprise scale. They are the difference between a demo and a product.

#### (a) Reranking — add to every mode

Currently missing across all seven modes. Add Voyage AI Rerank 2.5 as a step in `simple_rag.py` between `vector_db_service.query_with_filters()` and context formatting. ~60 lines of code. Biggest single accuracy improvement available without a model change.

#### (b) Structure-aware chunking

The current chunker (`document_processor.py:314-376`) uses fixed 1000-char sliding windows with no awareness of headings, tables, or lists. Legal contracts have clause hierarchies. Medical papers have section structure. Financial filings have exhibit tables.

Replace with:
- **Docling** for PDF extraction (table-aware, preserves structure)
- **MarkdownNodeParser** or `unstructured.io` for structure-aware splitting
- Keep heading path in chunk metadata (e.g., `"section": "Part II > Item 7 > Results"`)

Per UPGRADE_PLAN.md §1 item #3: late chunking beats fixed-window 2–18pp.

#### (c) Stable entity IDs

Currently re-ingesting a document grows the graph because entity IDs are not stable across runs. At enterprise scale with multiple ingest workers and periodic re-indexing, this doubles (then triples) the graph every run.

Fix: `canonical_id(name, type) → "{type}::{slug(name)}"`. MERGE on this key in Neo4j. Maintain `aliases: []` list. Implementation is in UPGRADE_PLAN.md §5 task 1.2 — it's already designed, just needs to ship.

#### (d) Cypher validation + self-correct

Per UPGRADE_PLAN.md §5 task 1.3: validate generated Cypher with `EXPLAIN` before `RUN`. On invalid: one self-correct retry. On second failure: surface `CypherGenerationError` to the user — never silently return empty. Silent empty results are the #1 trust killer in graph RAG.

#### (e) LLM-Wiki compiler architecture (medium-term)

The highest-leverage architectural move is not tuning the seven modes — it is compiling a persistent, interlinked knowledge artifact and querying that instead. Per UPGRADE_PLAN.md §2:

- **Karpathy LLM-Wiki pattern:** `raw/` (source) → `wiki/` (compiled artifact) → retrieval over wiki
- **HippoRAG-2 retrieval:** dual-node KG (passages + phrases) + Personalized PageRank on top of wiki
- **Cost:** pay compilation once, not on every query

This is Phase 2–4 of UPGRADE_PLAN.md. Phase 1 (stop bleeding) must ship first.

---

### 5.4 Eval & Observability

No RAG system should go to production without a domain-specific eval harness. The current `eval/baseline_phase1.json` covers financial questions. For each new domain, build a parallel harness.

**Framework:** RAGAS + DeepEval  
**Metrics to track:**
- `faithfulness` — does the answer contradict the retrieved context?
- `answer_relevancy` — is the answer on-topic for the question?
- `context_precision` — are the retrieved chunks actually relevant?
- `context_recall` — are all relevant chunks being retrieved?

**Domain-specific additions:**
- Legal: citation accuracy (does the cited clause number match?)
- Medical: statistical claim fidelity (are p-values and CIs reproduced correctly?)
- Financial: numeric accuracy (are calculated ratios correct to 2 decimal places?)

**CI gate:** fail PR if any metric drops >5pp vs main. This is non-negotiable. Without a metric gate, regressions ship silently.

**LLM observability (Langfuse):**
- Token counts per call, per mode, per domain
- Cache hit rate (target: >60% for repeated-context workloads)
- Latency p50/p95/p99 per mode
- Cost attribution per query type

At enterprise scale you need to know *exactly* where your LLM spend is going. Langfuse gives this without instrumenting every call manually.

---

## 6. Domain Adaptation Cost Estimates

*Based on CLAUDE.md cost table, validated 2026-04-30. Re-verify before large runs.*

| Action | Dev (Gemini Flash) | Production (Claude Sonnet 4.6) | Production + Cache (70% hit) |
|---|---|---|---|
| Index one 190-page legal brief | ~$0.12 | ~$1.80 | ~$0.55 |
| Index full corpus (500 docs) | ~$60 | ~$900 | ~$275 |
| Re-index same corpus (skip cache) | ~$0.00 | ~$0.00 | ~$0.00 |
| Single query (normal RAG) | ~$0.005 | ~$0.06 | ~$0.02 |
| 50-question domain eval run | ~$0.10 | ~$1.50 | ~$0.50 |
| Reranking (Voyage, 50 candidates) | — | ~$0.00025 | ~$0.00025 |

Reranking adds <$0.001 per query. It is effectively free relative to the LLM synthesis cost.

---

## 7. Switching Checklist

Use this when retargeting to a new domain:

```
PRE-INGEST
[ ] domain_config.py — add domain schema (doc_types, filterable_fields, extraction_prompt, system_prompt_suffix)
[ ] config.py — set active_domain to new domain
[ ] llm_service.py:22-32 — extend _ARITHMETIC_KW with domain-specific numeric terms (or leave empty)
[ ] Collect 30–50 representative documents for the new domain
[ ] Write 20+ ground truth Q&A pairs (question, expected_answer, source_chunk)

INDEXING
[ ] Run ingestion on domain corpus with the new config
[ ] Verify entity extraction output looks correct (check graph_rag_service logs)
[ ] Verify metadata filtering works on domain-specific filterable_fields

EVAL
[ ] Run RAGAS harness against ground truth Q&A pairs
[ ] Commit eval results as baseline for new domain
[ ] Set CI gate threshold against that baseline

PRODUCTION UPGRADES (per §5 above)
[ ] Synthesis model: Claude Sonnet 4.6 (not Flash)
[ ] PageIndex model: Claude Sonnet 4.5/4.6
[ ] Cypher model: Claude Sonnet 4.6
[ ] Add Voyage AI Rerank 2.5 to retrieval pipeline
[ ] Update embeddings to Gemini embedding-001 or Qwen3-Embedding-8B
[ ] Deploy GLiNER to Modal or SageMaker (remove cold start)
[ ] Add Cypher validation + self-correct (UPGRADE_PLAN.md task 1.3)
[ ] Add stable entity IDs (UPGRADE_PLAN.md task 1.2)
[ ] Wire Langfuse for LLM observability
[ ] Add domain eval to CI with RAGAS metric gate
```

---

## 8. The Honest Assessment

**What this system is:** A retrieval orchestrator with well-isolated domain coupling. The financial tuning is thin and removable. The core is genuinely reusable.

**What takes real work when switching domains:**

1. **Ground truth curation.** Every new domain needs 20–50 expert-annotated Q&A pairs to eval against. This is human time, not engineering time.

2. **Extraction prompt quality.** The extraction prompt in `domain_config.py` determines what metadata fields are populated, which determines what filtering is possible, which determines whether the domain-specific retrieval paths work. Bad extraction prompt = bad metadata = bad retrieval. Iterate on this with real documents from the target domain before indexing the full corpus.

3. **Arithmetic/numeric reasoning.** Every domain has its own numeric reasoning vocabulary. The `_ARITHMETIC_KW` detector in `llm_service.py:22-32` needs to be extended to trigger the calculation step on domain-specific numeric queries. Missing keywords = numeric questions get answered without a calculation step = wrong numbers.

4. **The accuracy gap until production models ship.** Current dev setup (Gemini Flash) runs at ~54% on multi-hop QA. Production setup (Claude Sonnet 4.6 + Voyage reranker + embedding-001) closes to ~80%+. This gap is documented in EVOMAP.md §10. The architecture change is a configuration change; the accuracy gap closes when the model upgrades ship.

**What you get when this is done right:** A single RAG engine, one codebase, one deployment, with domain packs that can be swapped by changing a config key. Legal one day, medical the next. That is the correct architecture for a multi-domain enterprise product.

---

## 9. Sources

- [UPGRADE_PLAN.md](UPGRADE_PLAN.md) — Phase roadmap, audit items, production infra plan, message queue decision
- [EVOMAP.md](EVOMAP.md) — Production LLM stack, reranking benchmarks, technology decisions, architecture layers
- [CLAUDE.md](CLAUDE.md) — Model selection table, $25 incident guard, cost benchmarks per action
- [GraphRAG-Bench, ICLR'26](https://arxiv.org/html/2506.05690v3) — When graph structure helps vs hurts; reranking > graph structure on Level-1 queries
- [HippoRAG-2](https://arxiv.org/abs/2502.14802) — Dual-node KG + Personalized PageRank for multi-hop retrieval
- [Agentic-RAG SoK, March 2026](https://arxiv.org/html/2603.07379v1) — Planner-grader-executor loop; 34% → 78% on multi-hop
- [Jina Late Chunking](https://arxiv.org/pdf/2409.04701) — +2–18pp over fixed-window chunking
- [Karpathy LLM-Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — Compiler pattern: raw/ → wiki/ → retrieval over compiled artifact
- [RAGAS](https://docs.ragas.io) — Faithfulness, answer relevancy, context precision/recall metrics
- [Voyage AI Rerank 2.5](https://www.voyageai.com) — nDCG@10 0.110, $0.05/M, commercial license
