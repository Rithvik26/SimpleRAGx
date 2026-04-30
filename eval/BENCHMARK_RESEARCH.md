# Benchmark Research — Normal RAG & Graph RAG

Researched: 2026-04-29 | Source: web research + primary papers  
**Warning:** LLM pricing and model rankings change frequently. Re-verify before acting.

---

## 1. Best Benchmark for Normal RAG

### Recommended: FinanceBench (primary) + RAGAS metrics (secondary)

**FinanceBench** — [HuggingFace](https://huggingface.co/datasets/PatronusAI/financebench) | [GitHub](https://github.com/patronus-ai/financebench) | [arXiv 2311.11944](https://arxiv.org/abs/2311.11944)

- **Why best for this system:** We already have AMD/Boeing 10-K PDFs indexed. FinanceBench questions are ground-truth verified against real SEC filings. The eval methodology (answer accuracy on open-book financial QA) is exactly what this system is built for.
- **Dataset:** 10,231 questions over 361 public company filings (10-Ks, 10-Qs, 8-Ks). A 150-question annotated sample is on HuggingFace.
- **Baseline accuracy:**
  - Basic vector-store RAG (GPT-4-Turbo or Llama-2): **~19% accuracy** (fails 81% of questions)
  - Improved RAG (better chunking + retrieval): **~76% accuracy**
  - Our system with Gemini Flash: we reported **~70% on PageIndex mode** (14 questions)
- **Metric:** LLM-as-judge pass rate (exact-match numeric tolerance, equivalent representations allowed)
- **Our current 14-question subset:** Representative sample, not statistically complete. 150+ questions recommended for a publishable result.

**RAGAS** — [docs.ragas.io](https://docs.ragas.io) — for retrieval quality metrics:
- faithfulness, answer_relevancy, context_precision, context_recall
- Good system target: **≥ 0.80** on all four metrics
- **Current status:** Previously invalid (contexts were always empty — bug fixed 2026-04-29)
- Re-run with updated harness to get valid RAGAS scores.

### Why NOT BEIR for this system

- BEIR FiQA (financial QA) is a retrieval-only benchmark (NDCG@10) — it measures how well the retriever ranks relevant documents, not how well the full RAG pipeline answers questions.
- BEIR requires a pre-built corpus and relevance judgments that don't match our PDF-first setup.
- Use BEIR if you want to isolate and benchmark the Qdrant retriever in isolation.

### Why NOT TAT-QA or FinQA

- **TAT-QA** (16,552 questions, F1 metric, [GitHub](https://github.com/NExTplusplus/TAT-QA)): Explicitly requires bridging tabular + textual content. Human expert F1 is 90.8%; SOTA is 58.0%.
- **FinQA** (execution accuracy, [website](https://finqasite.github.io/)): Requires numerical reasoning across structured table data. GPT-4 reaches ~76%; human experts reach 91%.
- **Both are unfair for this system:** Our PDF extractor (pymupdf) converts tables to jumbled text. We do not have a structured table extractor. Running against these benchmarks would undercount performance due to parser limitations, not RAG limitations.
- **Verdict:** Do not run TAT-QA or FinQA until a proper table extractor is implemented (pdfplumber, Camelot, or a vision-based approach).

---

## 2. Best Benchmark for Graph / Neo4j RAG

### Recommended: MultiHop-RAG (primary) + GraphRAG Sensemaking (secondary)

**MultiHop-RAG** — [HuggingFace](https://huggingface.co/datasets/yixuantt/MultiHopRAG) | [GitHub](https://github.com/yixuantt/MultiHopRAG) | [arXiv 2401.15391](https://arxiv.org/abs/2401.15391) | COLM 2024

- **Why best for graph mode:** Tests whether the system can retrieve evidence scattered across 2-4 documents and synthesize a correct multi-hop answer — exactly the use case where graph traversal should help over single-vector retrieval.
- **Dataset:** 2,556 multi-hop queries over English news articles from mediastack API. Each query requires evidence from 2-4 separate articles.
- **Baseline accuracy:**
  - GPT-4 with standard RAG: **0.56** (56%) accuracy
  - GPT-4 with improved multi-hop methods: **0.63** (63%), +12% improvement
  - PaLM with standard RAG: 0.47; improved: 0.61, +26%
- **Metrics:** Answer accuracy (LLM-judge), evidence hit rate (what fraction of required evidence items appear in retrieved contexts), per-hop-count breakdown.
- **Setup:** Requires downloading the full news article corpus from HuggingFace and indexing it. `run_benchmarks.py` handles this.
- **Caveat:** The MultiHop-RAG news corpus is different from our financial docs. You must re-index the news corpus for fair evaluation.

**GraphRAG Global Sensemaking** — [Microsoft GraphRAG](https://arxiv.org/abs/2404.16130) | [BenchmarkQED](https://github.com/microsoft/benchmark-qed)

- **Why relevant:** The original Microsoft GraphRAG paper (2404.16130) uses pairwise LLM judge on comprehensiveness, diversity, empowerment, and relevance. This is the industry-standard way to compare graph vs. normal RAG on thematic global queries.
- **Datasets:** Podcast transcripts (70 episodes of "Behind the Tech") and AP News health articles (1,397 articles).
- **Baseline results from the GraphRAG paper:**
  - Comprehensiveness: GraphRAG wins 72-83% pairwise vs vector RAG (Podcast/News)
  - Diversity: GraphRAG wins 75-82% pairwise
  - Token efficiency: 26-33% fewer tokens with low-level summaries; 97%+ fewer with root-level summaries
- **CRITICAL NOTE:** These results are for Microsoft GraphRAG WITH community summaries. Our system does NOT have community summaries. Our "graph mode" will likely show smaller or no improvement over normal RAG on global sensemaking queries. This is by design and should be stated clearly.
- **Setup in `run_benchmarks.py`:** Uses AMD/Boeing docs as the corpus with broad thematic questions. Not a fair reproduction of the BenchmarkQED datasets — but a useful relative comparison between our modes.

### Why NOT HotpotQA / MuSiQue / 2WikiMultiHopQA

- These are Wikipedia-domain multi-hop datasets. Running them requires indexing Wikipedia or using the provided Wikipedia passages.
- Feasible but requires more setup. MultiHop-RAG is better because it uses news articles (similar format to financial docs) and is simpler to set up.
- Use HotpotQA if you want a domain-agnostic multi-hop benchmark that doesn't require a custom corpus.

---

## 3. Benchmark Setup Commands

### Suite 1: FinanceBench (existing AMD/Boeing docs)

```bash
# Requires AMD/Boeing PDFs in eval/docs/
# Index and evaluate in normal mode
python eval/run_benchmarks.py --suite financebench --modes normal graph --n 14

# Use existing app collections (skips re-indexing)
python eval/run_benchmarks.py --suite financebench --modes normal graph --n 14 --use-default-collections
```

### Suite 2: MultiHop-RAG

```bash
# Install datasets package
pip install datasets

# Download MultiHop-RAG and run (uses scaffold questions if download fails)
python eval/run_benchmarks.py --suite multihop --modes normal graph --n 25

# To use the real MultiHop-RAG news corpus, you must first download and index it:
# python -c "
# from datasets import load_dataset
# ds = load_dataset('yixuantt/MultiHopRAG', split='corpus')
# # ds contains the news articles to index
# "
```

### Suite 3: GraphRAG Sensemaking

```bash
# Pairwise comparison: normal vs graph (use existing app collections)
python eval/run_benchmarks.py --suite sensemaking --modes normal graph --use-default-collections
```

### Run all suites

```bash
python eval/run_benchmarks.py --suite all --modes normal graph --n 14
```

### RAGAS harness (after fixing context capture bug)

```bash
# Runs all RAGAS metrics on normal and graph modes
python eval/ragas_harness.py --modes normal graph --out eval/results/ragas_$(date +%Y%m%d).json
```

---

## 4. RAGAS Metric Targets

| Metric | Poor | Acceptable | Good (target) |
|---|---|---|---|
| faithfulness | < 0.5 | 0.5–0.75 | **≥ 0.80** |
| answer_relevancy | < 0.5 | 0.5–0.75 | **≥ 0.80** |
| context_precision | < 0.5 | 0.5–0.75 | **≥ 0.80** |
| context_recall | < 0.5 | 0.5–0.75 | **≥ 0.80** |

Previous baseline (baseline_phase1.json): faithfulness=0.036, answer_relevancy=0.0, context_precision=0.0, context_recall=0.0. All invalid due to the empty-context bug.

---

## 5. TAT-QA and FinQA — Setup Prerequisites (If/When Implemented)

These benchmarks can only be run fairly after implementing table extraction. Required additions:

1. **Table extraction:** Use `pdfplumber`, `camelot-py`, or a vision model (Gemini vision) to extract structured tables from PDFs.
2. **Table-to-text conversion:** Convert tables to markdown or structured text that preserves row/column semantics.
3. **Hybrid chunker:** Separate chunking strategies for table rows vs. narrative text.

Without these, TAT-QA F1 scores will be severely underestimated and do not reflect the system's true capability.

**TAT-QA download:**
```bash
git clone https://github.com/NExTplusplus/TAT-QA
# Dataset: TAT-QA/dataset_raw/{train,dev,test}_original.json
```

**FinQA download:**
```bash
# HuggingFace: contextqa/finqa or official site
python -c "from datasets import load_dataset; ds = load_dataset('contextqa/finqa')"
```

---

## 6. Neo4j GraphRAG Official Retrievers

**Package:** `neo4j-graphrag` (formerly `neo4j-genai`)  
**GitHub:** https://github.com/neo4j/neo4j-graphrag-python  
**Docs:** https://neo4j.com/docs/neo4j-graphrag-python/current/

Available retriever types:

| Retriever | Indexes required | Use case |
|---|---|---|
| VectorRetriever | vector index in Neo4j | Pure semantic similarity |
| VectorCypherRetriever | vector index | Semantic + graph traversal via Cypher template |
| HybridRetriever | vector + full-text index | Semantic + keyword search |
| HybridCypherRetriever | vector + full-text | Hybrid + graph traversal (gold standard) |
| Text2CypherRetriever | schema only | NL → Cypher (similar to our current approach) |
| ToolsRetriever | varies | LLM selects retriever per query |

**To use HybridCypherRetriever, we need:**
1. Store entity embeddings in Neo4j node properties
2. `CREATE VECTOR INDEX entity_vector FOR (e:Entity) ON (e.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}`
3. `CREATE FULLTEXT INDEX entity_fulltext FOR (e:Entity) ON EACH [e.name, e.description]`

This is a significant refactor but would enable true hybrid graph-vector retrieval within Neo4j, eliminating the Qdrant-Neo4j split architecture.
