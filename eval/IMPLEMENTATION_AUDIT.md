# Implementation Audit — SimpleRAGx Normal RAG & Graph RAG

Audited: 2026-04-29  
Codebase: `/Users/rithvikgolthi/Desktop/SimpleRAG`  
Auditor: automated inspection + LLM-as-judge cross-check

---

## TL;DR Verdicts

| Component | Grade | Verdict |
|---|---|---|
| Normal RAG — chunking | Prototype | Fixed-size chars, no sentence/section boundary awareness |
| Normal RAG — embeddings | Production | gemini-embedding-001, 768-dim, batch parallel |
| Normal RAG — retrieval | Industry-grade | HyDE + decomposition + RRF + reranking |
| Normal RAG — generation | Industry-grade | Grounded prompt, citations, complexity routing |
| Normal RAG — eval telemetry | **Broken** | `query()` returns plain string; contexts never captured → RAGAS scores invalid |
| Graph — entity extraction | Partially industry-grade | GLiNER NER is good; entity types are reasonable |
| Graph — relationship extraction | Prototype | `RELATES_TO` is a single generic type; relationship as property kills Cypher |
| Graph — canonicalization | Industry-grade | stable `canonical_id`, aliases tracked, suffix stripping |
| Graph — Neo4j indexes | Prototype | btree only, no full-text index, no vector index in Neo4j |
| Graph — traversal persistence | **Fixed** | NetworkX removed from retrieval path; Neo4j is source of truth |
| Graph — community detection | Missing | Does not exist. Not Microsoft-style GraphRAG. |
| Neo4j Text2Cypher safety | Partially industry-grade | EXPLAIN validation + retry, but no read-only guard |
| Overall graph mode | Prototype | Local graph-enhanced RAG, not Microsoft GraphRAG |

---

## 1. Normal RAG

### 1.1 Parsing / Chunking
**Grade: Prototype**

- `document_processor.py` uses **fixed-size character-based chunking** (default: 1000 chars, 200 overlap).
- PDF extraction uses `pymupdf` (`fitz`), which is a good choice and preserves text order better than PyPDF2.
- **Problems:**
  - No sentence-boundary awareness — chunks routinely split mid-sentence.
  - No section-boundary awareness — no detection of headers, paragraphs, tables, footnotes.
  - No layout-aware extraction — columns, headers, page numbers not stripped; they pollute chunks.
  - No table extraction — tables in PDFs are treated as jumbled text. This makes TAT-QA and FinQA unfair benchmarks for this system.
  - PPTX support exists (slide-level chunking), which is better, but PDF is the main format.
- **Industry baseline:** Production systems use recursive splitting with sentence-boundary overlap, or semantic splitting (via embeddings), or section-aware splitting using document structure signals.

### 1.2 Embeddings
**Grade: Production**

- Uses **`gemini-embedding-001`** (768-dim) via direct REST API (`generativelanguage.googleapis.com`), not LiteLLM.
- Batch parallel embedding with `max_workers=5`, batch size 50. Fast.
- Consistent model used for both indexing and querying (same endpoint in `embedding_service.py`).
- **No issue here.**

### 1.3 Qdrant Collection Isolation
**Grade: Industry-grade**

- Separate collections: `collection_name` for document chunks, `graph_collection_name` for graph elements.
- Collections are configured per-instance. Good.
- Connection handling with retry logic and alternative SSL modes.

### 1.4 Query Planning (HyDE + Decomposition)
**Grade: Industry-grade**

- `query_planner.py` implements HyDE (Hypothetical Document Embedding) and query decomposition.
- HyDE generates a hypothetical answer passage then embeds it — reduces query-document embedding gap.
- Decomposition breaks complex queries into 2-4 sub-questions; each retrieved independently then merged via RRF.
- Complexity detection (multi-hop signals) triggers decomposition; simple queries skip it.
- Best-effort: any failure falls back to single-embedding retrieval without breaking the query.
- **This is the strongest part of the Normal RAG pipeline.**

### 1.5 RRF Merge
**Grade: Industry-grade**

- `rrf_merge()` in `query_planner.py` implements Reciprocal Rank Fusion (k=60 default).
- Deduplicates by text hash — same chunk retrieved for multiple sub-queries merges into one entry.
- Scores accumulated across lists; consistent with published RRF papers.

### 1.6 Reranking
**Grade: Industry-grade (when configured)**

- `RerankerService` is Gemini-based reranking. Not a dedicated cross-encoder like `cross-encoder/ms-marco-MiniLM`, but Gemini as a reranker is reasonable.
- Retrieves `top_k * 4` candidates then reranks down to `top_k` — proper oversampling pattern.
- **Gap:** No dedicated bi-encoder or cross-encoder reranker (Cohere Rerank, BGE-reranker, etc.). Gemini API for reranking is expensive per-chunk compared to a local cross-encoder.

### 1.7 Generation Prompt Grounding
**Grade: Industry-grade**

- `llm_service.generate_answer()` sends retrieved contexts to Gemini with a grounding instruction.
- Query complexity routing: simple vs. detail vs. comparison vs. timeline queries get different prompt templates.
- Sources extracted from metadata and cited in answer (seen in `_clean_source_name`).
- Thinking disabled for Flash to avoid cost overrun (CLAUDE.md).

### 1.8 Citations / Sources
**Grade: Partial**

- LLM answer includes source citations in text.
- **BUT:** The `query()` method returns only a plain string — the list of source chunks is not exposed. You cannot programmatically access which chunks were cited.
- This also means the REST API returns only the answer string for normal/graph/neo4j modes.

### 1.9 Eval Telemetry — CRITICAL BUG
**Grade: Broken**

This is the most important finding.

- `simple_rag.py:query()` returns a plain `str` for all modes (normal, graph, neo4j, hybrid_neo4j).
- `ragas_harness.py:_query_mode()` calls `rag.query(question)`, gets a string, then tries `result.get("sources", ...)` — but `.get()` on a string raises `AttributeError`, which is silently caught, so `contexts` becomes `[]`.
- **Result:** RAGAS context_precision and context_recall were always 0.0 in every run.
- **Confirmed:** `baseline_phase1.json` shows `context_precision=0.0`, `context_recall=0.0`. `results_normal.json` shows all `NaN` (even worse — RAGAS may have gotten confused by empty contexts entirely).
- **Fix applied:** Added `query_debug()` to `simple_rag.py` and updated `ragas_harness.py` to call it. The `query()` method is unchanged.

---

## 2. Graph / Neo4j RAG

### 2.1 Entity Extraction Quality
**Grade: Partially industry-grade**

- GLiNER small-v2.1 (50M param local NER model) for entity spotting.
- 8 entity types: PERSON, ORGANIZATION, PRODUCT, TECHNOLOGY, NUMBER, DATE, LOCATION, EVENT.
- Score-based deduplication: if same span appears multiple times, highest-confidence occurrence wins.
- **Good:** Local model, zero API cost for NER, ~0.2s/chunk.
- **Gaps:**
  - GLiNER small is a weaker model than `GLiNER-large-v2.1` or `urchade/gliner_medium-v2.1`. Financial text may benefit from a larger model.
  - No financial-domain-specific entity types (e.g., FINANCIAL_METRIC, REGULATORY_BODY, TICKER_SYMBOL).
  - No co-reference resolution — "the company" after "Boeing" is not resolved.
  - Threshold 0.5 is reasonable but not tuned for this domain.

### 2.2 Relationship Extraction Quality
**Grade: Prototype**

- Gemini 2.5 Flash (not Flash-lite as CLAUDE.md recommends — code says `gemini/gemini-2.5-flash`) is called per chunk for relationship extraction.
- The LLM prompt is tight and focused: given entity names + text, extract relationships.
- Relationships are validated: both endpoints must be in the entity list, source != target.
- **Critical gap: The `relationship` field is a free-text verb phrase** ("founded", "leads", "acquired", etc.), not a typed enum. When stored in Neo4j, all relationships get type `RELATES_TO` and the verb phrase is a property value.

### 2.3 Canonicalization / Aliasing
**Grade: Industry-grade**

- `entity_canonicalizer.canonical_id()` provides stable IDs across re-ingestions.
- Aliases collected and deduplicated (`order-stable dict.fromkeys`).
- Suffix stripping (Inc, Corp, Ltd, etc.) and prefix stripping (The, Dr, Mr, etc.).
- Longest name wins as canonical display name.
- **This is solid.**

### 2.4 Typed Nodes and Typed Relationships — CRITICAL DESIGN FLAW
**Grade: Broken for Cypher**

- **Neo4j nodes:** All entities use single label `Entity` with a `type` property. This means Cypher cannot use label-based filtering (`MATCH (p:PERSON)`). You must use `WHERE e.type = 'PERSON'` which is slower and less expressive.
- **Neo4j relationships:** ALL relationships use a single type `RELATES_TO` with `relationship` as a property value. This means:
  - `MATCH (a:Entity)-[:FOUNDED]->(b:Entity)` is impossible.
  - You must write `MATCH (a)-[r:RELATES_TO]->(b) WHERE r.relationship = 'founded'`.
  - Multi-hop Cypher path queries are severely limited — no semantic relationship typing.
  - The generated Cypher in `generate_cypher_from_question()` correctly uses `RELATES_TO` and filters by `r.relationship CONTAINS ...` — but this is a text search on a property, not a graph traversal on typed edges.
- **Industry standard:** Microsoft GraphRAG, Neo4j official KG builder, and all published graph RAG systems use **typed relationship labels** that map to a schema. `RELATES_TO` is an anti-pattern.

### 2.5 Chunk-to-Entity Links
**Grade: Good**

- Entities have `source_chunks` and `source_texts` lists.
- `EXTRACTED_FROM` relationship from Entity to Document node.
- Source provenance is maintained at the entity level.

### 2.6 Neo4j Indexes
**Grade: Prototype**

Existing indexes (from `create_indexes()`):
```
CREATE INDEX entity_id_idx   FOR (e:Entity)   ON (e.id)
CREATE INDEX entity_name_idx FOR (e:Entity)   ON (e.name)
CREATE INDEX entity_type_idx FOR (e:Entity)   ON (e.type)
CREATE INDEX document_name_idx FOR (d:Document) ON (d.name)
```

- **Missing: Full-text index.** `CONTAINS` on `e.name` and `e.description` is a linear scan. For large graphs this is O(n) per query. Should be:
  ```cypher
  CREATE FULLTEXT INDEX entity_fulltext FOR (e:Entity) ON EACH [e.name, e.description, e.aliases]
  CALL db.index.fulltext.queryNodes("entity_fulltext", "Boeing") YIELD node, score
  ```
- **Missing: Vector index in Neo4j.** Embeddings are in Qdrant only. For `HybridCypherRetriever` (official neo4j-graphrag pattern), Neo4j needs its own vector index to search by embedding and traverse in one query:
  ```cypher
  CALL db.index.vector.queryNodes("entity_vector", 5, $embedding) YIELD node, score
  ```
- **Missing: Relationship type indexes.** Since all relationships are `RELATES_TO`, no relationship index exists. The `relationship` property is queried with `CONTAINS` (linear scan on edge properties).

### 2.7 Graph Traversal and Architecture
**Grade: Industry-grade for the retrieval path (as of 2026-04-29 refactor)**

**NetworkX removed from the production retrieval path.** `search_graph()` now:
1. Qdrant vector search → seed entity names
2. `Neo4jService.traverse_neighbors(seed_names, depth=2)` → Cypher path expansion
3. Dedup and combine; return with provenance (`discovery_method = "neo4j_traversal"`)

If Neo4j is not configured, graph mode falls back to Qdrant vector-only results with an explicit `logger.warning`. Production graph mode requires Neo4j.

NetworkX (`self.graph`) is still built during document indexing (via `_build_graph()`) but is no longer called during query retrieval. It can be removed entirely in a future cleanup.

**Current architecture (post-refactor):**

| Store | Persists? | Used for |
|---|---|---|
| Qdrant doc collection | ✓ | Document chunk retrieval (all modes) |
| Qdrant graph collection | ✓ | Entity/relationship vector similarity (seed finding) |
| Neo4j | ✓ | Graph topology traversal via Cypher — single source of truth |
| NetworkX (in-memory) | ✗ | **No longer used for retrieval. Only built during indexing as a side effect.** |

**Graph mode is still local graph-enhanced RAG, not Microsoft GraphRAG.** Community detection and community summaries are not implemented. See §2.9.

**Remaining gaps:**
- All relationships use `RELATES_TO` type with verb phrase as property — typed Cypher path queries (`MATCH (a)-[:FOUNDED]->(b)`) are not possible. See §2.4.
- No full-text index in Neo4j — entity name CONTAINS search is O(n). See §2.6.
- No vector index in Neo4j — embeddings live in Qdrant, not Neo4j nodes. HybridCypherRetriever pattern not yet used.

### 2.8 Text2Cypher Safety
**Grade: Partially industry-grade**

- `generate_cypher_from_question()` uses LiteLLM + `EXPLAIN` for syntax validation.
- Single retry loop with error context in the prompt.
- `CypherGenerationError` raised if retry also fails — good.
- **Gaps:**
  - No read-only guard: generated Cypher could include `CREATE`, `MERGE`, `DELETE`. EXPLAIN validates syntax but not safety. Should wrap execution in a read-only transaction: `with session.begin_transaction() as tx:` with no `tx.commit()`.
  - Cypher examples in the prompt use only `MATCH ... RETURN ... LIMIT` which guides the LLM toward safe queries, but it's not enforced.
  - Schema info could expose sensitive property names.

### 2.9 Is This Microsoft-Style GraphRAG?
**Answer: No.**

Microsoft GraphRAG requires:
1. **Community detection** (Leiden/Louvain algorithm over entity graph)
2. **Community summaries** (LLM-generated summaries at each community level)
3. **Hierarchical community structure** (multiple levels of abstraction)
4. **Global query mode** (answer from community summaries, no vector search)
5. **Local query mode** (combine community summary + entity neighborhood + source chunks)

This repo has:
- Entity extraction ✓ (weaker than MS GraphRAG's full LLM extraction)
- Relationship extraction ✓ (generic, not typed)
- Graph traversal ✓ (Qdrant vector seeds + Neo4j Cypher traversal — see §2.7)
- Community detection ✗ — **does not exist**
- Community summaries ✗ — **does not exist**
- Hierarchical structure ✗
- Global query mode ✗

**Verdict:** This is **local graph-enhanced RAG**. The graph layer augments retrieval with entity relationships but does not provide the community-summarization layer that makes Microsoft GraphRAG strong on global sensemaking queries ("what are the main themes across this corpus?"). The benchmark suite (Suite 3) reflects this honestly: it labels graph mode as local graph-enhanced RAG and does not claim Microsoft GraphRAG parity.

### 2.10 Should We Use neo4j-graphrag Retrievers?
**Current:** Custom Qdrant (vector seeds) + Neo4j Cypher traversal combination. NetworkX removed from retrieval path.

**neo4j-graphrag Python library offers:**
- `VectorRetriever` — vector search in Neo4j (needs vector index in Neo4j)
- `VectorCypherRetriever` — vector search + automatic graph traversal via Cypher template
- `HybridRetriever` — vector + full-text in Neo4j
- `HybridCypherRetriever` — hybrid + Cypher traversal (the gold standard pattern)
- `Text2CypherRetriever` — natural language to Cypher (similar to what we have)

**Assessment:** Our custom solution is architecturally reasonable but lacks full-text index and Neo4j vector index. Switching to `HybridCypherRetriever` would require:
1. Adding vector index to Neo4j (store embeddings in Neo4j nodes)
2. Adding full-text index on entity name/description
3. Refactoring `search_graph()` to use the retriever instead of Qdrant + NetworkX

This is a significant refactor but would consolidate retrieval into Neo4j and enable true hybrid path queries.

---

## 3. Summary of Concrete Issues

| Priority | Issue | Impact | Fix |
|---|---|---|---|
| P0 | RAGAS contexts always empty — `query()` returns str | Invalid eval | Fixed: `query_debug()` + updated `ragas_harness.py` |
| P1 | ~~NetworkX graph lost on restart~~ | **Fixed** | NetworkX removed from retrieval; Neo4j `traverse_neighbors()` used instead |
| P1 | All relationships `RELATES_TO` with property | Kills typed Cypher paths | Use typed relationship labels (`FOUNDED`, `ACQUIRED`, etc.) |
| P1 | No full-text index in Neo4j | CONTAINS is O(n) linear scan | Add `FULLTEXT INDEX` on name/description/aliases |
| P2 | Single `Entity` label for all types | Can't use label-based filtering | Add typed labels (`Person`, `Organization`, etc.) |
| P2 | No vector index in Neo4j | Can't use HybridCypherRetriever | Store embeddings in Neo4j nodes + add vector index |
| P2 | No read-only guard on Cypher execution | LLM could write to graph | Wrap in read-only transaction |
| P2 | Chunking splits mid-sentence | Chunks lose semantic coherence | Switch to sentence-boundary recursive splitter |
| P3 | No community detection/summaries | Can't do global sensemaking | Implement Leiden + LLM community summaries |
| P3 | GLiNER small vs medium/large | Lower entity recall in finance | Upgrade to `gliner_medium-v2.1` or add finance entity types |
| P3 | No co-reference resolution | "the company" not resolved to entity | Post-process with spaCy corefs or LLM pass |
