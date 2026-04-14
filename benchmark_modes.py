#!/usr/bin/env python3
"""
SimpleRAGx — RAG Mode Benchmark
=================================
Runs the TechCorp test document through Normal RAG, Graph RAG, and PageIndex,
asks the same 5 questions to each, measures timing, and prints a comparative report.

Usage:
    python benchmark_modes.py
"""

import os, sys, time, json, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GEMINI_KEY = "REDACTED_ROTATE_THIS_KEY"
QDRANT_URL = "https://42101fde-47fd-4914-9f7b-ab368688ea6a.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_KEY = "REDACTED_QDRANT_KEY"
NEO4J_URI  = "neo4j+s://4521331b.databases.neo4j.io"
NEO4J_USER = "4521331b"
NEO4J_PASS = "REDACTED_NEO4J_PASS"
NEO4J_DB   = "4521331b"

os.environ["GEMINI_API_KEY"] = GEMINI_KEY

TEST_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "techcorp_q1_2025_report.pdf")
PI_WORKSPACE = "/tmp/pi_bench_ws"

QUESTIONS = [
    "Who are the founders of TechCorp and what are their backgrounds?",
    "What is TechCorp's Q1 2025 revenue breakdown by product and key financial metrics?",
    "What are TechCorp's key strategic partnerships and their financial terms?",
    "What are the main risk factors facing TechCorp?",
    "What is TechCorp's IPO timeline, target valuation, and Series D details?",
]

CFG = {
    "gemini_api_key":        GEMINI_KEY,
    "claude_api_key":        "",
    "openai_api_key":        "",
    "qdrant_url":            QDRANT_URL,
    "qdrant_api_key":        QDRANT_KEY,
    "neo4j_uri":             NEO4J_URI,
    "neo4j_username":        NEO4J_USER,
    "neo4j_password":        NEO4J_PASS,
    "neo4j_database":        NEO4J_DB,
    "neo4j_enabled":         True,
    "collection_name":       "bench_techcorp",
    "graph_collection_name": "bench_techcorp_graph",
    "rag_mode":              "normal",
    "setup_completed":       True,
    "preferred_llm":                  "raw",
    "embedding_dimension":            768,
    "chunk_size":                     1000,
    "chunk_overlap":                  200,
    "top_k":                          5,
    "rate_limit":                     60,
    "enable_cache":                   False,
    "cache_dir":                      None,
    "max_entities_per_chunk":         20,
    "relationship_extraction_prompt": "extract_relationships",
    "graph_reasoning_depth":          2,
    "entity_similarity_threshold":    0.8,
    "graph_extraction_timeout":       60,
    "max_chunk_length_for_graph":     1000,
    "enable_agentic_ai":              False,
    "pageindex_workspace":            PI_WORKSPACE,
}

GEMINI_MODEL = "gemini/gemini-2.5-flash"


def llm_answer(context: str, question: str) -> str:
    """Call Gemini 2.5 Flash directly via LiteLLM for answer synthesis."""
    import litellm
    prompt = (
        f"Answer the following question based ONLY on the provided context. "
        f"Be concise and precise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    resp = litellm.completion(
        model=GEMINI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


# ── Helpers ──────────────────────────────────────────────────────────────────

def hr(c="─", w=72): print(c * w)
def section(t): hr(); print(f"  {t}"); hr()
def wrap(text, width=68, indent="    "):
    if not text: return indent + "(no answer)"
    lines = []
    for line in str(text).split("\n"):
        if line.strip():
            lines.extend(textwrap.wrap(line, width, initial_indent=indent, subsequent_indent=indent))
        else:
            lines.append("")
    return "\n".join(lines)


# ── Build the test PDF if missing ────────────────────────────────────────────

def ensure_pdf():
    if os.path.exists(TEST_PDF):
        print(f"  PDF ready: {TEST_PDF}  ({os.path.getsize(TEST_PDF):,} bytes)")
    else:
        raise FileNotFoundError(f"Test PDF not found: {TEST_PDF}\nRun the PDF generator first.")


# ── Normal RAG ───────────────────────────────────────────────────────────────

def run_normal_rag():
    section("MODE 1 — Normal RAG  (embed → Qdrant → LLM)")
    from embedding_service import EmbeddingService
    from vector_db_service import VectorDBService
    from document_processor import DocumentProcessor

    emb = EmbeddingService(CFG)
    vdb = VectorDBService(CFG)
    dp  = DocumentProcessor(CFG)

    # Index: process_document returns chunks with key "text" and "metadata"
    print("  Indexing document...")
    t0 = time.time()
    chunks = dp.process_document(TEST_PDF)
    texts      = [c["text"] for c in chunks]
    embeddings = emb.get_embeddings_batch(texts)
    vdb.insert_documents(chunks, embeddings)
    index_time = round(time.time() - t0, 2)
    print(f"  Indexed {len(chunks)} chunks in {index_time}s")

    results = []
    for q in QUESTIONS:
        t0 = time.time()
        q_emb = emb.get_embedding(q)
        hits  = vdb.search_similar(q_emb, top_k=3) if q_emb else []
        context = "\n\n".join(h.get("text", h.get("content", "")) for h in hits)
        answer  = llm_answer(context, q)
        elapsed = round(time.time() - t0, 2)
        results.append({"q": q, "answer": answer, "time": elapsed,
                         "chunks_retrieved": len(hits)})
        print(f"  Q: {q[:60]}...")
        print(f"  A: {str(answer)[:200]}")
        print(f"  Time: {elapsed}s  |  chunks: {len(hits)}")
        print()

    return {"mode": "Normal RAG", "index_time": index_time, "results": results}


# ── Graph RAG — Vector Store variant ─────────────────────────────────────────

def run_graph_rag_vector():
    """
    Graph RAG (Vector): extract entities/relationships → embed each as text →
    store in Qdrant graph collection → retrieve via vector similarity + NetworkX
    multi-hop traversal → LLM synthesises.
    """
    section("MODE 2a — Graph RAG (Vector)  (entities → Qdrant graph collection → LLM)")
    from graph_rag_service import GraphRAGService
    from embedding_service import EmbeddingService
    from vector_db_service import VectorDBService
    from document_processor import DocumentProcessor

    emb = EmbeddingService(CFG)
    vdb = VectorDBService(CFG)
    svc = GraphRAGService(CFG)
    svc.set_services(emb, vdb)          # ← this was missing before
    dp  = DocumentProcessor(CFG)

    print("  Extracting entities + relationships, embedding, storing in Qdrant...")
    t0 = time.time()
    chunks = dp.process_document(TEST_PDF)
    graph_result = svc.process_document_for_graph(chunks)
    entity_count = len(graph_result.get("entities", []))
    rel_count    = len(graph_result.get("relationships", []))
    nodes        = graph_result["graph_stats"]["nodes"]
    edges        = graph_result["graph_stats"]["edges"]
    index_time   = round(time.time() - t0, 2)
    print(f"  Done in {index_time}s  |  {entity_count} entities  |  {rel_count} rels  "
          f"|  graph: {nodes} nodes / {edges} edges")

    results = []
    for q in QUESTIONS:
        t0 = time.time()
        hits    = svc.search_graph(q, top_k=5)
        # hits is a list of dicts with entity/relationship info
        context = json.dumps(hits, indent=2) if hits else "(no graph hits)"
        answer  = llm_answer(context, q)
        elapsed = round(time.time() - t0, 2)
        results.append({"q": q, "answer": answer, "time": elapsed,
                         "entities_retrieved": len(hits)})
        print(f"  Q: {q[:60]}...")
        print(f"  A: {str(answer)[:200]}")
        print(f"  Time: {elapsed}s  |  entities/rels retrieved: {len(hits)}")
        print()

    return {"mode": "Graph RAG (Vector)", "index_time": index_time, "results": results}


# ── Graph RAG — Neo4j variant ─────────────────────────────────────────────────

def run_graph_rag_neo4j():
    """
    Graph RAG (Neo4j): extract entities/relationships → store in Neo4j Aura →
    LLM generates Cypher from question → execute Cypher → LLM synthesises answer.
    """
    section("MODE 2b — Graph RAG (Neo4j)  (entities → Neo4j Aura → Cypher → LLM)")
    from graph_rag_service import GraphRAGService
    from neo4j_service import Neo4jService
    from document_processor import DocumentProcessor

    neo4j = Neo4jService(
        uri=NEO4J_URI, username=NEO4J_USER,
        password=NEO4J_PASS, database=NEO4J_DB,
    )
    svc = GraphRAGService(CFG)
    dp  = DocumentProcessor(CFG)

    print("  Extracting entities + relationships...")
    t0 = time.time()
    chunks = dp.process_document(TEST_PDF)
    graph_result = svc.process_document_for_graph(chunks)
    entities = graph_result.get("entities", [])
    rels     = graph_result.get("relationships", [])
    print(f"  Extracted {len(entities)} entities, {len(rels)} relationships")

    print("  Storing in Neo4j Aura...")
    stats = neo4j.store_entities_and_relationships(entities, rels, document_name="test_techcorp")
    index_time = round(time.time() - t0, 2)
    print(f"  Stored in {index_time}s  |  {stats}")

    schema = neo4j.get_schema_info()
    print(f"  Schema: {schema[:200]}...")

    results = []
    for q in QUESTIONS:
        t0 = time.time()
        cypher, cypher_err = neo4j.generate_cypher_from_question(q)
        if cypher_err:
            print(f"  [!] Cypher generation failed: {cypher_err}")
            context = f"Cypher generation error: {cypher_err}"
            rows = []
        else:
            print(f"  Cypher (LLM-generated): {cypher[:120]}")
            try:
                rows, err = neo4j.execute_cypher_query(cypher)
                if err:
                    context = f"Cypher error: {err}"
                else:
                    context = json.dumps(rows, indent=2) if rows else "(no matching entities found in graph)"
            except Exception as e:
                context = f"(error: {e})"
                rows = []
        answer  = llm_answer(context, q)
        elapsed = round(time.time() - t0, 2)
        results.append({"q": q, "answer": answer, "time": elapsed, "cypher": cypher})
        print(f"  Q: {q[:60]}...")
        print(f"  A: {str(answer)[:200]}")
        print(f"  Time: {elapsed}s  |  rows: {len(rows) if isinstance(rows, list) else 0}")
        print()

    neo4j.close()
    return {"mode": "Graph RAG (Neo4j)", "index_time": index_time, "results": results}


# ── PageIndex ────────────────────────────────────────────────────────────────

def run_pageindex():
    section("MODE 3 — PageIndex  (tree index → agent → LLM)")
    import shutil
    if os.path.exists(PI_WORKSPACE):
        shutil.rmtree(PI_WORKSPACE)

    from pageindex_service import PageIndexService
    svc = PageIndexService({**CFG, "pageindex_workspace": PI_WORKSPACE})

    print("  Building tree index...")
    t0 = time.time()
    idx = svc.index_document(TEST_PDF)
    index_time = round(time.time() - t0, 2)
    doc_id = idx.get("doc_id")
    section_count = idx.get("section_count", 0)
    print(f"  Tree built in {index_time}s  |  {section_count} sections  |  doc_id: {doc_id}")

    results = []
    for q in QUESTIONS:
        t0 = time.time()
        r  = svc.query(q, doc_id=doc_id)
        elapsed = round(time.time() - t0, 2)
        answer  = r.get("answer", "")
        citations = r.get("citations", [])
        tree_path = r.get("tree_path", [])
        tool_calls = [t["tool"] for t in r.get("tool_calls", [])]
        results.append({"q": q, "answer": answer, "time": elapsed,
                         "citations": citations, "tree_path": tree_path,
                         "tool_calls": tool_calls})
        print(f"  Q: {q[:60]}...")
        print(f"  A: {str(answer)[:200]}")
        print(f"  Time: {elapsed}s  |  pages cited: {[c['pages'] for c in citations]}  |  tools: {tool_calls}")
        print()

    return {"mode": "PageIndex", "index_time": index_time, "results": results}


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(all_results):
    section("COMPARATIVE REPORT — SimpleRAGx RAG Mode Benchmark")

    modes = [r["mode"] for r in all_results]

    print()
    print("  DOCUMENT: TechCorp Road to IPO — Q1 2025 Report (5 pages)")
    print("  Extractor: GLiNER small-v2.1 + gemini-2.5-flash-lite  |  Synthesis: Gemini 2.5 Flash")
    print(f"  Modes tested: {', '.join(modes)}")
    print()

    # ── Index time table ──────────────────────────────────────────────────────
    print("  ┌─────────────────────────────────────────┐")
    print("  │         INDEX / SETUP TIME              │")
    print("  ├──────────────┬──────────────────────────┤")
    print("  │ Mode         │ Time (s)                 │")
    print("  ├──────────────┼──────────────────────────┤")
    for r in all_results:
        print(f"  │ {r['mode']:<12} │ {r['index_time']:<26} │")
    print("  └──────────────┴──────────────────────────┘")
    print()

    # ── Per-question answer comparison ───────────────────────────────────────
    for qi, q in enumerate(QUESTIONS):
        print(f"  ── Q{qi+1}: {q}")
        print()
        for r in all_results:
            res = r["results"][qi]
            ans = str(res.get("answer") or "(no answer)")
            print(f"  [{r['mode']}]  ({res['time']}s)")
            print(wrap(ans[:400]))
            extra = []
            if res.get("chunks_retrieved"):
                extra.append(f"chunks={res['chunks_retrieved']}")
            if res.get("entities_retrieved"):
                extra.append(f"entities={res['entities_retrieved']}")
            if res.get("citations"):
                extra.append(f"pages={[c['pages'] for c in res['citations']]}")
            if res.get("tool_calls"):
                extra.append(f"tools={res['tool_calls']}")
            if extra:
                print(f"    Metadata: {', '.join(extra)}")
            print()
        hr("·")
        print()

    # ── Avg query time ───────────────────────────────────────────────────────
    print("  AVERAGE QUERY TIME (across 5 questions):")
    for r in all_results:
        times = [x["time"] for x in r["results"]]
        avg = round(sum(times) / len(times), 2)
        mn  = round(min(times), 2)
        mx  = round(max(times), 2)
        print(f"    {r['mode']:<14} avg={avg}s  min={mn}s  max={mx}s")
    print()

    # ── Pros / Cons ───────────────────────────────────────────────────────────
    section("POST-MORTEM: ADVANTAGES & DISADVANTAGES")
    analysis = [
        {
            "mode": "Normal RAG",
            "how":  "Chunks text → embeds chunks → stores in Qdrant vector DB → retrieves top-k by cosine similarity → LLM synthesises.",
            "pros": [
                "Fastest to index (seconds)",
                "Low infrastructure — just a vector DB",
                "Works on any document type (PDF, DOCX, TXT, HTML)",
                "Great for keyword/semantic queries on long docs",
                "Cheap: embedding model is the only LLM cost",
            ],
            "cons": [
                "Loses relational context — can't reason about WHO did WHAT with WHOM",
                "Chunk boundaries can split facts across chunks → incomplete answers",
                "Embedding quality varies by domain (financial jargon, legal terms)",
                "No traceable source beyond 'chunk number'",
                "Sensitive to chunk size / overlap tuning",
            ],
            "best_for": "General-purpose Q&A, large corpora, quick lookup queries.",
        },
        {
            "mode": "Graph RAG (Vector)",
            "how":  "Extracts entities + relationships via LLM → embeds each as rich text → stores in Qdrant graph collection → retrieves via vector similarity → NetworkX multi-hop traversal → LLM synthesises.",
            "pros": [
                "No graph DB needed — only Qdrant",
                "Multi-hop traversal via NetworkX reveals indirect connections",
                "Vector search finds semantically similar entities even with different wording",
                "Good for 'who worked with whom' and 'what is related to X' questions",
                "Entity merging deduplicates aliases (e.g. 'TechCorp' = 'Tech Corp')",
            ],
            "cons": [
                "Slow to index — entity extraction + embedding = many LLM calls",
                "NetworkX graph is in-memory only, lost on restart",
                "Struggles with numerical facts (numbers aren't entities)",
                "Graph quality depends on extraction LLM quality",
                "Answer traceability limited to entity/rel snippets, not original pages",
            ],
            "best_for": "Relationship questions across many docs without a dedicated graph DB.",
        },
        {
            "mode": "Graph RAG (Neo4j)",
            "how":  "Extracts entities + relationships via LLM → stores permanently in Neo4j Aura → LLM generates Cypher query from question → executes Cypher → LLM synthesises from structured rows.",
            "pros": [
                "True persistent knowledge graph — survives restarts, queryable any time",
                "Cypher queries are precise — exact entity/relationship matches",
                "Multi-hop traversal built into the query language natively",
                "Best for 'who leads what', 'what partnered with whom', org-structure questions",
                "Graph grows richer as more documents are indexed",
            ],
            "cons": [
                "Slowest overall — entity extraction + Neo4j write + Cypher generation",
                "Requires Neo4j instance (cloud or self-hosted)",
                "Cypher generation can fail or be imprecise for complex questions",
                "Numerical/financial queries need special handling (properties, not nodes)",
                "Schema must be consistent across documents for good results",
            ],
            "best_for": "Org charts, legal contracts, multi-document relationship networks, persistent knowledge bases.",
        },
        {
            "mode": "PageIndex",
            "how":  "LLM reads full doc → builds hierarchical tree (like a smart ToC with page ranges + summaries) → saved as JSON → agent navigates tree, fetches tight page ranges, synthesises.",
            "pros": [
                "No vector DB required — pure JSON workspace",
                "Exact page-level citations — fully traceable",
                "Best for structured long docs: annual reports, SEC filings, manuals",
                "Agent reasons about WHICH section to look at (not just similarity)",
                "98.7% on FinanceBench benchmark — best for financial/legal docs",
                "Handles numerical + factual questions equally well",
            ],
            "cons": [
                "Slowest to index (LLM reads entire doc to build tree — O(pages) LLM calls)",
                "Only supports PDF currently (not DOCX/TXT)",
                "Agentic loop adds latency per query (2-4 LLM calls per question)",
                "Small/unstructured docs (emails, notes) don't benefit — tree is flat",
                "Depends on LLM quality for tree construction — bad model = bad index",
            ],
            "best_for": "Professional structured docs: annual reports, SEC filings, research papers, legal manuals, technical specs.",
        },
    ]

    for a in analysis:
        if a["mode"] not in modes:
            continue
        print(f"\n  ═══ {a['mode']} ═══")
        print(f"  How it works: {a['how']}")
        print()
        print("  Advantages:")
        for p in a["pros"]:
            print(f"    + {p}")
        print()
        print("  Disadvantages:")
        for c in a["cons"]:
            print(f"    - {c}")
        print()
        print(f"  Best for: {a['best_for']}")

    print()
    section("RECOMMENDATION MATRIX")
    print()
    print("  Use Case                                  → Best Mode")
    print("  ───────────────────────────────────────────────────────────────")
    print("  Quick Q&A on any document pile            → Normal RAG")
    print("  'Who worked with whom?' (no graph DB)     → Graph RAG (Vector)")
    print("  'Who worked with whom?' (persistent)      → Graph RAG (Neo4j)")
    print("  Annual report / SEC filing analysis       → PageIndex")
    print("  Financial/numerical factual questions     → PageIndex")
    print("  Org charts, multi-doc entity network      → Graph RAG (Neo4j)")
    print("  Any doc, unknown structure, quick setup   → Normal RAG (safe default)")
    print("  Needs exact page citations + traceability → PageIndex")
    print()
    hr()
    print("  SimpleRAGx Benchmark complete.")
    hr()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    section("SimpleRAGx — RAG Mode Comparative Benchmark")
    print(f"  Document  : TechCorp Road to IPO — Q1 2025 Report (5 pages, rich entities)")
    print(f"  Questions : {len(QUESTIONS)}")
    print(f"  LLM       : gemini/gemini-2.5-flash (synthesis) + gemini-2.5-flash-lite (Cypher/rels)")
    print(f"  Extractor : GLiNER small-v2.1 (entities) + gemini-2.5-flash-lite (relationships)")
    print(f"  Vector DB : Qdrant cloud")
    print(f"  Graph DB  : Neo4j Aura")
    print()

    ensure_pdf()

    all_results = []
    failed = []

    # ── Normal RAG ────────────────────────────────────────────────────────────
    try:
        all_results.append(run_normal_rag())
    except Exception as e:
        print(f"  [!] Normal RAG failed: {e}")
        import traceback; traceback.print_exc()
        failed.append(("Normal RAG", str(e)))

    # ── Graph RAG (Vector) ────────────────────────────────────────────────────
    try:
        all_results.append(run_graph_rag_vector())
    except Exception as e:
        print(f"  [!] Graph RAG (Vector) failed: {e}")
        import traceback; traceback.print_exc()
        failed.append(("Graph RAG (Vector)", str(e)))

    # ── Graph RAG (Neo4j) ─────────────────────────────────────────────────────
    try:
        all_results.append(run_graph_rag_neo4j())
    except Exception as e:
        print(f"  [!] Graph RAG (Neo4j) failed: {e}")
        import traceback; traceback.print_exc()
        failed.append(("Graph RAG (Neo4j)", str(e)))

    # ── PageIndex ─────────────────────────────────────────────────────────────
    try:
        all_results.append(run_pageindex())
    except Exception as e:
        print(f"  [!] PageIndex failed: {e}")
        import traceback; traceback.print_exc()
        failed.append(("PageIndex", str(e)))

    if failed:
        print()
        print("  FAILED MODES:")
        for mode, err in failed:
            print(f"    {mode}: {err[:120]}")

    if all_results:
        print_report(all_results)


if __name__ == "__main__":
    main()
