#!/usr/bin/env python3
"""
SimpleRAGx — Benchmark Runner
==============================
Runs evaluation suites and produces per-mode metric tables.

Suite 1 — FinanceBench (AMD/Boeing 10-K)
    AMD/Boeing questions from eval/ground_truth.yaml.
    Metrics: LLM-judge accuracy, keyword recall, retrieved context count, latency.

Suite 2 — Multi-Hop Finance Scaffold
    IMPORTANT: This is NOT the real MultiHop-RAG benchmark.
    Real MultiHop-RAG (yixuantt/MultiHopRAG) requires indexing a separate
    news-article corpus. This scaffold runs finance multi-hop questions
    against the existing AMD/Boeing index to validate retrieval plumbing.
    See BENCHMARK_RESEARCH.md §2 for real MultiHop-RAG setup.

Suite 3 — Global Sensemaking (pairwise LLM judge)
    Broad thematic queries compared pairwise across modes.
    Mirrors Microsoft BenchmarkQED methodology (comprehensiveness, diversity,
    empowerment, relevance) but WITHOUT community summaries.
    IMPORTANT: SimpleRAGx lacks community detection. Graph mode here is local
    graph-enhanced retrieval, not Microsoft-style global GraphRAG.

Usage:
    # FinanceBench, use existing app collections (no re-indexing)
    python eval/run_benchmarks.py --suite financebench --modes normal graph --use-default-collections

    # FinanceBench, use isolated bench_* collections (re-index first)
    python eval/run_benchmarks.py --suite financebench --modes normal graph

    # MultiHop finance scaffold (no re-index)
    python eval/run_benchmarks.py --suite multihop --modes normal graph --use-default-collections --n 3

    # Global sensemaking pairwise (no re-index)
    python eval/run_benchmarks.py --suite sensemaking --modes normal graph --use-default-collections

    # All suites, isolated collections
    python eval/run_benchmarks.py --suite all --modes normal graph --n 14

Environment variables (required):
    GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY

Optional (for neo4j/hybrid_neo4j modes):
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_DIR    = Path(__file__).parent
DOCS_DIR    = EVAL_DIR / "docs"
RESULTS_DIR = EVAL_DIR / "results"
GT_PATH     = EVAL_DIR / "ground_truth.yaml"

SUPPORTED_MODES = ["normal", "graph", "neo4j", "hybrid_neo4j"]
GEMINI_JUDGE    = "gemini/gemini-2.5-flash"

# ── LLM judge ────────────────────────────────────────────────────────────────

_JUDGE_ANSWER = """\
You are evaluating whether an AI-generated answer correctly answers a question.

Question: {question}
Golden answer: {golden}
AI-generated answer: {generated}

Rules:
- Minor rounding differences are acceptable (e.g. "0.62%" vs "0.6%")
- Equivalent representations are acceptable (e.g. "$25,867M" vs "$25.867 billion")
- The AI answer is correct if it conveys the same meaning or is a correct superset
- "I don't know" / "not found" / empty = WRONG

Reply with exactly one word: TRUE or FALSE"""

_JUDGE_PAIRWISE = """\
You are evaluating two AI answers on the dimension of {dimension}.

{dim_description}

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Which is better on {dimension}? Reply with exactly: A, B, or TIE"""

_DIM_DESCRIPTIONS = {
    "comprehensiveness": "Does the answer address all relevant aspects?",
    "diversity":         "Does the answer present a variety of perspectives or entities?",
    "empowerment":       "Does the answer help the reader make informed judgments?",
    "relevance":         "Does the answer directly address what is asked, without off-topic content?",
}


def _llm_call(prompt: str, max_tokens: int = 10) -> str:
    import litellm
    os.environ.setdefault("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    resp = litellm.completion(
        model=GEMINI_JUDGE,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}},
    )
    return (resp.choices[0].message.content or "").strip()


def judge_answer(question: str, golden: str, generated: str) -> bool:
    if not generated or not generated.strip():
        return False
    try:
        v = _llm_call(_JUDGE_ANSWER.format(
            question=question, golden=golden.strip(), generated=generated.strip()
        )).lower()
        return "true" in v
    except Exception as e:
        print(f"    [judge error] {e}")
        return False


def judge_pairwise(question: str, a: str, b: str, dimension: str) -> str:
    """Returns 'A', 'B', or 'TIE'."""
    try:
        v = _llm_call(_JUDGE_PAIRWISE.format(
            dimension=dimension,
            dim_description=_DIM_DESCRIPTIONS.get(dimension, dimension),
            question=question,
            answer_a=a or "(no answer)",
            answer_b=b or "(no answer)",
        ), max_tokens=5).upper()
        if "A" in v and "B" not in v:
            return "A"
        if "B" in v and "A" not in v:
            return "B"
        return "TIE"
    except Exception as e:
        print(f"    [pairwise judge error] {e}")
        return "TIE"


# ── RAG runner ────────────────────────────────────────────────────────────────

_rag_cache: Dict[str, Any] = {}

# Set once by main() from CLI args; consumed by _get_rag and _sanity_check.
_COLLECTION_OVERRIDE: str = ""
_GRAPH_COLLECTION_OVERRIDE: str = ""
_ALLOW_EMPTY_GRAPH: bool = False


def _build_cfg(
    mode: str,
    use_default_collections: bool,
    collection_name_override: str = "",
    graph_collection_name_override: str = "",
) -> Dict:
    def req(k):
        v = os.environ.get(k)
        if not v:
            raise EnvironmentError(f"Missing required env var: {k}")
        return v

    if collection_name_override:
        collection_name = collection_name_override
    elif use_default_collections:
        collection_name = "simple_rag_docs"
    else:
        collection_name = f"bench_{mode}"

    if graph_collection_name_override:
        graph_collection_name = graph_collection_name_override
    elif use_default_collections:
        graph_collection_name = "simple_rag_graph"
    else:
        graph_collection_name = f"bench_{mode}_graph"

    return {
        "gemini_api_key":         req("GEMINI_API_KEY"),
        "qdrant_url":             req("QDRANT_URL"),
        "qdrant_api_key":         req("QDRANT_API_KEY"),
        "neo4j_uri":              os.environ.get("NEO4J_URI", ""),
        "neo4j_username":         os.environ.get("NEO4J_USER", "neo4j"),
        "neo4j_password":         os.environ.get("NEO4J_PASSWORD", ""),
        "neo4j_database":         os.environ.get("NEO4J_DATABASE", "neo4j"),
        "neo4j_enabled":          bool(os.environ.get("NEO4J_URI")),
        "collection_name":        collection_name,
        "graph_collection_name":  graph_collection_name,
        "rag_mode":               mode,
        "setup_completed":        True,
        "preferred_llm":          "raw",
        "embedding_dimension":    768,
        "chunk_size":             1000,
        "chunk_overlap":          200,
        "top_k":                  5,
        "rate_limit":             300,
        "enable_cache":           False,
        "cache_dir":              None,
        "max_entities_per_chunk": 20,
        "graph_reasoning_depth":  2,
        "entity_similarity_threshold": 0.8,
        "graph_extraction_timeout": 60,
        "max_chunk_length_for_graph": 1000,
        "enable_agentic_ai":      False,
        "enable_query_planning":  True,
        "enable_reranking":       True,
        "enable_metadata_extraction": False,
        "pageindex_enabled":      False,
        "pageindex_workspace":    "/tmp/bench_pi_ws",
        "relationship_extraction_prompt": "extract_relationships",
    }


def _get_rag(mode: str, use_default_collections: bool) -> Any:
    key = f"{mode}_{use_default_collections}_{_COLLECTION_OVERRIDE}_{_GRAPH_COLLECTION_OVERRIDE}"
    if key not in _rag_cache:
        from config import ConfigManager
        from simple_rag import SimpleRAG
        cfg = _build_cfg(
            mode, use_default_collections,
            collection_name_override=_COLLECTION_OVERRIDE,
            graph_collection_name_override=_GRAPH_COLLECTION_OVERRIDE,
        )
        cm = ConfigManager.__new__(ConfigManager)
        cm.config_path = f"/tmp/bench_cfg_{mode}.json"
        cm.force_fresh_start = False
        cm.config = cfg
        _rag_cache[key] = (SimpleRAG(cm), cfg)
    return _rag_cache[key]


def _index_docs(mode: str, pdf_paths: List[str], use_default_collections: bool):
    rag, cfg = _get_rag(mode, use_default_collections)
    print(f"  Collection: {cfg['collection_name']}")
    for pdf in pdf_paths:
        if not os.path.exists(pdf):
            print(f"  [skip] {pdf} not found — place PDF in eval/docs/")
            continue
        print(f"  Indexing {Path(pdf).name} →")
        result = rag.index_document(pdf)
        if result.get("success"):
            print(f"    chunks={result.get('chunks_indexed',0)} "
                  f"entities={result.get('entities_extracted',0)} "
                  f"rels={result.get('relationships_extracted',0)} "
                  f"time={result.get('time_elapsed',0):.1f}s")
        else:
            print(f"    ERROR: {result.get('error', 'unknown')}")


def _check_one_collection(client, name: str) -> int:
    """
    Return point count for a collection. sys.exit(1) if missing or empty.
    Never continues silently — every path either returns a positive count or exits.
    """
    try:
        info  = client.get_collection(name)
        count = info.points_count
    except Exception as e:
        print(f"\n  ERROR: Collection '{name}' not found or unreachable: {e}")
        print("  Index documents first, or pass --use-default-collections to use the app's existing index.")
        print("  Run without any index flag to build isolated bench_* collections.")
        sys.exit(1)

    print(f"  [sanity] {name} → {count} points")
    if count == 0:
        print(f"\n  ERROR: Collection '{name}' has 0 points — no documents indexed.")
        print("  Index documents first, or pass --use-default-collections to use the app's existing index.")
        sys.exit(1)
    return count


def _sanity_check(mode: str, use_default_collections: bool) -> None:
    allow_empty_graph = _ALLOW_EMPTY_GRAPH
    """
    Print collection point counts for the given mode and exit if any required
    collection is missing or empty.

    For graph / hybrid_neo4j modes the graph collection is also checked.
    Pass allow_empty_graph=True only for explicit scaffold testing where you
    know graph embeddings were never built.
    """
    rag, cfg = _get_rag(mode, use_default_collections)
    client   = rag.vector_db_service.client

    doc_name   = cfg["collection_name"]
    graph_name = cfg.get("graph_collection_name", "")

    print(f"  [sanity] mode={mode}  doc_collection={doc_name}")
    _check_one_collection(client, doc_name)

    if mode in ("graph", "hybrid_neo4j") and graph_name:
        print(f"  [sanity] mode={mode}  graph_collection={graph_name}")
        if allow_empty_graph:
            try:
                info  = client.get_collection(graph_name)
                count = info.points_count
                print(f"  [sanity] {graph_name} → {count} points (--allow-empty-graph: skipping exit)")
            except Exception:
                print(f"  [sanity] {graph_name} → not found (--allow-empty-graph: skipping exit)")
        else:
            graph_count = _check_one_collection(client, graph_name)
            if graph_count == 0:
                sys.exit(1)
            print(f"  [sanity] graph_collection={graph_name}: {graph_count} points OK")

    # Neo4j count check: required for neo4j / hybrid_neo4j; advisory for graph mode.
    if mode in ("graph", "neo4j", "hybrid_neo4j"):
        neo4j_svc = getattr(rag, "neo4j_service", None)
        if neo4j_svc is None:
            if mode in ("neo4j", "hybrid_neo4j"):
                print(f"\n  ERROR: Neo4j service not initialised for mode={mode}.")
                print("  Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in your environment.")
                sys.exit(1)
            else:
                print(f"  [sanity] Neo4j not configured — graph mode will use vector-only retrieval.")
        else:
            try:
                ns = neo4j_svc.get_graph_stats()
                n_nodes = ns.get("nodes", 0)
                n_rels  = ns.get("relationships", 0)
                print(f"  [sanity] Neo4j: {n_nodes} nodes, {n_rels} relationships")
                if n_nodes == 0 or n_rels == 0:
                    if mode in ("neo4j", "hybrid_neo4j"):
                        print(f"\n  ERROR: Neo4j graph is empty (nodes={n_nodes}, relationships={n_rels}) for mode={mode}.")
                        print("  Index documents with Neo4j enabled first.")
                        sys.exit(1)
                    else:
                        print(f"  WARNING: Neo4j graph has no traversable data (nodes={n_nodes}, relationships={n_rels}) — traversal will return no neighbors.")
            except Exception as e:
                print(f"  WARNING: Could not query Neo4j stats: {e}")


def _query(mode: str, question: str, use_default_collections: bool) -> Dict[str, Any]:
    rag, _ = _get_rag(mode, use_default_collections)
    return rag.query_debug(question)


# ── Suite 1: FinanceBench ─────────────────────────────────────────────────────

def run_financebench(
    modes: List[str],
    index: bool,
    use_default_collections: bool,
    n: Optional[int] = None,
) -> Dict:
    print("\n" + "=" * 65)
    print("  SUITE 1 — FinanceBench (AMD/Boeing 10-K)")
    print("=" * 65)

    pdfs = [str(DOCS_DIR / "amd_2022_10k.pdf"), str(DOCS_DIR / "boeing_2022_10k.pdf")]

    try:
        import yaml
    except ModuleNotFoundError:
        print("ERROR: PyYAML not installed. Run: pip install -r requirements.txt")
        sys.exit(1)
    with open(GT_PATH) as f:
        questions = yaml.safe_load(f)["questions"]
    if n:
        questions = questions[:n]

    if index:
        for mode in modes:
            print(f"\n[indexing] mode={mode}")
            _index_docs(mode, pdfs, use_default_collections)

    results_by_mode = {}
    for mode in modes:
        print(f"\n[sanity check] mode={mode}")
        _sanity_check(mode, use_default_collections)

        print(f"\n[eval] mode={mode} ({len(questions)} questions)")
        mode_results = []
        total_ctx = 0

        for qt in questions:
            qid      = qt["id"]
            question = qt["question"]
            golden   = qt["expected_answer"].strip()
            keywords = qt.get("expected_contexts", [])

            t0 = time.time()
            out = _query(mode, question, use_default_collections)
            elapsed = time.time() - t0

            answer    = out.get("answer", "")
            ctx_count = len(out.get("contexts_doc", [])) + len(out.get("contexts_graph", []))
            total_ctx += ctx_count
            passed    = judge_answer(question, golden, answer)
            kw_hits   = sum(1 for kw in keywords if kw.lower() in answer.lower())
            kw_pct    = kw_hits / len(keywords) * 100 if keywords else 0.0

            icon = "✓" if passed else "✗"
            print(f"  {icon} [{qid}] judge={passed} ctx={ctx_count} "
                  f"kw={kw_hits}/{len(keywords)} ({kw_pct:.0f}%) {elapsed:.1f}s")

            mode_results.append({
                "id":           qid,
                "question":     question,
                "answer":       answer,
                "golden":       golden,
                "passed":       passed,
                "kw_pct":      round(kw_pct, 1),
                "ctx_count":    ctx_count,
                "latency_s":    round(elapsed, 2),
                "cypher_query": out.get("cypher_query", ""),
            })

        avg_ctx = total_ctx / len(mode_results) if mode_results else 0
        if avg_ctx == 0:
            print(f"\n  WARNING: avg retrieved context count is 0 for mode={mode}.")
            print("  Answer accuracy numbers are unreliable — the system returned answers without context.")

        n_pass  = sum(1 for r in mode_results if r["passed"])
        acc     = n_pass / len(mode_results) * 100
        avg_lat = sum(r["latency_s"] for r in mode_results) / len(mode_results)
        avg_kw  = sum(r["kw_pct"]    for r in mode_results) / len(mode_results)

        print(f"\n  [{mode}] accuracy={acc:.0f}% ({n_pass}/{len(mode_results)}) "
              f"avg_kw={avg_kw:.1f}% avg_ctx={avg_ctx:.1f} avg_lat={avg_lat:.1f}s")

        results_by_mode[mode] = {
            "suite":        "financebench",
            "n_questions":  len(mode_results),
            "accuracy_pct": round(acc, 1),
            "avg_kw_recall": round(avg_kw, 1),
            "avg_ctx_count": round(avg_ctx, 1),
            "avg_latency_s": round(avg_lat, 2),
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "per_question": mode_results,
        }

    return results_by_mode


# ── Suite 2: Multi-Hop Finance Scaffold (NOT real MultiHop-RAG) ───────────────

# These are multi-hop finance questions answered against the AMD/Boeing index.
# They are NOT from the yixuantt/MultiHopRAG dataset and do NOT use its corpus.
# Label clearly: "multi-hop finance scaffold".
_SCAFFOLD_QUESTIONS = [
    {
        "question": "What was AMD's FY2022 revenue, and how did the Xilinx acquisition affect operating margins?",
        "answer": ("AMD FY2022 revenue was $23.6B (+44% YoY). Operating margin decreased "
                   "due to amortization of intangible assets from the Xilinx acquisition."),
        "evidence": ["23.6 billion", "Xilinx", "amortization", "intangible assets"],
        "hop_count": 2,
    },
    {
        "question": "Which Boeing aircraft programs are forecast to increase production in FY2023, "
                    "and what were Boeing's FY2022 gross margins?",
        "answer": ("Boeing forecasts production rate increases for the 737, 777X, and 787 in 2023. "
                   "Gross margin improved from 4.8% in FY2021 to 5.3% in FY2022."),
        "evidence": ["737", "777X", "787", "gross profit", "5.3%", "4.8%"],
        "hop_count": 2,
    },
    {
        "question": "Did AMD have a healthy liquidity profile in FY2022 based on its quick ratio, "
                    "and what products drove its highest revenue segment?",
        "answer": ("AMD's quick ratio was 1.57, indicating healthy liquidity. "
                   "The Data Center segment, driven by EPYC server processors, had the highest proportional growth."),
        "evidence": ["quick ratio", "1.57", "Data Center", "EPYC"],
        "hop_count": 2,
    },
    {
        "question": "What were Boeing's primary customer categories in FY2022 and what legal risks did it face?",
        "answer": ("Primary customers are commercial airlines and the US government (40% of revenues). "
                   "Legal risks include lawsuits from the 2018 Lion Air and 2019 Ethiopian Airlines crashes."),
        "evidence": ["commercial airlines", "US government", "40%", "Lion Air", "Ethiopian Airlines"],
        "hop_count": 2,
    },
    {
        "question": "From FY21 to FY22, which AMD segment grew most proportionally, and what customer "
                    "concentration risk did AMD report?",
        "answer": ("The Data Center segment grew most, from $3,694M to $6,043M. "
                   "One customer accounted for 16% of consolidated net revenue from the Gaming segment."),
        "evidence": ["Data Center", "6,043", "3,694", "16%", "Gaming segment"],
        "hop_count": 2,
    },
]


def run_multihop_scaffold(
    modes: List[str],
    index: bool,
    use_default_collections: bool,
    n: Optional[int] = None,
) -> Dict:
    """Multi-hop finance scaffold — NOT the real MultiHop-RAG benchmark."""
    print("\n" + "=" * 65)
    print("  SUITE 2 — Multi-Hop Finance Scaffold")
    print("  NOTE: This is NOT the real MultiHop-RAG benchmark.")
    print("  Real MultiHop-RAG requires indexing the news-article corpus.")
    print("  See BENCHMARK_RESEARCH.md §2 for real setup instructions.")
    print("=" * 65)

    samples = _SCAFFOLD_QUESTIONS[:n] if n else _SCAFFOLD_QUESTIONS

    if index:
        pdfs = [str(DOCS_DIR / "amd_2022_10k.pdf"), str(DOCS_DIR / "boeing_2022_10k.pdf")]
        for mode in modes:
            print(f"\n[indexing] mode={mode}")
            _index_docs(mode, pdfs, use_default_collections)

    results_by_mode = {}
    for mode in modes:
        print(f"\n[sanity check] mode={mode}")
        _sanity_check(mode, use_default_collections)

        print(f"\n[eval] mode={mode} ({len(samples)} scaffold questions)")
        mode_results = []
        total_ctx = 0

        for i, sample in enumerate(samples):
            question  = sample["question"]
            golden    = sample["answer"]
            evidence  = sample.get("evidence", [])
            hop_count = sample.get("hop_count", 0)

            t0 = time.time()
            out = _query(mode, question, use_default_collections)
            elapsed = time.time() - t0

            answer    = out.get("answer", "")
            ctx_count = len(out.get("contexts_doc", [])) + len(out.get("contexts_graph", []))
            total_ctx += ctx_count
            passed    = judge_answer(question, golden, answer)

            all_ctx_text = " ".join(
                c.get("text", "") for c in out.get("contexts_doc", []) + out.get("contexts_graph", [])
            ).lower()
            ev_hits   = sum(1 for ev in evidence if ev.lower() in all_ctx_text)
            ev_rate   = ev_hits / len(evidence) * 100 if evidence else 0.0

            icon = "✓" if passed else "✗"
            print(f"  {icon} [Q{i+1}] hops={hop_count} judge={passed} "
                  f"ctx={ctx_count} ev_hit={ev_rate:.0f}% {elapsed:.1f}s")

            mode_results.append({
                "question":    question[:80],
                "passed":      passed,
                "hop_count":   hop_count,
                "ev_hit_rate": round(ev_rate, 1),
                "ctx_count":   ctx_count,
                "latency_s":   round(elapsed, 2),
            })

        avg_ctx = total_ctx / len(mode_results) if mode_results else 0
        if avg_ctx == 0:
            print(f"\n  WARNING: avg retrieved context count is 0 for mode={mode}.")

        n_pass  = sum(1 for r in mode_results if r["passed"])
        acc     = n_pass / len(mode_results) * 100
        avg_ev  = sum(r["ev_hit_rate"] for r in mode_results) / len(mode_results)
        avg_lat = sum(r["latency_s"]   for r in mode_results) / len(mode_results)

        print(f"\n  [{mode}] accuracy={acc:.0f}% ({n_pass}/{len(mode_results)}) "
              f"avg_ev_hit={avg_ev:.1f}% avg_ctx={avg_ctx:.1f} avg_lat={avg_lat:.1f}s")

        results_by_mode[mode] = {
            "suite":            "multihop_finance_scaffold",
            "scaffold_warning": (
                "NOT real MultiHop-RAG. Uses AMD/Boeing docs with hand-written "
                "multi-hop finance questions. See BENCHMARK_RESEARCH.md for real setup."
            ),
            "n_questions":  len(mode_results),
            "accuracy_pct": round(acc, 1),
            "avg_ev_hit_rate": round(avg_ev, 1),
            "avg_ctx_count": round(avg_ctx, 1),
            "avg_latency_s": round(avg_lat, 2),
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "per_question": mode_results,
        }

    return results_by_mode


# ── Suite 3: Global Sensemaking ───────────────────────────────────────────────

_SENSEMAKING_QUESTIONS = [
    "What are the main strategic themes and challenges facing these companies?",
    "How do these companies approach risk management and what are their most significant risks?",
    "What are the key differences in how these companies drive revenue growth?",
    "What major acquisitions or partnerships have shaped these companies' strategies?",
    "How do these companies describe their competitive positioning and moats?",
    "What are the primary workforce and talent-related challenges mentioned across these companies?",
    "How do these companies address regulatory and legal risks in their filings?",
    "What are the common themes in how these companies discuss their technology and R&D investments?",
]

_SENSEMAKING_DIMS = ["comprehensiveness", "diversity", "empowerment", "relevance"]


def run_sensemaking(
    modes: List[str],
    index: bool,
    use_default_collections: bool,
) -> Dict:
    print("\n" + "=" * 65)
    print("  SUITE 3 — Global Sensemaking (Pairwise LLM Judge)")
    print("  NOTE: SimpleRAGx has no community detection/summaries.")
    print("  Graph mode here is local graph-enhanced retrieval, NOT")
    print("  Microsoft-style global GraphRAG. Expect marginal or no")
    print("  improvement over normal RAG on global/thematic queries.")
    print("=" * 65)

    if len(modes) < 2:
        print("  [SKIP] Need ≥2 modes for pairwise. Pass --modes normal graph")
        return {}

    mode_a, mode_b = modes[0], modes[1]
    print(f"\n  Comparing: {mode_a} (A) vs {mode_b} (B)")

    if index:
        pdfs = [str(DOCS_DIR / "amd_2022_10k.pdf"), str(DOCS_DIR / "boeing_2022_10k.pdf")]
        for mode in [mode_a, mode_b]:
            print(f"\n[indexing] mode={mode}")
            _index_docs(mode, pdfs, use_default_collections)

    print(f"\n[sanity check]")
    _sanity_check(mode_a, use_default_collections)
    _sanity_check(mode_b, use_default_collections)

    pairwise_results = []
    for qi, question in enumerate(_SENSEMAKING_QUESTIONS):
        print(f"\n  Q{qi+1}: {question[:65]} …")

        out_a = _query(mode_a, question, use_default_collections)
        out_b = _query(mode_b, question, use_default_collections)
        ans_a = out_a.get("answer", "")
        ans_b = out_b.get("answer", "")

        ctx_a = len(out_a.get("contexts_doc", [])) + len(out_a.get("contexts_graph", []))
        ctx_b = len(out_b.get("contexts_doc", [])) + len(out_b.get("contexts_graph", []))
        print(f"    ctx: A={ctx_a} B={ctx_b}")

        dim_results = {}
        for dim in _SENSEMAKING_DIMS:
            # Counterbalance order every other question to reduce position bias
            if qi % 2 == 0:
                verdict = judge_pairwise(question, ans_a, ans_b, dim)
                winner  = verdict
            else:
                verdict = judge_pairwise(question, ans_b, ans_a, dim)
                winner  = {"A": "B", "B": "A", "TIE": "TIE"}[verdict]
            dim_results[dim] = winner
            print(f"    {dim:22s} → {winner}")

        pairwise_results.append({
            "question":   question,
            "ctx_a":      ctx_a,
            "ctx_b":      ctx_b,
            "dim_results": dim_results,
        })

    # Aggregate
    agg: Dict[str, Dict[str, int]] = {d: {"A": 0, "B": 0, "TIE": 0} for d in _SENSEMAKING_DIMS}
    for r in pairwise_results:
        for dim, winner in r["dim_results"].items():
            agg[dim][winner] += 1

    n_q = len(pairwise_results)
    print(f"\n  PAIRWISE SUMMARY ({mode_a}=A  {mode_b}=B):")
    print(f"  {'Dimension':22s} {'A wins':8s} {'B wins':8s} {'Tie':6s} {'A%':6s}")
    print("  " + "-" * 48)
    for dim in _SENSEMAKING_DIMS:
        aw, bw, tw = agg[dim]["A"], agg[dim]["B"], agg[dim]["TIE"]
        print(f"  {dim:22s} {aw:8d} {bw:8d} {tw:6d} {aw/n_q*100:5.0f}%")

    return {
        "suite":      "sensemaking",
        "mode_a":     mode_a,
        "mode_b":     mode_b,
        "note": (
            "No community summaries — graph mode is local graph-enhanced retrieval, "
            "not Microsoft GraphRAG global query mode."
        ),
        "n_questions": n_q,
        "aggregated":  agg,
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_question": pairwise_results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def _validate_env():
    missing = [k for k in ("GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")
               if not os.environ.get(k)]
    if missing:
        print(f"ERROR: Missing required env vars: {missing}")
        print("Load them with:  export $(grep -v '^#' .env | xargs)")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="SimpleRAGx Benchmark Runner")
    parser.add_argument(
        "--suite", default="financebench",
        choices=["financebench", "multihop", "sensemaking", "all"],
    )
    parser.add_argument(
        "--modes", nargs="+", default=["normal"], choices=SUPPORTED_MODES,
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Max questions per suite (default: all)",
    )
    parser.add_argument(
        "--use-default-collections", action="store_true",
        help=(
            "Use the app's existing Qdrant collections (simple_rag_docs / simple_rag_graph) "
            "instead of creating isolated bench_* collections. "
            "Skips indexing. Fails loudly if those collections are empty."
        ),
    )
    parser.add_argument(
        "--out", default=None,
        help="Output JSON path (default: eval/results/<suite>_<timestamp>.json)",
    )
    parser.add_argument(
        "--collection-name", default="",
        help="Override doc collection name (e.g. simple_rag_docs or a custom bench_ name)",
    )
    parser.add_argument(
        "--graph-collection-name", default="",
        help="Override graph collection name (e.g. simple_rag_graph or a custom bench_ name)",
    )
    parser.add_argument(
        "--allow-empty-graph", action="store_true",
        help="Skip sys.exit if graph collection exists but is empty (e.g. graph embeddings intentionally not built yet)",
    )
    args = parser.parse_args()

    global _COLLECTION_OVERRIDE, _GRAPH_COLLECTION_OVERRIDE, _ALLOW_EMPTY_GRAPH
    _COLLECTION_OVERRIDE       = args.collection_name
    _GRAPH_COLLECTION_OVERRIDE = args.graph_collection_name
    _ALLOW_EMPTY_GRAPH         = args.allow_empty_graph

    _validate_env()
    RESULTS_DIR.mkdir(exist_ok=True)

    # With --use-default-collections, never (re-)index — just use what's there.
    do_index = not args.use_default_collections
    udc      = args.use_default_collections

    all_results: Dict[str, Any] = {}

    if args.suite in ("financebench", "all"):
        r = run_financebench(args.modes, do_index, udc, n=args.n)
        all_results["financebench"] = r

    if args.suite in ("multihop", "all"):
        r = run_multihop_scaffold(args.modes, do_index, udc, n=args.n)
        all_results["multihop_finance_scaffold"] = r

    if args.suite in ("sensemaking", "all"):
        r = run_sensemaking(args.modes, do_index, udc)
        all_results["sensemaking"] = r

    ts       = time.strftime("%Y%m%dT%H%M%S")
    out_path = args.out or str(RESULTS_DIR / f"{args.suite}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved → {out_path}")

    # Per-mode graph diagnostics (Neo4j only — NetworkX removed from retrieval path)
    for mode in args.modes:
        key = f"{mode}_{udc}_{_COLLECTION_OVERRIDE}_{_GRAPH_COLLECTION_OVERRIDE}"
        if key in _rag_cache:
            rag, _ = _rag_cache[key]
            if getattr(rag, "neo4j_service", None):
                try:
                    ns = rag.neo4j_service.get_graph_stats()
                    print(f"\n[{mode}] Neo4j: nodes={ns['nodes']} relationships={ns['relationships']}")
                    if ns["nodes"] == 0:
                        print(f"  WARNING: Neo4j graph is empty — no traversal results will be returned.")
                except Exception as e:
                    print(f"\n[{mode}] Neo4j stats unavailable: {e}")
            elif mode in ("graph", "hybrid_neo4j", "neo4j"):
                print(f"\n[{mode}] Neo4j not configured — graph traversal disabled.")


if __name__ == "__main__":
    main()
