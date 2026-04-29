#!/usr/bin/env python3
"""
Real MultiHop-RAG Benchmark for SimpleRAGx
============================================
Dataset: yixuantt/MultiHopRAG (COLM 2024)
  corpus   — ~609 news-article documents
  MultiHopRAG — ~2,556 QA pairs requiring 2-4 document hops

This harness:
  1. Loads real corpus rows from HuggingFace (no hand-written scaffolds)
  2. Indexes them into isolated Qdrant collections (multihop_docs, multihop_graph)
  3. For graph mode, stores entities/relationships in Neo4j with benchmark_corpus tag
  4. Evaluates Normal RAG vs Graph RAG on evidence recall and answer accuracy

Benchmark context
-----------------
- This is the best current benchmark for our Neo4j Graph RAG (multi-hop retrieval).
- FinanceBench remains the best benchmark for Normal RAG on 10-K financial QA.
- Microsoft BenchmarkQED is NOT applicable here: SimpleRAGx has no community summaries.
- Full 609-corpus run requires: python eval/multihop_rag_real.py --n 50 --corpus-mode full

Usage (mini real smoke — index evidence docs + 25 distractors, 5 questions):
    python eval/multihop_rag_real.py --n 5 --corpus-mode mini \\
        --modes normal graph \\
        --out eval/results/multihop_real_smoke.json

Environment variables required:
    GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY

Optional (graph mode Neo4j traversal):
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
GEMINI_JUDGE = "gemini/gemini-2.5-flash"
BENCHMARK_CORPUS_TAG = "multihop_rag"
# Separate doc collections per mode — prevents same docs being inserted twice when both
# modes run in the same benchmark session.
DOC_COLLECTION_NORMAL  = "multihop_normal_docs"
DOC_COLLECTION_GRAPH   = "multihop_graph_docs"
GRAPH_COLLECTION       = "multihop_graph"

def _doc_collection_for(mode: str) -> str:
    return DOC_COLLECTION_NORMAL if mode == "normal" else DOC_COLLECTION_GRAPH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multihop_bench")

# ── dataset schema inspection ─────────────────────────────────────────────────

def _inspect_dataset(ds, name: str, sample_idx: int = 0):
    print(f"\n[schema] {name} columns: {ds.column_names}")
    row = ds[sample_idx]
    for k, v in row.items():
        display = str(v)[:120].replace("\n", " ")
        print(f"  {k:25s}: {display}")


# ── LLM judge ────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are evaluating whether an AI-generated answer correctly answers a question.

Question: {question}
Golden answer: {golden}
AI-generated answer: {generated}

Rules:
- Minor rounding or phrasing differences are acceptable.
- The AI answer is correct if it conveys the same meaning or is a correct superset.
- "I don't know" / "not found" / empty = WRONG

Reply with exactly one word: TRUE or FALSE"""


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
        v = _llm_call(_JUDGE_PROMPT.format(
            question=question,
            golden=golden.strip(),
            generated=generated.strip(),
        )).lower()
        return "true" in v
    except Exception as e:
        logger.warning("Judge error: %s", e)
        return False


# ── SimpleRAG config builder ──────────────────────────────────────────────────

def _load_dev_config() -> Dict:
    """Load dev_config.json from the project root (parent of eval/)."""
    dev_cfg_path = _ROOT / "dev_config.json"
    if dev_cfg_path.exists():
        with open(dev_cfg_path) as f:
            return json.load(f)
    return {}


_DEV_CFG: Dict = {}  # loaded once on first call


def _cfg_get(key_env: str, key_json: str, default: str = "") -> str:
    """Return value from env var first, then dev_config.json, then default."""
    global _DEV_CFG
    if not _DEV_CFG:
        _DEV_CFG = _load_dev_config()
    return os.environ.get(key_env) or _DEV_CFG.get(key_json, default)


def _build_cfg(mode: str) -> Dict:
    def req(env_k: str, json_k: str) -> str:
        v = _cfg_get(env_k, json_k)
        if not v:
            raise EnvironmentError(
                f"Missing required credential: env var {env_k!r} or dev_config.json key {json_k!r}"
            )
        return v

    return {
        "gemini_api_key":         req("GEMINI_API_KEY", "gemini_api_key"),
        "qdrant_url":             req("QDRANT_URL",      "qdrant_url"),
        "qdrant_api_key":         req("QDRANT_API_KEY",  "qdrant_api_key"),
        "neo4j_uri":              _cfg_get("NEO4J_URI",      "neo4j_uri"),
        "neo4j_username":         _cfg_get("NEO4J_USER",     "neo4j_username", "neo4j"),
        "neo4j_password":         _cfg_get("NEO4J_PASSWORD", "neo4j_password"),
        "neo4j_database":         _cfg_get("NEO4J_DATABASE", "neo4j_database", "neo4j"),
        "neo4j_enabled":          bool(_cfg_get("NEO4J_URI", "neo4j_uri")),
        "collection_name":        _doc_collection_for(mode),   # isolated per mode
        "graph_collection_name":  GRAPH_COLLECTION,
        "rag_mode":               mode,
        "setup_completed":        True,
        "preferred_llm":          "raw",
        "embedding_dimension":    768,
        "chunk_size":             800,
        "chunk_overlap":          150,
        "top_k":                  5,
        "rate_limit":             300,
        "enable_cache":           False,
        "cache_dir":              None,
        "max_entities_per_chunk": 15,
        "graph_reasoning_depth":  2,
        "entity_similarity_threshold": 0.8,
        "graph_extraction_timeout": 60,
        "max_chunk_length_for_graph": 800,
        "enable_agentic_ai":      False,
        "enable_query_planning":  True,
        "enable_reranking":       True,
        "enable_metadata_extraction": False,
        "pageindex_enabled":      False,
        "pageindex_workspace":    "/tmp/mh_pi_ws",
        "relationship_extraction_prompt": "extract_relationships",
        "active_domain":             "vc_financial",  # default, not used for multihop
        "benchmark_corpus_tag":      BENCHMARK_CORPUS_TAG,
        "max_cooccurrence_per_chunk": 10,
        "max_cooccurrence_edges_total": 100,
    }


_rag_instances: Dict[str, Any] = {}


def _get_rag(mode: str):
    if mode not in _rag_instances:
        from config import ConfigManager
        from simple_rag import SimpleRAG
        cfg = _build_cfg(mode)
        cm = ConfigManager.__new__(ConfigManager)
        cm.config_path = f"/tmp/multihop_bench_cfg_{mode}.json"
        cm.force_fresh_start = False
        cm.config = cfg
        _rag_instances[mode] = (SimpleRAG(cm), cfg)
    return _rag_instances[mode]


def _reset_benchmark_collections(modes: List[str]):
    """Delete benchmark-specific Qdrant collections and Neo4j corpus data before a run.

    Call with --reset to prevent stale data from a previous smoke contaminating results.
    Only touches collections/nodes tagged with BENCHMARK_CORPUS_TAG.
    """
    for mode in modes:
        rag, _ = _get_rag(mode)
        client = rag.vector_db_service.client

        for col in ([_doc_collection_for(mode)] + ([GRAPH_COLLECTION] if mode == "graph" else [])):
            try:
                client.delete_collection(col)
                print(f"  [reset] deleted Qdrant collection '{col}'")
            except Exception:
                pass  # collection may not exist yet — fine

        if mode == "graph":
            neo4j_svc = getattr(rag, "neo4j_service", None)
            if neo4j_svc:
                try:
                    with neo4j_svc._session() as session:
                        result = session.run(
                            "MATCH (n) WHERE n.benchmark_corpus = $bm DETACH DELETE n "
                            "RETURN count(n) AS deleted",
                            bm=BENCHMARK_CORPUS_TAG,
                        ).single()
                        print(f"  [reset] deleted {result['deleted']} Neo4j nodes "
                              f"with benchmark_corpus='{BENCHMARK_CORPUS_TAG}'")
                except Exception as e:
                    print(f"  [reset] Neo4j clear warning: {e}")

    # Force re-init of RAG instances so fresh collections are created on next index
    _rag_instances.clear()


# ── corpus document → temp txt file ──────────────────────────────────────────

def _corpus_doc_to_text(row: Dict) -> Tuple[str, Dict]:
    """
    Convert a corpus row to (plain_text, metadata).
    Handles both camelCase and snake_case field names from HF.
    """
    def _get(*keys):
        for k in keys:
            v = row.get(k, "")
            if v:
                return str(v)
        return ""

    title   = _get("title", "Title")
    body    = _get("body", "content", "text", "article", "Body")
    author  = _get("author", "Author")
    source  = _get("source", "Source", "publisher")
    pub_at  = _get("published_at", "publishedAt", "date", "Published_at")
    url     = _get("url", "link", "Url")
    cat     = _get("category", "Category", "topic")

    text = f"Title: {title}\n\n{body}" if title and not body.startswith(title) else body
    if not text.strip():
        text = f"{title}\n{author}\n{source}"

    meta = {
        "title": title,
        "author": author,
        "source": source,
        "published_at": pub_at,
        "url": url,
        "category": cat,
        "benchmark_corpus": BENCHMARK_CORPUS_TAG,
    }
    return text.strip(), meta


def _write_tmp_txt(text: str, path: str):
    Path(path).write_text(text, encoding="utf-8")


# ── evidence recall helpers ───────────────────────────────────────────────────

def _parse_evidence_list(evidence_list_raw) -> List[str]:
    """
    evidence_list can be a JSON string, a Python list, or a plain string.
    Returns a list of strings (title/source fragments to match in contexts).
    """
    if isinstance(evidence_list_raw, list):
        items = evidence_list_raw
    elif isinstance(evidence_list_raw, str):
        s = evidence_list_raw.strip()
        if s.startswith("["):
            try:
                items = json.loads(s)
            except Exception:
                items = [s]
        else:
            items = [s]
    else:
        items = [str(evidence_list_raw)] if evidence_list_raw else []

    result = []
    for item in items:
        if isinstance(item, dict):
            # Evidence items may be dicts with title/source/fact keys
            for k in ("title", "source", "fact", "text", "url"):
                v = item.get(k, "")
                if v:
                    result.append(str(v)[:200])
                    break
            else:
                result.append(str(item)[:200])
        elif isinstance(item, str) and item.strip():
            result.append(item.strip()[:200])
    return result


def _evidence_recall(evidence_strings: List[str], contexts: List[Dict]) -> Tuple[float, float]:
    """
    Returns (partial_recall, full_hop_recall).

    partial_recall  — fraction of evidence items found in at least one context
    full_hop_recall — 1.0 if ALL evidence items are covered, else 0.0
    """
    if not evidence_strings:
        return 0.0, 0.0

    all_ctx_text = " ".join(
        (c.get("text", "") + " " + json.dumps(c.get("metadata", {})))
        for c in contexts
    ).lower()

    hits = sum(1 for ev in evidence_strings if ev.lower()[:80] in all_ctx_text)
    partial = hits / len(evidence_strings)
    full    = 1.0 if hits == len(evidence_strings) else 0.0
    return partial, full


# ── sanity checks — fail loudly ───────────────────────────────────────────────

def _sanity_check_qdrant(rag, mode: str):
    client = rag.vector_db_service.client

    def _count(name):
        try:
            return client.get_collection(name).points_count
        except Exception as e:
            print(f"\n  ERROR: collection '{name}' not found: {e}")
            sys.exit(1)

    doc_col   = _doc_collection_for(mode)
    doc_count = _count(doc_col)
    print(f"  [sanity] {doc_col}: {doc_count} points")
    if doc_count == 0:
        print(f"\n  ERROR: doc collection has 0 points — indexing must have failed.")
        sys.exit(1)

    if mode == "graph":
        graph_count = _count(GRAPH_COLLECTION)
        print(f"  [sanity] {GRAPH_COLLECTION}: {graph_count} points")
        if graph_count == 0:
            print(f"\n  ERROR: graph collection has 0 points — graph indexing failed.")
            sys.exit(1)

        neo4j_svc = getattr(rag, "neo4j_service", None)
        if neo4j_svc is None:
            print("  [sanity] Neo4j not configured — graph mode will use vector-only traversal.")
        else:
            try:
                ns = neo4j_svc.get_graph_stats()
                n_nodes = ns.get("nodes", 0)
                n_rels  = ns.get("relationships", 0)
                print(f"  [sanity] Neo4j: {n_nodes} nodes, {n_rels} relationships")
                if n_nodes == 0 or n_rels == 0:
                    print(f"\n  ERROR: Neo4j graph is empty for graph mode. Index first.")
                    sys.exit(1)

                # Check neo4j_traversal actually fires
                sample_ctxs = rag.graph_rag_service.search_graph(
                    "test query for traversal check", top_k=3, neo4j_service=neo4j_svc
                )
                traversal_count = sum(
                    1 for c in sample_ctxs
                    if c.get("metadata", {}).get("discovery_method") == "neo4j_traversal"
                )
                print(f"  [sanity] neo4j_traversal contexts on probe query: {traversal_count}")
                if traversal_count == 0:
                    print("  WARNING: Neo4j configured but traversal returned 0 contexts on probe.")
                    print("           Graph mode will return vector-only graph results.")
            except Exception as e:
                print(f"  WARNING: Neo4j sanity check error: {e}")


# ── indexing ──────────────────────────────────────────────────────────────────

def _index_one_doc(args: Tuple) -> Tuple[int, bool, dict]:
    """Index a single doc — runs in a thread worker."""
    import tempfile
    i, text, meta, mode = args
    if not text.strip():
        return i, False, {}
    rag, _ = _get_rag(mode)
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, prefix=f"mh_{i}_", dir="/tmp")
    try:
        tmp.write(text.encode("utf-8", errors="replace"))
        tmp.flush()
        tmp.close()
        result = rag.index_document(tmp.name, extra_metadata=meta)
        return i, result.get("success", False), result
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _index_corpus_docs(mode: str, docs: List[Tuple[str, Dict]]) -> int:
    """
    Index (text, metadata) pairs into SimpleRAG in parallel (4 workers).
    Returns count of successfully indexed docs.
    """
    import concurrent.futures
    indexed = 0
    args = [(i, text, meta, mode) for i, (text, meta) in enumerate(docs)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_index_one_doc, a): a[0] for a in args}
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            i, success, result = future.result()
            done_count += 1
            if success:
                indexed += 1
                skipped = result.get("skipped", False)
                if done_count % 5 == 0 or done_count == 1 or done_count == len(docs):
                    tag = " (cached)" if skipped else ""
                    print(f"    [{done_count}/{len(docs)}] indexed{tag} — "
                          f"chunks={result.get('chunks_indexed',0)} "
                          f"entities={result.get('entities_extracted',0)}")
            else:
                logger.warning("index failed for doc %d: %s", i, result.get("error", "?"))

    return indexed


# ── main benchmark runner ─────────────────────────────────────────────────────

def run_multihop_real(
    modes: List[str],
    n: int,
    corpus_mode: str,
    out_path: str,
    seed: int = 42,
    reset: bool = False,
):
    print("\n" + "=" * 70)
    print("  REAL MultiHop-RAG Benchmark (yixuantt/MultiHopRAG, COLM 2024)")
    print("  Corpus: HuggingFace dataset — NOT hand-written scaffolds")
    print("  This is a mini real smoke unless --corpus-mode full is used.")
    print("=" * 70)

    # ── 1. Load dataset ────────────────────────────────────────────────────
    # Local cache lives in eval/multihop_cache/ — avoids HF network checks on every run.
    _CACHE_DIR  = Path(__file__).parent / "multihop_cache"
    _CORPUS_F   = _CACHE_DIR / "corpus.json"
    _QA_F       = _CACHE_DIR / "qa.json"

    print("\n[1/5] Loading MultiHop-RAG dataset…")

    class _DS:
        """Minimal dataset proxy — drop-in for HuggingFace Dataset in this harness."""
        def __init__(self, rows: list):
            self._rows = rows
            self.column_names = list(rows[0].keys()) if rows else []
        def __getitem__(self, i): return self._rows[i]
        def __iter__(self):       return iter(self._rows)
        def __len__(self):        return len(self._rows)

    if _CORPUS_F.exists() and _QA_F.exists():
        # Fast path — load from local JSON (no network, <1s)
        print(f"  Loading from local cache ({_CACHE_DIR})…")
        corpus_ds = _DS(json.loads(_CORPUS_F.read_text()))
        qa_ds     = _DS(json.loads(_QA_F.read_text()))
        print(f"  corpus rows: {len(corpus_ds)},  QA rows: {len(qa_ds)}")
    else:
        # First-run — download from HuggingFace and save locally
        print("  First run — downloading from HuggingFace and caching locally…")
        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: 'datasets' package not installed. Run: pip install datasets")
            sys.exit(1)

        os.environ["HF_DATASETS_OFFLINE"] = "0"  # allow download on first run
        try:
            corpus_ds = load_dataset("yixuantt/MultiHopRAG", "corpus",      split="train")
            qa_ds     = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train")
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            sys.exit(1)

        # Save to local cache then wrap in _DS so the rest of the harness is identical
        _CACHE_DIR.mkdir(exist_ok=True)
        corpus_rows = list(corpus_ds)
        qa_rows     = list(qa_ds)
        _CORPUS_F.write_text(json.dumps(corpus_rows, default=str))
        _QA_F.write_text(json.dumps(qa_rows,     default=str))
        corpus_ds = _DS(corpus_rows)
        qa_ds     = _DS(qa_rows)
        print(f"  Saved to local cache ({_CACHE_DIR}) — future runs will be instant.")
        print(f"  corpus rows: {len(corpus_ds)},  QA rows: {len(qa_ds)}")

    _inspect_dataset(corpus_ds, "corpus", sample_idx=0)
    _inspect_dataset(qa_ds, "MultiHopRAG QA", sample_idx=0)

    print(f"\n  corpus rows: {len(corpus_ds)},  QA rows: {len(qa_ds)}")

    # ── 2. Select QA questions ─────────────────────────────────────────────
    print(f"\n[2/5] Selecting {n} QA questions…")
    rng = random.Random(seed)

    # Identify field names dynamically
    qa_cols = set(qa_ds.column_names)
    corpus_cols = set(corpus_ds.column_names)

    q_field   = next((f for f in ("query", "question", "Question") if f in qa_cols), qa_ds.column_names[0])
    ans_field = next((f for f in ("answer", "Answer", "golden_answer") if f in qa_cols), None)
    ev_field  = next((f for f in ("evidence_list", "evidence", "evidences") if f in qa_cols), None)
    qt_field  = next((f for f in ("question_type", "type", "category") if f in qa_cols), None)

    print(f"  QA fields mapped: query={q_field!r} answer={ans_field!r} "
          f"evidence={ev_field!r} type={qt_field!r}")

    all_qa = list(qa_ds)
    selected_qa = rng.sample(all_qa, min(n, len(all_qa)))
    print(f"  Selected {len(selected_qa)} questions")

    # ── 3. Build corpus subset ─────────────────────────────────────────────
    print(f"\n[3/5] Building corpus for indexing (corpus_mode={corpus_mode})…")

    # Identify corpus id/title field
    corp_title_field = next(
        (f for f in ("title", "Title", "id", "doc_id") if f in corpus_cols),
        corpus_ds.column_names[0],
    )
    all_corpus = list(corpus_ds)
    corpus_by_title: Dict[str, Dict] = {}
    for row in all_corpus:
        t = str(row.get(corp_title_field, ""))
        if t:
            corpus_by_title[t] = row

    # Collect evidence documents needed for selected questions
    evidence_titles: set = set()
    if ev_field:
        for qa in selected_qa:
            ev_raw = qa.get(ev_field, [])
            ev_strings = _parse_evidence_list(ev_raw)
            for ev in ev_strings:
                # evidence strings often contain the document title/source
                evidence_titles.add(ev[:200])

    # Match evidence titles to corpus rows
    def _best_corpus_match(ev_title: str) -> Optional[Dict]:
        ev_lower = ev_title.lower()
        for title, row in corpus_by_title.items():
            if ev_lower in title.lower() or title.lower() in ev_lower:
                return row
        return None

    evidence_rows: List[Dict] = []
    for ev in evidence_titles:
        match = _best_corpus_match(ev)
        if match and match not in evidence_rows:
            evidence_rows.append(match)

    if corpus_mode == "mini":
        # Evidence docs + 25 random distractors
        non_evidence = [r for r in all_corpus if r not in evidence_rows]
        n_distract = max(0, 25 - len(evidence_rows))
        distractors = rng.sample(non_evidence, min(n_distract, len(non_evidence)))
        corpus_subset = evidence_rows + distractors
        print(f"  mini mode: {len(evidence_rows)} evidence docs + {len(distractors)} distractors "
              f"= {len(corpus_subset)} total")
    else:
        corpus_subset = all_corpus
        print(f"  full mode: all {len(corpus_subset)} corpus docs")

    # Convert to (text, metadata) pairs
    docs_to_index: List[Tuple[str, Dict]] = []
    for row in corpus_subset:
        text, meta = _corpus_doc_to_text(row)
        if text.strip():
            docs_to_index.append((text, meta))

    print(f"  {len(docs_to_index)} valid docs to index")

    # ── 4. Index ───────────────────────────────────────────────────────────
    if reset:
        print("\n[4/5-pre] Resetting benchmark collections (--reset flag set)…")
        _reset_benchmark_collections(modes)

    for mode in modes:
        print(f"\n[4/5] Indexing for mode={mode}…")
        rag, _ = _get_rag(mode)
        # Set mode on rag instance so index_document branches correctly
        rag.rag_mode = mode
        if hasattr(rag, "document_processor") and rag.document_processor:
            rag.document_processor.rag_mode = mode

        indexed = _index_corpus_docs(mode, docs_to_index)
        print(f"  Indexed {indexed}/{len(docs_to_index)} docs for mode={mode}")
        if indexed == 0:
            print(f"\n  ERROR: 0 docs indexed for mode={mode}. Aborting.")
            sys.exit(1)

        # Sanity check
        print(f"\n[sanity] mode={mode}")
        _sanity_check_qdrant(rag, mode)

    # ── 5. Evaluate ────────────────────────────────────────────────────────
    print(f"\n[5/5] Evaluating {len(selected_qa)} questions across modes: {modes}")

    all_mode_results: Dict[str, Any] = {}

    for mode in modes:
        rag, cfg = _get_rag(mode)
        rag.rag_mode = mode

        print(f"\n  [eval] mode={mode}")
        mode_rows: List[Dict] = []
        total_doc_ctx          = 0
        total_graph_ctx        = 0
        total_traversal        = 0
        total_semantic_trav    = 0
        total_cooccur_trav     = 0

        q_type_stats: Dict[str, Dict] = {}

        for qi, qa in enumerate(selected_qa):
            question  = str(qa.get(q_field, "")).strip()
            golden    = str(qa.get(ans_field, "")).strip() if ans_field else ""
            ev_raw    = qa.get(ev_field, []) if ev_field else []
            q_type    = str(qa.get(qt_field, "unknown")).strip() if qt_field else "unknown"

            if not question:
                continue

            ev_strings = _parse_evidence_list(ev_raw)

            t0 = time.time()
            try:
                out = rag.query_debug(question)
            except Exception as e:
                logger.error("query_debug error for Q%d: %s", qi + 1, e)
                out = {"answer": f"ERROR: {e}", "contexts_doc": [], "contexts_graph": []}
            elapsed = time.time() - t0

            answer    = out.get("answer", "")
            doc_ctxs  = out.get("contexts_doc", [])
            graph_ctxs = out.get("contexts_graph", [])
            all_ctxs  = doc_ctxs + graph_ctxs

            n_doc_ctx   = len(doc_ctxs)
            n_graph_ctx = len(graph_ctxs)
            n_traversal = sum(
                1 for c in graph_ctxs
                if c.get("metadata", {}).get("discovery_method") == "neo4j_traversal"
            )
            n_semantic_trav = sum(
                1 for c in graph_ctxs
                if c.get("metadata", {}).get("discovery_method") == "neo4j_traversal"
                and c.get("metadata", {}).get("provenance") != "co_occurrence"
            )
            n_cooccur_trav = sum(
                1 for c in graph_ctxs
                if c.get("metadata", {}).get("discovery_method") == "neo4j_traversal"
                and c.get("metadata", {}).get("provenance") == "co_occurrence"
            )

            total_doc_ctx       += n_doc_ctx
            total_graph_ctx     += n_graph_ctx
            total_traversal     += n_traversal
            total_semantic_trav += n_semantic_trav
            total_cooccur_trav  += n_cooccur_trav

            # Evidence recall
            partial_recall, full_hop_recall = _evidence_recall(ev_strings, all_ctxs)

            # Exact/normalized match (simple substring)
            exact_match = (
                golden.lower() in answer.lower() or answer.lower() in golden.lower()
                if golden and answer else False
            )

            # LLM judge (only if golden available)
            if golden:
                passed = judge_answer(question, golden, answer)
            else:
                passed = None

            icon = "✓" if passed else ("?" if passed is None else "✗")
            print(f"    {icon} [Q{qi+1}] type={q_type[:12]} judge={passed} "
                  f"ev_recall={partial_recall:.2f} full_hop={full_hop_recall:.0f} "
                  f"doc_ctx={n_doc_ctx} graph_ctx={n_graph_ctx} "
                  f"traversal={n_traversal}(sem={n_semantic_trav},co={n_cooccur_trav}) "
                  f"{elapsed:.1f}s")

            row = {
                "question":              question[:120],
                "golden":                golden[:120] if golden else "",
                "answer":                answer[:300] if answer else "",
                "question_type":         q_type,
                "passed":                passed,
                "exact_match":           exact_match,
                "evidence_recall":       round(partial_recall, 3),
                "full_hop_recall":       full_hop_recall,
                "n_evidence":            len(ev_strings),
                "doc_ctx_count":         n_doc_ctx,
                "graph_ctx_count":       n_graph_ctx,
                "traversal_count":       n_traversal,
                "semantic_traversal":    n_semantic_trav,
                "cooccurrence_traversal": n_cooccur_trav,
                "latency_s":             round(elapsed, 2),
            }
            mode_rows.append(row)

            # Per question-type aggregation
            if q_type not in q_type_stats:
                q_type_stats[q_type] = {"n": 0, "passed": 0, "ev_recall": 0.0}
            q_type_stats[q_type]["n"] += 1
            q_type_stats[q_type]["passed"] += (1 if passed else 0)
            q_type_stats[q_type]["ev_recall"] += partial_recall

        # Aggregate
        n_q       = len(mode_rows)
        n_judged  = sum(1 for r in mode_rows if r["passed"] is not None)
        n_pass    = sum(1 for r in mode_rows if r["passed"])
        acc       = n_pass / n_judged * 100 if n_judged else 0.0
        avg_ev    = sum(r["evidence_recall"]  for r in mode_rows) / n_q if n_q else 0.0
        avg_full  = sum(r["full_hop_recall"]  for r in mode_rows) / n_q if n_q else 0.0
        avg_lat   = sum(r["latency_s"]        for r in mode_rows) / n_q if n_q else 0.0
        avg_doc        = total_doc_ctx       / n_q if n_q else 0.0
        avg_graph      = total_graph_ctx     / n_q if n_q else 0.0
        avg_trav       = total_traversal     / n_q if n_q else 0.0
        avg_sem_trav   = total_semantic_trav / n_q if n_q else 0.0
        avg_co_trav    = total_cooccur_trav  / n_q if n_q else 0.0

        print(f"\n  [{mode}] accuracy={acc:.0f}% ({n_pass}/{n_judged}) "
              f"ev_recall={avg_ev:.2f} full_hop={avg_full:.2f} "
              f"avg_doc_ctx={avg_doc:.1f} avg_graph_ctx={avg_graph:.1f} "
              f"avg_traversal={avg_trav:.1f}(sem={avg_sem_trav:.1f},co={avg_co_trav:.1f}) "
              f"avg_lat={avg_lat:.1f}s")

        # Warn if avg context is 0
        if avg_doc + avg_graph == 0:
            print(f"\n  ERROR: avg retrieved context count is 0 for mode={mode}.")
            print("  Answer accuracy is unreliable — system returned no retrieved context.")
            sys.exit(1)

        # Per question-type breakdown
        if q_type_stats:
            print(f"\n  Question-type breakdown [{mode}]:")
            for qt, s in sorted(q_type_stats.items(), key=lambda x: -x[1]["n"]):
                qt_acc = s["passed"] / s["n"] * 100 if s["n"] else 0.0
                qt_ev  = s["ev_recall"] / s["n"] if s["n"] else 0.0
                print(f"    {qt[:20]:20s}  n={s['n']:3d}  acc={qt_acc:5.0f}%  ev_recall={qt_ev:.2f}")

        all_mode_results[mode] = {
            "suite":                      "multihop_real",
            "corpus_mode":                corpus_mode,
            "n_questions":                n_q,
            "n_judged":                   n_judged,
            "accuracy_pct":               round(acc, 1),
            "avg_evidence_recall":        round(avg_ev, 3),
            "avg_full_hop_recall":        round(avg_full, 3),
            "avg_doc_ctx_count":          round(avg_doc, 1),
            "avg_graph_ctx_count":        round(avg_graph, 1),
            "avg_traversal_count":        round(avg_trav, 1),
            "avg_semantic_traversal":     round(avg_sem_trav, 1),
            "avg_cooccurrence_traversal": round(avg_co_trav, 1),
            "avg_latency_s":              round(avg_lat, 2),
            "q_type_breakdown":  {
                qt: {
                    "n": s["n"],
                    "accuracy_pct": round(s["passed"] / s["n"] * 100 if s["n"] else 0.0, 1),
                    "avg_ev_recall": round(s["ev_recall"] / s["n"] if s["n"] else 0.0, 3),
                }
                for qt, s in q_type_stats.items()
            },
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "per_question":      mode_rows,
        }

    return all_mode_results


# ── Markdown report ───────────────────────────────────────────────────────────

def _write_markdown(results: Dict, out_md: str, corpus_mode: str, n: int, modes: List[str]):
    is_mini = corpus_mode == "mini"
    lines = [
        "# MultiHop-RAG Real Benchmark Results",
        "",
        f"**Dataset**: yixuantt/MultiHopRAG (HuggingFace, COLM 2024)",
        f"**Corpus mode**: {corpus_mode} ({'evidence docs + 25 distractors' if is_mini else 'full 609-doc corpus'})",
        f"**Questions**: {n}",
        f"**Modes evaluated**: {', '.join(modes)}",
        f"**Generated**: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "---",
        "",
        "## Benchmark Context",
        "",
        "- **This is a real MultiHop-RAG corpus smoke** (actual HuggingFace corpus rows).",
        "- It is **NOT a full benchmark** until the full 609-doc corpus and n ≥ 50 are used.",
        "- **FinanceBench** remains the best benchmark for Normal RAG on financial 10-K QA.",
        "- **MultiHop-RAG** (this dataset) is the best current benchmark for our Neo4j Graph RAG.",
        "- **Microsoft BenchmarkQED** is NOT applicable: SimpleRAGx has no community summaries.",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Mode | Accuracy | Ev Recall | Full-Hop | Avg Doc Ctx | Avg Graph Ctx | Avg Traversal | Avg Latency |",
        "|------|----------|-----------|----------|-------------|---------------|---------------|-------------|",
    ]

    for mode, r in results.items():
        lines.append(
            f"| {mode} | {r['accuracy_pct']}% ({r['n_judged']}q) | "
            f"{r['avg_evidence_recall']:.3f} | {r['avg_full_hop_recall']:.3f} | "
            f"{r['avg_doc_ctx_count']} | {r['avg_graph_ctx_count']} | "
            f"{r['avg_traversal_count']} | {r['avg_latency_s']}s |"
        )

    lines += [
        "",
        "---",
        "",
        "## Metrics Glossary",
        "",
        "- **Accuracy**: LLM judge (Gemini) scores generated answer vs. golden answer.",
        "- **Ev Recall**: fraction of evidence items found in retrieved contexts (partial).",
        "- **Full-Hop**: fraction of questions where ALL required evidence was retrieved.",
        "- **Avg Traversal**: avg Neo4j traversal contexts per query (graph mode only).",
        "",
    ]

    # Per question-type breakdown for each mode
    for mode, r in results.items():
        qt = r.get("q_type_breakdown", {})
        if qt:
            lines.append(f"## Question-Type Breakdown — {mode}")
            lines.append("")
            lines.append("| Type | N | Accuracy | Ev Recall |")
            lines.append("|------|---|----------|-----------|")
            for qtype, s in sorted(qt.items(), key=lambda x: -x[1]["n"]):
                lines.append(f"| {qtype} | {s['n']} | {s['accuracy_pct']}% | {s['avg_ev_recall']:.3f} |")
            lines.append("")

    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown report → {out_md}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _validate_env():
    pairs = [("GEMINI_API_KEY", "gemini_api_key"),
             ("QDRANT_URL",     "qdrant_url"),
             ("QDRANT_API_KEY", "qdrant_api_key")]
    missing = [env_k for env_k, json_k in pairs if not _cfg_get(env_k, json_k)]
    if missing:
        print(f"ERROR: Missing required credentials: {missing}")
        print("Either export env vars or add them to dev_config.json at the project root.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Real MultiHop-RAG benchmark for SimpleRAGx")
    parser.add_argument("--n", type=int, default=5, help="Number of QA questions to evaluate")
    parser.add_argument(
        "--corpus-mode", choices=["mini", "full"], default="mini",
        help="mini=evidence+25 distractors, full=all 609 corpus docs",
    )
    parser.add_argument(
        "--modes", nargs="+", default=["normal", "graph"],
        choices=["normal", "graph"],
        help="RAG modes to evaluate",
    )
    parser.add_argument(
        "--out", default=str(RESULTS_DIR / "multihop_real_smoke.json"),
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for question/distractor selection")
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete and recreate multihop_* Qdrant collections and Neo4j corpus data before indexing. "
             "Required to prevent stale data from a previous smoke run.",
    )
    args = parser.parse_args()

    _validate_env()
    RESULTS_DIR.mkdir(exist_ok=True)

    results = run_multihop_real(
        modes=args.modes,
        n=args.n,
        corpus_mode=args.corpus_mode,
        out_path=args.out,
        seed=args.seed,
        reset=args.reset,
    )

    # Write JSON
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON results → {args.out}")

    # Write Markdown
    md_path = args.out.replace(".json", ".md")
    _write_markdown(results, md_path, args.corpus_mode, args.n, args.modes)

    # Final summary
    print("\n" + "=" * 70)
    print("  BENCHMARK CONTEXT REMINDER")
    print("  This is a REAL MultiHop-RAG corpus smoke (actual HF dataset rows).")
    if args.corpus_mode == "mini":
        print(f"  It is NOT the full benchmark (only {args.n}q, mini corpus).")
        print("  For full: --n 50 --corpus-mode full")
    print("  FinanceBench = best Normal RAG / financial 10-K benchmark.")
    print("  MultiHop-RAG = best Graph RAG multi-hop benchmark.")
    print("  BenchmarkQED = NOT applicable (no community summaries in SimpleRAGx).")
    print("=" * 70)


if __name__ == "__main__":
    main()
