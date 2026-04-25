#!/usr/bin/env python3
"""
SimpleRAGx RAGAS Evaluation Harness (RAGAS 0.4.x)
====================================================
Runs ground-truth questions through each RAG mode and computes
RAGAS metrics: faithfulness, answer_relevancy, context_precision,
context_recall. Exports per-mode per-metric JSON.

Usage:
    python eval/ragas_harness.py [--modes normal graph] [--out eval/baseline_phase1.json]

Environment variables required (same as .env.example):
    GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.yaml")
DEFAULT_OUT       = os.path.join(os.path.dirname(__file__), "baseline_phase1.json")
TEST_PDF          = os.path.join(os.path.dirname(os.path.dirname(__file__)), "techcorp_q1_2025_report.pdf")

SUPPORTED_MODES = ["normal", "graph", "neo4j", "pageindex"]


def _load_ground_truth() -> List[Dict]:
    with open(GROUND_TRUTH_PATH) as f:
        data = yaml.safe_load(f)
    return data["questions"]


def _build_cfg(collection_suffix: str = "") -> Dict:
    def req(k):
        v = os.environ.get(k)
        if not v:
            raise EnvironmentError(f"Missing env var: {k}. See .env.example.")
        return v

    return {
        "gemini_api_key":        req("GEMINI_API_KEY"),
        "qdrant_url":            req("QDRANT_URL"),
        "qdrant_api_key":        req("QDRANT_API_KEY"),
        "neo4j_uri":             os.environ.get("NEO4J_URI", ""),
        "neo4j_username":        os.environ.get("NEO4J_USER", "neo4j"),
        "neo4j_password":        os.environ.get("NEO4J_PASSWORD", ""),
        "neo4j_database":        os.environ.get("NEO4J_DATABASE", "neo4j"),
        "neo4j_enabled":         bool(os.environ.get("NEO4J_URI")),
        "collection_name":       f"ragas_eval{collection_suffix}",
        "graph_collection_name": f"ragas_eval_graph{collection_suffix}",
        "rag_mode":              "normal",
        "setup_completed":       True,
        "preferred_llm":         "raw",
        "embedding_dimension":   768,
        "chunk_size":            1000,
        "chunk_overlap":         200,
        "top_k":                 5,
        "rate_limit":            60,
        "enable_cache":          False,
        "cache_dir":             None,
        "max_entities_per_chunk": 20,
        "relationship_extraction_prompt": "extract_relationships",
        "graph_reasoning_depth": 2,
        "entity_similarity_threshold": 0.8,
        "graph_extraction_timeout": 60,
        "max_chunk_length_for_graph": 1000,
        "enable_agentic_ai":     False,
        "pageindex_workspace":   "/tmp/ragas_pi_ws",
    }


_rag_cache: Dict[str, Any] = {}  # mode -> SimpleRAG instance (reuse across questions)


def _query_mode(mode: str, question: str, cfg: Dict) -> Dict[str, Any]:
    """Run a single question through the given mode. Returns {answer, contexts}."""
    from config import ConfigManager
    from simple_rag import SimpleRAG

    if mode not in _rag_cache:
        # Build a ConfigManager from the plain cfg dict
        cm = ConfigManager.__new__(ConfigManager)
        cm.config_path = "/tmp/ragas_eval_config.json"
        cm.force_fresh_start = False
        cm.config = dict(cfg)
        cm.config["rag_mode"] = mode

        rag = SimpleRAG(cm)
        if not rag.is_ready():
            print(f"    Indexing {TEST_PDF} for mode={mode}…")
            rag.process_document(TEST_PDF)
        _rag_cache[mode] = rag
    else:
        rag = _rag_cache[mode]

    result = rag.query(question)
    answer   = result.get("answer", "")
    sources  = result.get("sources", result.get("context_chunks", []))
    contexts = [s.get("text", s.get("content", "")) for s in sources if s]
    return {"answer": answer, "contexts": contexts or [""]}


def _compute_ragas(samples: List[Dict], gemini_api_key: str) -> Dict[str, float]:
    """Compute RAGAS 0.4.x metrics for a list of sample dicts."""
    try:
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    except ImportError as e:
        print(f"Import error: {e}. Run: pip install ragas datasets langchain-google-genai")
        return {}

    # Configure RAGAS to use Gemini via the modern llm_factory / embeddings API
    try:
        from ragas.llms import llm_factory
        from ragas.embeddings import embedding_factory
        ragas_llm = llm_factory(
            model="gemini/gemini-2.0-flash",
            provider="litellm",
        )
        ragas_emb = embedding_factory(
            model="gemini/text-embedding-004",
            provider="litellm",
        )
    except Exception:
        # Fallback: LangchainLLMWrapper still works in 0.4.x (just deprecated)
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        lc_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_api_key,
            temperature=0,
        )
        lc_emb = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key,
        )
        ragas_llm = LangchainLLMWrapper(lc_llm)
        ragas_emb = LangchainEmbeddingsWrapper(lc_emb)

    ragas_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"] or [""],
            reference=s["ground_truth"],
        )
        for s in samples
    ]
    dataset = EvaluationDataset(samples=ragas_samples)

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]

    result = evaluate(dataset, metrics=metrics)

    # RAGAS 0.4.x: result.scores is a list of per-sample dicts; compute means
    scores = result.scores  # list[dict]
    if not scores:
        return {}
    metric_keys = [k for k in scores[0].keys() if k != "user_input"]
    averages: Dict[str, float] = {}
    for k in metric_keys:
        vals = [s[k] for s in scores if s.get(k) is not None]
        averages[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return averages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["normal"],
                        choices=SUPPORTED_MODES, help="Modes to evaluate")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise EnvironmentError("GEMINI_API_KEY not set. See .env.example.")

    questions = _load_ground_truth()
    all_results: Dict[str, Any] = {}

    for mode in args.modes:
        cfg = _build_cfg(collection_suffix=f"_{mode}")
        print(f"\n{'─'*60}")
        print(f"  Evaluating mode: {mode}  ({len(questions)} questions)")
        print(f"{'─'*60}")

        samples = []
        rag_instance = None

        for qt in questions:
            q  = qt["question"]
            gt = qt["expected_answer"].strip()
            try:
                t0 = time.time()
                out = _query_mode(mode, q, cfg)
                elapsed = round(time.time() - t0, 2)
                print(f"  [{elapsed:5.1f}s] {q[:65]}")
                samples.append({
                    "question":     q,
                    "answer":       out["answer"],
                    "contexts":     out["contexts"],
                    "ground_truth": gt,
                })
            except Exception as e:
                print(f"  [ERROR] {q[:65]}: {e}")
                samples.append({
                    "question":     q,
                    "answer":       "",
                    "contexts":     [""],
                    "ground_truth": gt,
                })

        print(f"\n  Computing RAGAS metrics for {mode}…")
        metrics = _compute_ragas(samples, gemini_key)
        all_results[mode] = {
            "metrics":     metrics,
            "n_questions": len(samples),
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        print(f"  {mode} → {metrics}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBaseline saved to {args.out}")


if __name__ == "__main__":
    main()
