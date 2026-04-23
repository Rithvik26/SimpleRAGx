#!/usr/bin/env python3
"""
SimpleRAGx RAGAS Evaluation Harness
=====================================
Runs ground-truth questions through each RAG mode and computes
RAGAS metrics: faithfulness, answer_relevancy, context_precision,
context_recall. Exports per-mode per-metric JSON.

Usage:
    python eval/ragas_harness.py [--modes normal graph neo4j pageindex] [--out eval/results.json]

Environment variables required (same as .env.example):
    GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
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
DEFAULT_OUT       = os.path.join(os.path.dirname(__file__), "results.json")
TEST_PDF          = os.path.join(os.path.dirname(os.path.dirname(__file__)), "techcorp_q1_2025_report.pdf")

SUPPORTED_MODES = ["normal", "graph", "neo4j", "pageindex"]


def _load_ground_truth() -> List[Dict]:
    with open(GROUND_TRUTH_PATH) as f:
        data = yaml.safe_load(f)
    return data["questions"]


def _query_mode(mode: str, question: str, cfg: Dict) -> Dict[str, Any]:
    """Run a single question through the given mode. Returns {answer, contexts}."""
    from simple_rag import SimpleRAG
    rag = SimpleRAG(cfg)
    rag.set_rag_mode(mode)

    if not rag.is_ready():
        # Load test PDF if index is empty
        rag.process_document(TEST_PDF)

    result = rag.query(question)
    answer   = result.get("answer", "")
    contexts = [c.get("text", c.get("content", "")) for c in result.get("sources", [])]
    return {"answer": answer, "contexts": contexts}


def _compute_ragas(samples: List[Dict]) -> Dict[str, float]:
    """Compute RAGAS metrics for a list of {question, answer, contexts, ground_truth} dicts."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError:
        print("RAGAS not installed. Run: pip install ragas datasets")
        return {}

    dataset = Dataset.from_list(samples)
    result  = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return {k: round(float(v), 4) for k, v in result.items()}


def _build_cfg() -> Dict:
    def req(k):
        v = os.environ.get(k)
        if not v:
            raise EnvironmentError(f"Missing env var: {k}. See .env.example.")
        return v

    return {
        "gemini_api_key":   req("GEMINI_API_KEY"),
        "qdrant_url":       req("QDRANT_URL"),
        "qdrant_api_key":   req("QDRANT_API_KEY"),
        "neo4j_uri":        os.environ.get("NEO4J_URI", ""),
        "neo4j_username":   os.environ.get("NEO4J_USER", "neo4j"),
        "neo4j_password":   os.environ.get("NEO4J_PASSWORD", ""),
        "neo4j_database":   os.environ.get("NEO4J_DATABASE", "neo4j"),
        "neo4j_enabled":    bool(os.environ.get("NEO4J_URI")),
        "collection_name":        "ragas_eval",
        "graph_collection_name":  "ragas_eval_graph",
        "rag_mode":         "normal",
        "setup_completed":  True,
        "preferred_llm":    "raw",
        "embedding_dimension": 768,
        "chunk_size":       1000,
        "chunk_overlap":    200,
        "top_k":            5,
        "enable_cache":     False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["normal", "graph"],
                        choices=SUPPORTED_MODES, help="Modes to evaluate")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args()

    questions = _load_ground_truth()
    cfg       = _build_cfg()
    all_results: Dict[str, Any] = {}

    for mode in args.modes:
        print(f"\n{'─'*60}")
        print(f"  Evaluating mode: {mode}  ({len(questions)} questions)")
        print(f"{'─'*60}")

        samples = []
        for qt in questions:
            q  = qt["question"]
            gt = qt["expected_answer"].strip()
            try:
                t0 = time.time()
                out = _query_mode(mode, q, cfg)
                elapsed = round(time.time() - t0, 2)
                print(f"  [{elapsed}s] {q[:70]}")
                samples.append({
                    "question":     q,
                    "answer":       out["answer"],
                    "contexts":     out["contexts"] or [""],
                    "ground_truth": gt,
                })
            except Exception as e:
                print(f"  [ERROR] {q[:70]}: {e}")
                samples.append({
                    "question":     q,
                    "answer":       "",
                    "contexts":     [""],
                    "ground_truth": gt,
                })

        metrics = _compute_ragas(samples)
        all_results[mode] = {"metrics": metrics, "n_questions": len(samples)}
        print(f"\n  {mode} metrics: {metrics}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
