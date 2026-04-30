#!/usr/bin/env python3
"""
FinanceBench-style eval using the already-indexed AMD + Boeing 10-Ks.

Reads ground_truth.yaml (14 questions), runs them through Normal and/or Graph
mode against the real app collections, judges with LLM-as-judge (same as
multihop eval), reports accuracy + latency.

Usage:
    python eval/financebench_eval.py --modes normal graph
    python eval/financebench_eval.py --modes graph --out eval/results/fb_graph_v1.json

Environment variables required (same as main app):
    GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY
Optional (graph mode Neo4j):
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import yaml

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

RESULTS_DIR      = Path(__file__).parent / "results"
GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.yaml"
GEMINI_JUDGE     = "gemini/gemini-2.5-flash"

_JUDGE_PROMPT = """\
You are evaluating whether an AI-generated answer correctly answers a financial question.

Question: {question}
Golden answer: {golden}
AI-generated answer: {generated}

Rules:
- TRUE if the AI answer contains the key facts from the golden answer (numbers, names, conclusions).
- For Yes/No questions: TRUE if the AI starts with or clearly states the same Yes/No as the golden.
- Minor rounding differences (e.g. 39% vs ~39%), partial name matches, and extra context are acceptable.
- Ignore trailing disclaimers like "documents do not contain..." — judge only the stated answer.
- "I don't know" / "insufficient information" = WRONG (golden answers are always known here).

Reply with exactly one word: TRUE or FALSE"""


def _judge(question: str, golden: str, generated: str) -> bool:
    if not generated or not generated.strip():
        return False
    import litellm
    os.environ.setdefault("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    try:
        resp = litellm.completion(
            model=GEMINI_JUDGE,
            messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
                question=question, golden=golden.strip(), generated=generated.strip()
            )}],
            temperature=0,
            max_tokens=10,
            extra_body={"generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}},
        )
        return "true" in (resp.choices[0].message.content or "").strip().lower()
    except Exception as e:
        print(f"  [judge error] {e}")
        return False


def _build_rag(mode: str):
    """Build a SimpleRAG instance pointed at the real app collections."""
    import json as _json
    dev_cfg_path = _ROOT / "dev_config.json"
    with open(dev_cfg_path) as f:
        cfg = _json.load(f)

    cfg["rag_mode"] = mode
    cfg["top_k"] = 10

    from config import ConfigManager
    cm = ConfigManager.__new__(ConfigManager)
    cm.config = cfg
    cm.config_path = str(dev_cfg_path)

    from simple_rag import EnhancedSimpleRAG
    rag = EnhancedSimpleRAG(config_manager=cm)
    return rag


def _query(rag, mode: str, question: str) -> str:
    result = rag.query(question)
    if isinstance(result, dict):
        return result.get("answer", result.get("response", str(result)))
    return str(result)


def run_financebench(modes: List[str], out_path: str):
    with open(GROUND_TRUTH_PATH) as f:
        questions = yaml.safe_load(f)["questions"]

    all_results: Dict[str, Any] = {}

    for mode in modes:
        print(f"\n{'─'*60}")
        print(f"  Mode: {mode}   ({len(questions)} questions)")
        print(f"{'─'*60}")

        rag = _build_rag(mode)
        per_q = []
        n_correct = 0

        for qt in questions:
            q       = qt["question"]
            golden  = qt["expected_answer"].strip()
            doc_id  = qt["id"]

            t0 = time.time()
            try:
                answer = _query(rag, mode, q)
            except Exception as e:
                answer = f"ERROR: {e}"
            latency = round(time.time() - t0, 2)

            passed = _judge(q, golden, answer)
            n_correct += int(passed)

            mark = "✓" if passed else "✗"
            print(f"  [{mark}] [{latency:5.1f}s] [{doc_id}] {q[:65]}")
            if not passed:
                print(f"         GT:  {golden[:120]}")
                print(f"         GEN: {answer[:120]}")

            per_q.append({
                "id":       doc_id,
                "question": q,
                "golden":   golden,
                "answer":   answer,
                "passed":   passed,
                "latency_s": latency,
            })

        accuracy = round(100 * n_correct / len(questions), 1)
        avg_lat  = round(sum(r["latency_s"] for r in per_q) / len(per_q), 2)

        print(f"\n  Accuracy: {n_correct}/{len(questions)} = {accuracy}%")
        print(f"  Avg latency: {avg_lat}s")

        all_results[mode] = {
            "accuracy_pct":  accuracy,
            "n_correct":     n_correct,
            "n_questions":   len(questions),
            "avg_latency_s": avg_lat,
            "per_question":  per_q,
            "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Markdown summary
    md_lines = ["# FinanceBench Eval — AMD + Boeing 10-Ks\n"]
    md_lines.append(f"**Questions:** {len(questions)} (7 AMD FY2022, 7 Boeing FY2022)\n")
    md_lines.append(f"**Collections:** real app collections (already indexed)\n\n")
    md_lines.append("| Mode | Accuracy | Avg Latency |")
    md_lines.append("|------|----------|-------------|")
    for mode, r in all_results.items():
        md_lines.append(f"| {mode} | {r['accuracy_pct']}% ({r['n_correct']}/{r['n_questions']}) | {r['avg_latency_s']}s |")
    md = "\n".join(md_lines) + "\n"

    md_path = out_path.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write(md)

    print(f"\nJSON → {out_path}")
    print(f"MD   → {md_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="FinanceBench eval on real AMD+Boeing 10-Ks")
    parser.add_argument("--modes", nargs="+", default=["normal", "graph"],
                        choices=["normal", "graph"], help="RAG modes to evaluate")
    parser.add_argument("--out", default=str(RESULTS_DIR / "financebench_v1.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    run_financebench(modes=args.modes, out_path=args.out)


if __name__ == "__main__":
    main()
