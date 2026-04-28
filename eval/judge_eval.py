"""
LLM-as-judge evaluation for PageIndex — matches VectifyAI's Mafin2.5 eval methodology.

Usage:
    python eval/judge_eval.py                          # default: Flash Lite (~$0.07)
    python eval/judge_eval.py gemini/gemini-2.5-flash  # better (~$0.12)
    python eval/judge_eval.py gemini/gemini-2.5-pro    # best, matches GPT-4o (~$1.20)
"""

import os, sys, time, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import litellm
from config import get_config_manager
from pageindex_service import PageIndexService

DEFAULT_MODEL = "gemini/gemini-2.5-flash"

JUDGE_PROMPT = """You are evaluating whether an AI-generated answer correctly answers a financial question.

Question: {question}

Golden answer (human expert): {golden}

AI-generated answer: {generated}

Rules:
- Minor rounding differences are acceptable (e.g. "0.62%" vs "0.6%")
- Equivalent representations are acceptable (e.g. "$25,867M" vs "$25.867 billion")
- The AI answer is correct if it conveys the same meaning, conclusion, or rationale
- The AI answer is also correct if it is a superset (contains the golden answer plus extra correct info)
- The AI answer is WRONG if it gives a different number, wrong direction, or missing the key fact

Reply with exactly one word: TRUE or FALSE"""


def build_extra_kwargs(model: str, for_judge: bool = False) -> dict:
    """Build LiteLLM extra kwargs for Gemini models.
    Pro requires thinking (can't disable). Lite/Flash disable it to save cost."""
    if "gemini" not in model.lower():
        return {}
    if "pro" in model.lower():
        return {}  # Pro mandates thinking — don't send thinkingBudget at all
    return {"extra_body": {"generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}}}


def judge_answer(question: str, golden: str, generated: str, model: str) -> bool:
    if not generated or not generated.strip():
        return False
    prompt = JUDGE_PROMPT.format(question=question, golden=golden.strip(), generated=generated.strip())
    try:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
            **build_extra_kwargs(model, for_judge=True),
        )
        verdict = (resp.choices[0].message.content or "").strip().lower()
        return "true" in verdict
    except Exception as e:
        print(f"    [judge error] {e}")
        return False


def run_eval(override_model: str = None):
    cfg = get_config_manager().get_all()
    svc = PageIndexService(cfg)

    # Override model for eval if requested
    eval_model = override_model or svc.model
    if override_model:
        svc.model = override_model
        print(f"Model overridden: {eval_model}")
    else:
        print(f"Model: {eval_model}")

    extra_kwargs = build_extra_kwargs(eval_model)

    with open("eval/ground_truth.yaml") as f:
        gt = yaml.safe_load(f)

    results = []
    for q in gt["questions"]:
        qid       = q["id"]
        question  = q["question"]
        golden    = q["expected_answer"]
        keywords  = q.get("expected_contexts", [])

        print(f"\n[{qid}] {question[:72]}...")
        t0 = time.time()
        r  = svc.query(question)
        elapsed = time.time() - t0
        answer  = r.get("answer", "")

        # LLM judge
        passed = judge_answer(question, golden, answer, eval_model)

        # Also track keyword score for reference
        kw_hits = sum(1 for kw in keywords if kw.lower() in answer.lower())
        kw_pct  = kw_hits / len(keywords) * 100 if keywords else 0

        tools = [t["tool"] for t in r.get("tool_calls", [])]
        icon  = "✓" if passed else "✗"
        print(f"  {icon} LLM-judge={passed} | kw={kw_hits}/{len(keywords)} ({kw_pct:.0f}%) | {elapsed:.1f}s")
        print(f"  Answer: {answer[:200]}")

        results.append({
            "id": qid, "passed": passed, "kw_pct": kw_pct,
            "elapsed": elapsed, "tools": tools, "answer": answer,
        })

    print("\n" + "=" * 60)
    n_pass = sum(1 for r in results if r["passed"])
    avg_t  = sum(r["elapsed"] for r in results) / len(results)
    print(f"RESULT (LLM-as-judge): {n_pass}/{len(results)} = {n_pass/len(results)*100:.0f}%")
    print(f"Avg latency: {avg_t:.1f}s")
    print()
    for r in results:
        icon = "✓" if r["passed"] else "✗"
        print(f"  {icon} {r['id']:12s} kw={r['kw_pct']:3.0f}%  {r['elapsed']:.1f}s  tools={r['tools']}")

    return results


if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_eval(override_model=model_arg)
