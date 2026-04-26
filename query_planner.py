"""
Query planner for SimpleRAG — decomposition + HyDE + RRF merge.

Two techniques, composed per query complexity:

  HyDE (Hypothetical Document Embedding)
    Generate a short hypothetical answer passage, embed it alongside the
    real query. A plausible answer paragraph is closer in embedding space
    to real supporting chunks than the question form — improves recall for
    all query types with one extra Gemini call.

  Query decomposition
    For multi-hop / comparative queries: break into 2-4 focused sub-questions,
    retrieve independently per sub-question, merge with Reciprocal Rank Fusion.
    Agentic-RAG SoK (arXiv 2603.07379) shows 34% → 78% multi-hop improvement.

Decision logic:
  simple  (≤1 hop signal) → HyDE only
  complex (multi-hop)     → decompose + HyDE per sub-query

Everything is best-effort — failure at any step falls back to the original
single-query embedding, so this layer never breaks existing query paths.
"""

import hashlib
import json
import logging
import os
from typing import List, Dict, Any

import litellm

logger = logging.getLogger(__name__)
_MODEL = "gemini/gemini-2.5-flash"

# Signals that suggest multi-hop reasoning is needed
_MULTI_HOP_SIGNALS = [
    "and", "also", "as well as", "both", "compare", "versus", " vs ",
    "relationship between", "how does", "why does", "what caused",
    "list all", "enumerate", "which of", "across", "between", "each",
    "contrast", "similarities", "differences", "what are the",
]


class QueryPlanner:
    """
    Plans multi-embedding retrieval strategies for a user query.

    Usage:
        planner = QueryPlanner(gemini_api_key)
        plan = planner.plan(query)
        # plan["sub_queries"]  — list of query strings to embed
        # plan["hyde_docs"]    — list of hypothetical docs (one per sub_query)
        # plan["strategy"]     — "simple" | "hyde" | "decomposed"
    """

    def __init__(self, gemini_api_key: str):
        self.api_key = gemini_api_key
        if gemini_api_key:
            os.environ.setdefault("GEMINI_API_KEY", gemini_api_key)

    def plan(self, query: str) -> Dict[str, Any]:
        """Return a retrieval plan for the query."""
        if not self.api_key:
            return _simple_plan(query)

        is_complex = _is_complex(query)
        sub_queries = self._decompose(query) if is_complex else [query]
        hyde_docs = self._generate_hyde_docs(sub_queries)

        strategy = "decomposed" if len(sub_queries) > 1 else ("hyde" if any(hyde_docs) else "simple")
        logger.info(
            f"QueryPlanner [{strategy}]: {len(sub_queries)} sub-queries, "
            f"{sum(1 for d in hyde_docs if d)} hyde docs"
        )
        return {"strategy": strategy, "sub_queries": sub_queries, "hyde_docs": hyde_docs}

    def _decompose(self, query: str) -> List[str]:
        """Break complex query into focused sub-questions."""
        prompt = (
            "Break this complex question into 2-4 simpler sub-questions that together "
            "cover everything needed to answer it. Each sub-question should be independently answerable.\n"
            "Return ONLY a JSON array of question strings. No prose.\n\n"
            f"Question: {query}\n\nSub-questions (JSON array):"
        )
        try:
            raw = self._call(prompt, max_tokens=250)
            subs = json.loads(_strip_fence(raw))
            if isinstance(subs, list) and subs:
                # Original query always first so it's always retrieved against
                unique = [query] + [s for s in subs if isinstance(s, str) and s.strip() != query]
                return unique[:5]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
        return [query]

    def _generate_hyde_docs(self, queries: List[str]) -> List[str]:
        """
        Generate one hypothetical answer passage per sub-query.

        The passage is written as if it appears in a source document, not as a
        direct answer — this keeps its embedding close to real document chunks.
        """
        hyde_docs = []
        for q in queries:
            prompt = (
                "Write a short passage (2-4 sentences) that would appear in a source document "
                "and directly supports the answer to this question. "
                "Write as factual document text, not as a conversational answer.\n\n"
                f"Question: {q}\n\nPassage:"
            )
            try:
                doc = self._call(prompt, max_tokens=180)
                hyde_docs.append(doc.strip())
            except Exception as e:
                logger.warning(f"HyDE generation failed: {e}")
                hyde_docs.append("")
        return hyde_docs

    def _call(self, prompt: str, max_tokens: int) -> str:
        os.environ["GEMINI_API_KEY"] = self.api_key
        resp = litellm.completion(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


# ── RRF helpers ───────────────────────────────────────────────────────────────

def rrf_merge(result_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion across multiple ranked result lists.

    Deduplicates by text hash — the same chunk retrieved for multiple
    sub-queries merges into a single entry with accumulated score.
    Returns merged list sorted by descending fused score.
    """
    scores: Dict[str, float] = {}
    chunk_by_id: Dict[str, Dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            cid = _chunk_id(chunk)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunk_by_id:
                chunk_by_id[cid] = chunk

    return sorted(chunk_by_id.values(), key=lambda c: scores[_chunk_id(c)], reverse=True)


def _chunk_id(chunk: Dict[str, Any]) -> str:
    return hashlib.md5(chunk.get("text", "")[:200].encode()).hexdigest()


def _is_complex(query: str) -> bool:
    q = query.lower()
    score = sum(1 for sig in _MULTI_HOP_SIGNALS if sig in q)
    return score >= 2 or len(query.split()) > 18


def _simple_plan(query: str) -> Dict[str, Any]:
    return {"strategy": "simple", "sub_queries": [query], "hyde_docs": []}


def _strip_fence(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        inner = parts[1] if len(parts) > 1 else text
        return inner.lstrip("json").strip()
    return text
