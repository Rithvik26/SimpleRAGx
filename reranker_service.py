"""
Reranker service for SimpleRAG.

Single Gemini call ranks N retrieved chunks by true query-document relevance,
replacing noisy cosine-score ordering. Zero new dependencies — uses existing LiteLLM.

Only activates for non-trivial result sets (>= 3 chunks); falls back to
original retrieval order on any failure.
"""

import json
import logging
import os
from typing import List, Dict, Any

import litellm

logger = logging.getLogger(__name__)
_MODEL = "gemini/gemini-2.5-flash"
_CHUNK_PREVIEW = 350  # chars sent to reranker per chunk — enough signal, minimal tokens


class RerankerService:
    """
    Reranks retrieved chunks by relevance using Gemini as a cross-attention scorer.

    One batched prompt replaces per-chunk API calls. For top_k=5 from a pool of 20,
    the prompt is ~3-4k tokens — fast and cheap.
    """

    def __init__(self, gemini_api_key: str):
        self.api_key = gemini_api_key
        if gemini_api_key:
            os.environ.setdefault("GEMINI_API_KEY", gemini_api_key)

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Re-rank chunks by relevance to query, return top_k most relevant.

        Args:
            query: the user's question
            chunks: retrieved chunks from vector search (may be larger than top_k)
            top_k: how many to return after reranking

        Returns:
            Reranked list of up to top_k chunks. Falls back to original order on failure.
        """
        if not chunks:
            return []
        if len(chunks) <= 1 or not self.api_key:
            return chunks[:top_k]

        candidates_text = "\n\n".join(
            f"[{i}] {chunk.get('text', '')[:_CHUNK_PREVIEW]}"
            for i, chunk in enumerate(chunks)
        )

        prompt = (
            f"Query: {query}\n\n"
            f"Rank these {len(chunks)} passages by relevance to the query. "
            f"Return ONLY a JSON array of integers (0-based indices) ordered most to least relevant. "
            f"Include all {len(chunks)} indices. No prose.\n\n"
            f"Passages:\n{candidates_text}\n\n"
            f"Ranked indices:"
        )

        try:
            os.environ["GEMINI_API_KEY"] = self.api_key
            resp = litellm.completion(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150,
            )
            raw = resp.choices[0].message.content.strip()
            raw = _strip_fence(raw)
            indices = json.loads(raw)

            if not isinstance(indices, list):
                raise ValueError("not a list")

            seen = set()
            reranked = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
                    reranked.append(chunks[idx])
                    seen.add(idx)

            # Safety: append anything the model omitted
            for i, chunk in enumerate(chunks):
                if i not in seen:
                    reranked.append(chunk)

            logger.debug(f"Reranker: pool={len(chunks)} → top={top_k}, new_order={indices[:top_k]}")
            return reranked[:top_k]

        except Exception as e:
            logger.warning(f"Reranker failed, using retrieval order: {e}")
            return chunks[:top_k]


def _strip_fence(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        inner = parts[1] if len(parts) > 1 else text
        return inner.lstrip("json").strip()
    return text
