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
import re
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

        Falls back to original order on any parse or API failure.
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
            f"Include all {len(chunks)} indices. Example: [2,0,1]. No prose, no markdown.\n\n"
            f"Passages:\n{candidates_text}\n\n"
            f"Ranked indices (JSON array only):"
        )

        try:
            os.environ["GEMINI_API_KEY"] = self.api_key
            resp = litellm.completion(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
                extra_body={"generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}},
            )
            raw = resp.choices[0].message.content or ""
            indices = _extract_int_array(raw)
            if indices is None:
                # One retry with a stricter prompt
                retry_prompt = (
                    f"Output ONLY a valid JSON integer array, nothing else. "
                    f"Example for 3 items: [2,0,1]\n\n"
                    f"Rank these {len(chunks)} passages (0-based indices) by relevance to: {query[:200]}"
                )
                resp2 = litellm.completion(
                    model=_MODEL,
                    messages=[{"role": "user", "content": retry_prompt}],
                    temperature=0,
                    max_tokens=256,
                    extra_body={"generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}},
                )
                raw2 = resp2.choices[0].message.content or ""
                indices = _extract_int_array(raw2)
                if indices is None:
                    raise ValueError(f"reranker: could not parse int array from: {raw[:120]!r}")

            seen = set()
            reranked = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(chunks) and idx not in seen:
                    reranked.append(chunks[idx])
                    seen.add(idx)

            # Append anything the model omitted to preserve completeness
            for i, chunk in enumerate(chunks):
                if i not in seen:
                    reranked.append(chunk)

            logger.debug("Reranker: pool=%d → top=%d, new_order=%s", len(chunks), top_k, indices[:top_k])
            return reranked[:top_k]

        except Exception as e:
            logger.warning("Reranker failed, using retrieval order: %s", e)
            return chunks[:top_k]


def _extract_int_array(text: str):
    """
    Extract the first JSON integer array from text, tolerating:
    - markdown code fences (```json ... ```)
    - surrounding prose
    - multiline whitespace inside the array

    Returns a list[int] or None if nothing valid found.
    """
    # Strip markdown fences first
    text = _strip_fence(text)
    # Find the outermost [...] bracket span (greedy — captures the whole array)
    m = re.search(r'\[([^\[\]]*)\]', text, re.DOTALL)
    if not m:
        return None
    inner = re.sub(r'\s+', ' ', m.group(0)).strip()
    try:
        parsed = json.loads(inner)
        if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _strip_fence(text: str) -> str:
    """Remove the outermost markdown code fence if present."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        # parts[1] is the content between the first pair of fences
        inner = parts[1] if len(parts) > 1 else text
        return inner.lstrip("json").strip()
    return text
