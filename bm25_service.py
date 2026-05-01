"""
BM25 sparse encoder using fastembed.

Produces sparse vectors for Qdrant hybrid search.
Fully deterministic — no LLM involved.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

_model = None  # module-level singleton — expensive to load, share across calls


def _get_model():
    global _model
    if _model is None:
        from fastembed.sparse.bm25 import Bm25
        logger.info("Loading BM25 sparse model (first call only)…")
        _model = Bm25("Qdrant/bm25")
        logger.info("BM25 model ready")
    return _model


def encode_documents(texts: List[str]):
    """Return list of SparseVector for a batch of document texts."""
    from qdrant_client.models import SparseVector
    model = _get_model()
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in model.passage_embed(texts)
    ]


def encode_query(text: str):
    """Return SparseVector for a single query string."""
    from qdrant_client.models import SparseVector
    model = _get_model()
    return next(
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in model.query_embed(text)
    )
