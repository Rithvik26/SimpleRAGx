"""
Graph extractor — hybrid GLiNER + LiteLLM approach.

Entity extraction    : GLiNER small-v2.1 (local 50M-param model, ~0.2s/chunk, zero API cost)
Relationship extraction: gemini-2.5-flash-lite via LiteLLM (~2-4s/chunk, cheap + non-thinking)

~15-20x faster than the original Gemini-2.5-Flash-per-chunk approach.
Real-world reference: production Graph RAG systems (LightRAG, Microsoft GraphRAG) all recommend
a lightweight NER stage (GLiNER / spaCy) followed by a focused LLM pass for relationships only.
"""

import concurrent.futures
import json
import logging
import re
import threading
import time
import os
from typing import Dict, Any, List, Optional

import litellm
from entity_canonicalizer import canonical_id

logger = logging.getLogger(__name__)

# ── GLiNER label set → internal entity types ─────────────────────────────────
_GLINER_LABELS = [
    "person",
    "organization",
    "product",
    "technology",
    "financial amount",
    "date",
    "location",
    "event",
]

_LABEL_TO_TYPE = {
    "person":           "PERSON",
    "organization":     "ORGANIZATION",
    "product":          "PRODUCT",
    "technology":       "TECHNOLOGY",
    "financial amount": "NUMBER",
    "date":             "DATE",
    "location":         "LOCATION",
    "event":            "EVENT",
}

# ── GLiNER backend — local (default) or Modal (set MODAL_GLINER=1) ───────────
_gliner_model = None
_gliner_lock = threading.Lock()  # local only: PyTorch inter-op thread pool is shared

# Modal service handle — resolved once at first use
_modal_gliner: object = None
_modal_gliner_resolved = False

_USE_MODAL = os.environ.get("MODAL_GLINER", "").strip() == "1"


def _get_gliner_model():
    global _gliner_model
    if _gliner_model is None:
        from gliner import GLiNER
        logger.info("Loading GLiNER small-v2.1 (first call only — ~7s)…")
        t0 = time.time()
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
        logger.info(f"GLiNER loaded in {round(time.time() - t0, 1)}s")
    return _gliner_model


def _get_modal_gliner():
    """Resolve the Modal GLiNER service handle (once per process)."""
    global _modal_gliner, _modal_gliner_resolved
    if not _modal_gliner_resolved:
        try:
            import modal as _modal
            _modal_gliner = _modal.Cls.from_name("simplerag-gliner", "GLiNERService")
            logger.info("Modal GLiNER service resolved — using cloud inference")
        except Exception as e:
            logger.warning("Modal GLiNER unavailable, falling back to local: %s", e)
            _modal_gliner = None
        _modal_gliner_resolved = True
    return _modal_gliner


def _gliner_batch(texts: list) -> list:
    """
    Run GLiNER on a list of texts.

    Uses Modal (parallel containers, no lock) when MODAL_GLINER=1 and the
    service is deployed. Falls back to local batch inference automatically.
    Returns list[list[span_dict]] — one inner list per input text.
    """
    if _USE_MODAL:
        svc = _get_modal_gliner()
        if svc is not None:
            try:
                return svc().batch_ner.remote(texts, _GLINER_LABELS, threshold=0.5)
            except Exception as e:
                logger.warning("Modal call failed, falling back to local: %s", e)

    # Local fallback — one lock acquisition, one forward pass
    model = _get_gliner_model()
    with _gliner_lock:
        return model.inference(texts, labels=_GLINER_LABELS, threshold=0.5, batch_size=16)


# ── Relationship extraction via cheap LLM ────────────────────────────────────
def _extract_relationships_llm(
    text: str,
    entities: List[Dict],
    api_key: str,
) -> List[Dict]:
    """Use gemini-2.5-flash-lite to extract relationships between already-known entities.

    The LLM only needs to answer ONE focused question: given these entity names and this
    text, which entities are related and how?  No entity spotting required → prompt is
    tiny, response is fast, non-thinking model is sufficient.
    """
    if len(entities) < 2:
        return []

    entity_names = [e["name"] for e in entities]

    # Cap entity list to avoid overwhelming the LLM; top 20 is sufficient
    entity_names = entity_names[:20]

    prompt = (
        "Extract relationships between the listed entities from the text.\n"
        "Output ONLY a valid JSON array. No explanation, no markdown fences.\n\n"
        f"ENTITIES: {json.dumps(entity_names)}\n\n"
        f"TEXT:\n{text[:3000]}\n\n"
        "Each item must be: "
        "{\"source\": \"EntityA\", \"relationship\": \"short_verb\", "
        "\"target\": \"EntityB\", \"description\": \"one sentence\"}\n"
        "Rules:\n"
        "- source AND target must both be in the ENTITIES list above\n"
        "- source != target\n"
        "- relationship is a short verb or verb phrase (founded, leads, partnered_with, raised_from, etc.)\n"
        "- return [] if no clear relationships exist\n\n"
        "JSON array:"
    )

    try:
        os.environ.setdefault("GEMINI_API_KEY", api_key)
        resp = litellm.completion(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2048,
            extra_body={"generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}},
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Extract the JSON array even if there's surrounding text
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []

        rels = json.loads(raw[start:end])
        if not isinstance(rels, list):
            return []

        # Validate: both endpoints must be in entity_names, source != target
        entity_set = set(entity_names)
        valid = []
        for r in rels:
            if (
                isinstance(r, dict)
                and r.get("source") in entity_set
                and r.get("target") in entity_set
                and r.get("source") != r.get("target")
                and isinstance(r.get("relationship"), str)
                and r["relationship"].strip()
            ):
                valid.append({
                    "source":       r["source"],
                    "relationship": r["relationship"].strip(),
                    "target":       r["target"],
                    "description":  r.get("description", ""),
                })
        return valid

    except Exception as e:
        logger.warning(f"Relationship LLM call failed: {e}")
        return []


# ── Helper: find the sentence(s) containing an entity span ───────────────────
def _entity_description(entity_text: str, full_text: str) -> str:
    """Return the first sentence containing entity_text as the description.
    Falls back to the first 150 chars of the chunk if not found.
    """
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    for s in sentences:
        if entity_text in s:
            return s.strip()[:200]
    return full_text[:150].strip()


# ── Main extractor class ──────────────────────────────────────────────────────
class GraphExtractor:
    """Hybrid entity+relationship extractor.

    Stage 1 — GLiNER (local, ~0.2 s/chunk): spots named entities with zero API cost.
    Stage 2 — gemini-2.5-flash-lite (~2-4 s/chunk): extracts relationships between the
              spotted entities using a tight, focused prompt.
    """

    def __init__(self, config: Dict[str, Any]):
        self.api_key              = config["gemini_api_key"]
        self.max_entities_per_chunk = config.get("max_entities_per_chunk", 20)
        self.max_chunk_length     = config.get("max_chunk_length_for_graph", 2000)

        if not self.api_key:
            raise ValueError("Gemini API key is required (used for relationship extraction)")

        # Warm up the model on init so the first chunk isn't slow
        try:
            _get_gliner_model()
        except Exception as e:
            logger.warning(f"GLiNER model pre-load failed (will retry on first chunk): {e}")

        logger.info("GraphExtractor (GLiNER + flash-lite) ready")

    # ── Public interface (unchanged from original) ────────────────────────────

    def extract_entities_and_relationships(
        self, text: str, chunk_id: str = None
    ) -> Dict[str, Any]:
        """Extract entities (GLiNER) and relationships (LLM) from one text chunk."""
        if not text or not text.strip():
            logger.warning("Empty text — skipping extraction")
            return {"entities": [], "relationships": []}

        cleaned = re.sub(r"\s+", " ", text.strip())

        # ── Stage 1: GLiNER entity extraction (serialized — shared PyTorch thread pool) ──
        try:
            model = _get_gliner_model()
            t0 = time.time()
            with _gliner_lock:
                raw = model.predict_entities(cleaned, _GLINER_LABELS, threshold=0.5)
            logger.debug(f"GLiNER: {len(raw)} spans in {round(time.time()-t0, 3)}s")
        except Exception as e:
            logger.error(f"GLiNER extraction failed: {e}")
            raw = []

        # Deduplicate by name; use the highest-confidence occurrence
        seen: Dict[str, Dict] = {}
        for span in raw:
            name  = span["text"].strip()
            score = span["score"]
            if not name:
                continue
            if name not in seen or score > seen[name]["_score"]:
                seen[name] = {
                    "name":         name,
                    "type":         _LABEL_TO_TYPE.get(span["label"], "CONCEPT"),
                    "description":  _entity_description(name, cleaned),
                    "source_chunks": [chunk_id] if chunk_id else [],
                    "source_texts":  [text[:200]],
                    "merged_from":   1,
                    "_score":        score,  # internal, stripped later
                }

        # Drop internal key, attach stable ID, and cap
        entities = []
        for e in list(seen.values())[: self.max_entities_per_chunk]:
            e.pop("_score", None)
            e["id"] = canonical_id(e["name"], e["type"])
            e.setdefault("aliases", [e["name"]])
            entities.append(e)

        # ── Stage 2: LLM relationship extraction ───────────────────────────
        relationships = []
        if len(entities) >= 2:
            try:
                t0 = time.time()
                relationships = _extract_relationships_llm(cleaned, entities, self.api_key)
                logger.debug(
                    f"Relationships: {len(relationships)} in {round(time.time()-t0, 2)}s"
                )
            except Exception as e:
                logger.error(f"Relationship extraction failed: {e}")

        # Tag with chunk metadata
        for r in relationships:
            r.setdefault("source_chunk", chunk_id or "")
            r.setdefault("source_text",  text[:200])

        logger.info(
            f"Chunk {chunk_id}: {len(entities)} entities, {len(relationships)} relationships"
        )
        return {"entities": entities, "relationships": relationships}

    def extract_from_multiple_chunks(
        self, chunks: List[str], progress_callback=None, max_workers: int = 8
    ) -> Dict[str, Any]:
        """Extract from all chunks with batched LLM call.

        GLiNER runs sequentially (global lock makes parallelism pointless).
        All chunks' entities are merged, then ONE LLM relationship call covers
        the whole doc — reduces LLM calls from N_chunks to 1 per doc.
        """
        total = len(chunks)
        logger.info(f"Starting batched extraction from {total} chunks")
        t_start = time.time()

        # Stage 1: GLiNER batch inference — one forward pass for all chunks
        valid_indices = [i for i, c in enumerate(chunks) if c and c.strip()]
        cleaned_texts = [re.sub(r"\s+", " ", chunks[i].strip()) for i in valid_indices]

        all_entities: List[Dict] = []
        seen_ids: set = set()
        if cleaned_texts:
            try:
                batch_results = _gliner_batch(cleaned_texts)
            except Exception as e:
                logger.error(f"GLiNER batch failed: {e}")
                batch_results = [[] for _ in cleaned_texts]

            for result_idx, (chunk_idx, raw) in enumerate(zip(valid_indices, batch_results)):
                chunk = chunks[chunk_idx]
                cleaned = cleaned_texts[result_idx]
                for span in raw:
                    name = span["text"].strip()
                    if not name:
                        continue
                    etype = _LABEL_TO_TYPE.get(span["label"], "CONCEPT")
                    eid = canonical_id(name, etype)
                    if eid in seen_ids:
                        continue
                    seen_ids.add(eid)
                    all_entities.append({
                        "id":            eid,
                        "name":          name,
                        "type":          etype,
                        "description":   _entity_description(name, cleaned),
                        "source_chunks": [f"chunk_{chunk_idx}"],
                        "source_texts":  [chunk[:200]],
                        "merged_from":   1,
                        "aliases":       [name],
                    })

        if progress_callback:
            progress_callback(total, total, f"GLiNER batch done ({total} chunks)")

        all_entities = all_entities[: self.max_entities_per_chunk * total]

        # Stage 2: ONE batched LLM call for relationships across all chunks
        combined_text = "\n\n".join(
            c.strip()[:self.max_chunk_length] for c in chunks if c.strip()
        )[:4500]  # Gemini Flash has 1M context; 4500 chars is plenty for news articles

        relationships: List[Dict] = []
        if len(all_entities) >= 2:
            try:
                t0 = time.time()
                relationships = _extract_relationships_llm(combined_text, all_entities, self.api_key)
                logger.debug(f"Batched relationships: {len(relationships)} in {round(time.time()-t0,2)}s")
            except Exception as e:
                logger.error(f"Batched relationship extraction failed: {e}")

        elapsed = round(time.time() - t_start, 1)
        logger.info(
            f"Extraction complete: {len(all_entities)} entities, "
            f"{len(relationships)} relationships from {total} chunks in {elapsed}s"
        )
        return {
            "entities":         all_entities,
            "relationships":    relationships,
            "chunks_processed": total,
        }

    def get_extraction_stats(self) -> Dict[str, Any]:
        return {
            "backend":                "GLiNER-small-v2.1 + gemini-2.5-flash-lite",
            "api_key_configured":     bool(self.api_key),
            "max_entities_per_chunk": self.max_entities_per_chunk,
            "max_chunk_length":       self.max_chunk_length,
        }
