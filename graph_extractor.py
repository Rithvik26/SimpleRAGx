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

# ── Singleton GLiNER model (loaded once, reused across all chunks) ────────────
_gliner_model = None
_gliner_lock = threading.Lock()  # PyTorch inter-op thread pool is shared; serialize inference

def _get_gliner_model():
    global _gliner_model
    if _gliner_model is None:
        from gliner import GLiNER
        logger.info("Loading GLiNER small-v2.1 (first call only — ~7s)…")
        t0 = time.time()
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
        logger.info(f"GLiNER loaded in {round(time.time() - t0, 1)}s")
    return _gliner_model


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

    prompt = (
        "Extract relationships between the listed entities from the text.\n"
        "Output ONLY a valid JSON array. No explanation, no markdown fences.\n\n"
        f"ENTITIES: {json.dumps(entity_names)}\n\n"
        f"TEXT:\n{text[:1500]}\n\n"
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
            model="gemini/gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
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
        """Extract from all chunks in parallel (GLiNER local + LLM concurrent)."""
        total = len(chunks)
        logger.info(f"Starting hybrid extraction from {total} chunks ({max_workers} workers)")
        t_start = time.time()

        results: List[Optional[Dict]] = [None] * total
        completed = 0

        def _process(args):
            i, chunk = args
            chunk_id = f"chunk_{i}"
            return i, self.extract_entities_and_relationships(chunk, chunk_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process, (i, c)): i for i, c in enumerate(chunks)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    i, result = future.result()
                    results[i] = result
                except Exception as e:
                    logger.error(f"Error on chunk {futures[future]}: {e}")
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, f"Chunk {completed}/{total}")

        all_entities: List[Dict] = []
        all_relationships: List[Dict] = []
        for r in results:
            if r:
                all_entities.extend(r.get("entities", []))
                all_relationships.extend(r.get("relationships", []))

        elapsed = round(time.time() - t_start, 1)
        logger.info(
            f"Extraction complete: {len(all_entities)} entities, "
            f"{len(all_relationships)} relationships from {total} chunks in {elapsed}s"
        )
        return {
            "entities":         all_entities,
            "relationships":    all_relationships,
            "chunks_processed": total,
        }

    def get_extraction_stats(self) -> Dict[str, Any]:
        return {
            "backend":                "GLiNER-small-v2.1 + gemini-2.5-flash-lite",
            "api_key_configured":     bool(self.api_key),
            "max_entities_per_chunk": self.max_entities_per_chunk,
            "max_chunk_length":       self.max_chunk_length,
        }
