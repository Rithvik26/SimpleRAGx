"""
LLM-powered metadata extractor for SimpleRAG domain configurations.

At ingest time, runs one Gemini call per document to extract structured domain metadata
(sector, stage, founders, etc.) that gets stored in every chunk's Qdrant payload.
This enables metadata pre-filtering at query time.
"""

import json
import logging
import os

import litellm

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini/gemini-2.5-flash"
_MAX_TEXT_CHARS = 4000  # use first 4k chars of document for extraction


class MetadataExtractor:
    """Extracts domain-specific structured metadata from document text using an LLM."""

    def __init__(self, gemini_api_key: str, domain_config: dict):
        self.api_key = gemini_api_key
        self.domain_config = domain_config
        self.prompt_template = domain_config.get("extraction_prompt", "")
        if gemini_api_key:
            os.environ.setdefault("GEMINI_API_KEY", gemini_api_key)

    def extract(self, text: str, filename: str = "") -> dict:
        """
        Extract domain metadata from document text.

        Returns a flat dict of field→value pairs ready to merge into chunk metadata.
        Returns empty dict on failure (non-fatal — extraction is best-effort).
        """
        if not self.api_key or not self.prompt_template or not text:
            return {}

        sample = text[:_MAX_TEXT_CHARS].strip()
        prompt = self.prompt_template.format(text=sample)

        try:
            os.environ["GEMINI_API_KEY"] = self.api_key
            resp = litellm.completion(
                model=GEMINI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=600,
            )
            raw = resp.choices[0].message.content.strip()
            raw = _strip_code_fence(raw)
            parsed = json.loads(raw)
            # Drop null/empty values — don't pollute Qdrant payload with noise
            return {k: v for k, v in parsed.items() if v is not None and v != "" and v != []}
        except Exception as e:
            logger.warning(f"Metadata extraction failed for '{filename}': {e}")
            return {}


def _strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` fences that LLMs sometimes add."""
    if text.startswith("```"):
        lines = text.split("\n")
        # drop first line (```json or ```) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        return "\n".join(inner)
    return text
