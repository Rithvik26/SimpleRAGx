"""
Stable canonical IDs for entities.

canonical_id("TechCorp Inc", "ORGANIZATION") -> "organization::techcorp"
canonical_id("Alice Chen",   "PERSON")       -> "person::alice_chen"

The ID is stable across re-ingests and case/punctuation variations.
Aliases (alternate surface forms) should be stored separately and queried
via `name IN aliases` rather than by ID.
"""

import re
import unicodedata


def canonical_id(name: str, entity_type: str) -> str:
    """Return a stable slug for (name, type) suitable for MERGE keys."""
    n = unicodedata.normalize("NFKC", name).lower()
    n = re.sub(r"[^\w\s-]", "", n)
    n = re.sub(r"[\s-]+", "_", n.strip())
    n = n.strip("_")
    return f"{entity_type.lower()}::{n}"
