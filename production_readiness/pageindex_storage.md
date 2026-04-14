# PageIndex — Production Storage & Latency Guide

## How PageIndex Stores Data (Current)

Every indexed document creates two artefacts in the workspace directory:

```
{workspace}/
├── _meta.json                  ← lightweight registry of all indexed docs
└── {doc_id}.json               ← full tree + raw page text for one doc
    client_workspace/
    └── {doc_id}.json           ← same tree written by the PageIndex library (authoritative)
```

### `_meta.json` — doc registry
```json
{
  "b579510f-...": {
    "doc_id": "b579510f-...",
    "status": "ready",
    "path": "/path/to/document.pdf",
    "doc_description": "LLM-written summary of the document",
    "page_count": 5,
    "section_count": 5,
    "indexed_at": "2026-04-14T11:51:36Z"
  }
}
```

### `{doc_id}.json` — the tree + pages
Two keys in one file:
- **`structure`** — hierarchical TOC the LLM built during indexing. Each node has `title`, `start_page`, `end_page`, and a `summary` paragraph. This is what the agent reads first on every query to decide which page to fetch.
- **`pages`** — raw extracted text per page, stored as plain strings. Returned when the agent calls `get_page_content(pages="2")`.

---

## Problems with the Current Dev Setup

| Problem | Detail |
|---|---|
| `/tmp` workspace | Wiped on every OS restart — forces full re-index of all docs on cold start |
| Full JSON load per query | On every query the agent loads the entire `{doc_id}.json` (tree + all page text) even though it only reads 1–2 pages |
| Single-machine only | JSON files on local disk don't work across multiple app instances |
| No cache | Structure (TOC) is re-read from disk on every single query |

---

## Production Architecture

### 1. Persistent Storage — replace `/tmp`

Change `pageindex_workspace` in config from `/tmp/pi_bench_ws` to a persistent path.

**Single instance / small scale:**
```python
# config.py
"pageindex_workspace": os.path.expanduser("~/.simpleragx/pageindex_workspace")
```
Already the default in `pageindex_service.py` — just don't override it to `/tmp`.

**Multi-instance / cloud:**

Store the workspace files in object storage and sync on startup:

```
S3 / GCS bucket
└── pageindex/
    ├── _meta.json
    ├── {doc_id_1}.json
    └── {doc_id_2}.json
```

Or use a **Postgres `jsonb` column** — one row per doc, indexed on `doc_id`:

```sql
CREATE TABLE pageindex_docs (
    doc_id      TEXT PRIMARY KEY,
    meta        JSONB NOT NULL,          -- fields from _meta.json
    structure   JSONB NOT NULL,          -- TOC tree (small, ~5 KB)
    pages       JSONB NOT NULL,          -- raw page text (large, ~50 KB–2 MB)
    indexed_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX ON pageindex_docs USING GIN (structure);  -- fast tree lookup
```

Postgres is ideal because `structure` and `pages` can be queried/updated independently without loading the full document.

---

### 2. In-Memory Cache — eliminate per-query disk reads

The agent makes 2–3 tool calls per query:
1. `get_document_structure()` — reads the TOC tree
2. `get_page_content(pages="X")` — reads 1–2 pages of raw text

Both should be cached.

**Recommended: two-tier LRU cache**

```python
from functools import lru_cache

# Tier 1: structure cache — cache ALL doc structures on startup
# Structure is ~5 KB per doc — cheap to keep 1,000 docs in memory
@lru_cache(maxsize=1000)
def get_structure(doc_id: str) -> list:
    return load_structure_from_db(doc_id)

# Tier 2: page cache — LRU, only recently accessed pages
# Pages are ~50 KB each — keep last 200 pages across all docs
@lru_cache(maxsize=200)
def get_page(doc_id: str, page_num: int) -> str:
    return load_page_from_db(doc_id, page_num)
```

Or use **Redis** for shared cache across instances:
```python
import redis
r = redis.Redis()

def get_structure(doc_id):
    cached = r.get(f"pi:structure:{doc_id}")
    if cached:
        return json.loads(cached)
    structure = load_from_db(doc_id)
    r.setex(f"pi:structure:{doc_id}", 3600, json.dumps(structure))  # TTL 1h
    return structure
```

---

### 3. Pre-warming — zero cold-start latency

On app startup, load all document structures into the cache before the first query arrives:

```python
def warm_cache():
    meta = load_meta()  # _meta.json or DB query
    for doc_id in meta:
        get_structure(doc_id)   # populates LRU / Redis
    logger.info(f"PageIndex cache warmed: {len(meta)} docs")

# Call at startup
warm_cache()
```

Structure is small (~5 KB per doc). Pre-warming 1,000 docs = ~5 MB in memory — negligible. The first query after startup will be just as fast as the thousandth.

---

### 4. Split `structure` and `pages` on disk

Currently both live in one JSON file. For large documents (100+ pages) this means loading megabytes of page text just to answer a simple structure lookup.

**Better layout:**
```
pageindex/
├── _meta.json
└── {doc_id}/
    ├── structure.json       ← TOC tree only (~5 KB, always loaded)
    └── pages/
        ├── 1.txt
        ├── 2.txt
        └── ...              ← raw page text, fetched individually on demand
```

This makes `get_document_structure()` a 5 KB read and `get_page_content(pages="3")` a single small file read, regardless of document length.

---

## Summary: Recommended Stack by Scale

| Scale | Storage | Cache | Notes |
|---|---|---|---|
| Local / dev | `~/.simpleragx/pageindex_workspace/` (default) | None needed | Just don't use `/tmp` |
| Single-server prod | Local persistent disk, split structure/pages layout | `lru_cache` in-process | Cheap, zero extra infra |
| Multi-instance prod | Postgres `jsonb` (structure + pages columns) | Redis, TTL 1h | Shared state, cache survives deploys |
| High-scale / many docs | S3/GCS for pages, Postgres for structure + meta | Redis cluster | Pages served from object storage, structure from fast DB |

---

## Config Keys in SimpleRAGx

```python
# pageindex_service.py defaults
DEFAULT_WORKSPACE = str(Path.home() / ".simpleragx" / "pageindex_workspace")

# Override in your config dict:
config = {
    "pageindex_workspace": "/your/persistent/path",   # change this
    ...
}
```

The benchmark overrides this to `/tmp/pi_bench_ws` for a clean run every time — **never use `/tmp` in production.**
