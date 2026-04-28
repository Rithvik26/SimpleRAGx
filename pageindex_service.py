"""
PageIndex Service — Vectorless, Reasoning-based RAG
Wraps PageIndexClient with:
  - LiteLLM-based agentic retrieval loop (no OpenAI Agents SDK required)
  - Atomic workspace writes (no corrupt JSON on killed threads)
  - Exponential-backoff retry around the slow page_index() call
  - Hard cap on pages-per-tool-call to prevent context overflow
  - Progress tracking compatible with SimpleRAGx's ProgressTracker
"""

import os
import re
import json
import time
import math
import uuid
import base64
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

# Maximum pages a single get_page_content() call may return.
# At ~500 words/page × 1.3 tokens/word ≈ 650 tokens/page → 8 pages ≈ 5 200 tokens.
# That keeps synthesis well within a 32 k context window even with system + history.
MAX_PAGES_PER_CALL = 8

# Rounds of tool-calling the agent may make before we force a final answer.
DEFAULT_MAX_TOOL_ROUNDS = 6

# Workspace lives here unless overridden in config.
DEFAULT_WORKSPACE = str(Path.home() / ".simpleragx" / "pageindex_workspace")

# Status values written into _meta.json so crashed indexing can be detected.
STATUS_INDEXING = "indexing"
STATUS_READY    = "ready"
META_INDEX      = "_meta.json"

# LiteLLM model names — chosen based on what API keys are present.
# Correct Anthropic model id for LiteLLM routing:
CLAUDE_MODEL  = "anthropic/claude-3-5-sonnet-20241022"
GEMINI_MODEL  = "gemini/gemini-2.5-flash"
OPENAI_MODEL  = "gpt-4o-2024-11-20"          # fallback if user supplies OPENAI_API_KEY

SYSTEM_PROMPT = """You are a document QA assistant. Answer questions using the tools below.

RULES:
1. Call get_document_structure() first to see section titles and page ranges.
2. Call get_page_content(pages="X-Y") to fetch the actual text — tight range, max {max_pages} pages.
3. Write your answer from the fetched text only. End with "Sources: pages X-Y".

Do NOT answer after only calling get_document_structure. Always fetch page content first.
""".format(max_pages=MAX_PAGES_PER_CALL)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_page_range(pages_str: str) -> List[int]:
    """Parse '5-7', '3,8', or '12' into a sorted list of page numbers."""
    nums: List[int] = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a.strip()), int(b.strip())
            # Handle reversed ranges like "54-4" → treat as "4-54" capped later
            if lo > hi:
                lo, hi = hi, lo
            nums.extend(range(lo, hi + 1))
        elif part.isdigit():
            nums.append(int(part))
    return sorted(set(nums))


def _cap_page_range(pages_str: str) -> str:
    """Enforce MAX_PAGES_PER_CALL.  Returns the (possibly trimmed) pages string."""
    try:
        nums = _parse_page_range(pages_str)
    except Exception:
        return pages_str   # pass through; let retrieve.py raise the error
    if len(nums) > MAX_PAGES_PER_CALL:
        logger.warning(
            "Agent requested %d pages (%s); trimming to first %d.",
            len(nums), pages_str, MAX_PAGES_PER_CALL,
        )
        nums = nums[:MAX_PAGES_PER_CALL]
    if not nums:
        return pages_str
    # Reconstruct a compact range/list string
    return ",".join(str(n) for n in nums)


def _atomic_write_json(path: Path, data: dict):
    """Write JSON to a temp file in the same directory, then atomically rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(path))        # atomic on POSIX + Windows ≥ Vista
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_json_safe(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


# ─── PageIndexService ─────────────────────────────────────────────────────────

class PageIndexService:
    """
    Thin orchestration layer around PageIndexClient.

    Key design choices vs. the gotcha list:
    - Model naming: uses anthropic/claude-3-5-sonnet-20241022 via LiteLLM (not the
      raw API model id).
    - Atomic writes: every JSON file is written via _atomic_write_json().
    - Corrupted-index cleanup: status="indexing" is written before indexing starts;
      if the process dies the entry stays "indexing" so the next is_ready() check
      returns False and the UI allows a retry.
    - Page cap: get_page_content calls go through _cap_page_range() before hitting
      the retrieve module.
    - Rate-limit / retry: page_index() (the slow tree-building call) is wrapped in
      _retry_with_backoff() with up to 5 attempts and 2^attempt second sleep.
    """

    def __init__(self, config: dict):
        self.config = config
        self.workspace = Path(
            config.get("pageindex_workspace") or DEFAULT_WORKSPACE
        ).expanduser()
        self.max_tool_rounds: int = int(
            config.get("pageindex_max_tool_rounds", DEFAULT_MAX_TOOL_ROUNDS)
        )

        # Resolve the LLM model (LiteLLM path)
        self.model: str = self._resolve_model()
        self._client = None          # lazy-loaded PageIndexClient

        # In-memory document store (mirrors workspace)
        self._documents: Dict[str, dict] = {}
        self._available: bool = False

        try:
            self._check_deps()
            self.workspace.mkdir(parents=True, exist_ok=True)
            self._load_workspace()
            self._available = True
            logger.info(
                "PageIndexService ready. model=%s workspace=%s docs=%d",
                self.model, self.workspace, len(self._documents),
            )
        except Exception as e:
            logger.warning("PageIndexService unavailable: %s", e)

    # ── Public API ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._available and bool(self.model)

    def index_document(
        self,
        file_path: str,
        progress_tracker=None,
    ) -> Dict[str, Any]:
        """
        Build a PageIndex tree for *file_path*.

        Returns a result dict with keys: success, doc_id, doc_name,
        page_count, section_count, time_elapsed, error.
        """
        if not self._available:
            return {"success": False, "error": "PageIndex dependencies not available"}

        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.exists(file_path):
            return {"success": False, "error": f"File not found: {file_path}"}

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".pdf", ".md", ".markdown"):
            return {"success": False, "error": f"Unsupported format: {ext}. Use PDF or Markdown."}

        doc_id   = str(uuid.uuid4())
        start_ts = time.time()

        # ── Mark as in-progress so a crash leaves a recoverable state ──────
        meta_entry = {
            "doc_id":   doc_id,
            "status":   STATUS_INDEXING,
            "path":     file_path,
            "doc_name": os.path.basename(file_path),
            "doc_description": "",
            "page_count": 0,
            "section_count": 0,
            "type":     "pdf" if ext == ".pdf" else "md",
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._save_meta_entry(doc_id, meta_entry)

        if progress_tracker:
            progress_tracker.update(5, 100, status="starting",
                                    message="Preparing PageIndex tree builder…")

        try:
            client = self._get_client()

            if progress_tracker:
                progress_tracker.update(10, 100, status="indexing",
                                        message="Building hierarchical tree index (this takes a few minutes for long PDFs)…")

            # ── Retry wrapper around the expensive tree-build call ──────────
            def _do_index():
                return client.index(file_path)

            doc_id_from_client = self._retry_with_backoff(
                _do_index,
                max_attempts=5,
                progress_tracker=progress_tracker,
                progress_range=(10, 80),
            )

            if progress_tracker:
                progress_tracker.update(82, 100, status="saving",
                                        message="Saving tree structure to workspace…")

            # Pull the full document dict out of the client's in-memory store.
            # NOTE: PageIndex writes structure/pages to its workspace JSON but does NOT
            # always populate client.documents[id] in memory — always read from disk.
            raw_doc = client.documents.get(doc_id_from_client, {})

            # Always supplement from the client workspace file (authoritative on disk)
            if client.workspace:
                client_doc_file = client.workspace / f"{doc_id_from_client}.json"
                full_from_disk = _read_json_safe(client_doc_file)
                if full_from_disk:
                    # Merge: disk data wins for structure and pages
                    if full_from_disk.get("structure"):
                        raw_doc["structure"] = full_from_disk["structure"]
                    if full_from_disk.get("pages") and not raw_doc.get("pages"):
                        raw_doc["pages"] = full_from_disk["pages"]
                    for k in ("doc_name", "doc_description", "page_count"):
                        if full_from_disk.get(k) and not raw_doc.get(k):
                            raw_doc[k] = full_from_disk[k]

            # Count sections by flattening the tree
            section_count = self._count_nodes(raw_doc.get("structure", []))

            # ── Save per-document JSON atomically ───────────────────────────
            doc_file = self.workspace / f"{doc_id}.json"
            doc_payload = dict(raw_doc)
            doc_payload["id"] = doc_id                      # use OUR uuid, not client's

            _atomic_write_json(doc_file, doc_payload)

            # ── Update meta with final status ────────────────────────────────
            meta_entry.update({
                "status":        STATUS_READY,
                "doc_name":      raw_doc.get("doc_name", os.path.basename(file_path)),
                "doc_description": raw_doc.get("doc_description", ""),
                "page_count":    raw_doc.get("page_count") or raw_doc.get("line_count", 0),
                "section_count": section_count,
            })
            self._save_meta_entry(doc_id, meta_entry)

            # ── Keep in memory ───────────────────────────────────────────────
            self._documents[doc_id] = dict(meta_entry)
            self._documents[doc_id]["structure"] = raw_doc.get("structure", [])
            self._documents[doc_id]["pages"]     = doc_payload.get("pages", [])

            elapsed = round(time.time() - start_ts, 2)

            if progress_tracker:
                progress_tracker.update(100, 100, status="complete",
                                        message=f"PageIndex ready: {section_count} sections, "
                                                f"{meta_entry['page_count']} pages in {elapsed}s")

            logger.info("PageIndex indexed doc_id=%s sections=%d elapsed=%.1fs",
                        doc_id, section_count, elapsed)
            return {
                "success":       True,
                "doc_id":        doc_id,
                "doc_name":      meta_entry["doc_name"],
                "page_count":    meta_entry["page_count"],
                "section_count": section_count,
                "time_elapsed":  elapsed,
            }

        except Exception as e:
            # Mark as failed so UI lets the user retry
            meta_entry["status"] = "failed"
            meta_entry["error"]  = str(e)
            self._save_meta_entry(doc_id, meta_entry)

            elapsed = round(time.time() - start_ts, 2)
            logger.error("PageIndex indexing failed for %s: %s", file_path, e, exc_info=True)

            if progress_tracker:
                progress_tracker.update(100, 100, status="error",
                                        message=f"PageIndex indexing failed: {e}")
            return {"success": False, "error": str(e), "time_elapsed": elapsed}

    def query(
        self,
        question: str,
        doc_id: Optional[str] = None,
        progress_tracker=None,
    ) -> Dict[str, Any]:
        """
        Agentic tree-search query using LiteLLM function calling.

        Flow (3 LLM calls max):
          Round 0 → LLM calls get_document_structure() [trimmed, ~2K tokens]
          Round 1 → LLM calls get_page_content() with tight page range
          Round 2 → LLM writes final answer

        Summaries are stripped from the structure before sending to LLM — this is
        the key latency fix. The LLM only needs title+pages to navigate; full summaries
        were ~15K tokens and caused most of the slowness.
        """
        if not self._available:
            return self._error_result("PageIndex service not available")

        target_id = doc_id or self._route_doc(question)
        if not target_id:
            return self._error_result(
                "No PageIndex documents found. Please upload and index a document first."
            )

        doc = self._load_doc(target_id)
        if not doc:
            return self._error_result(f"Document {target_id} not found in workspace.")

        if progress_tracker:
            progress_tracker.update(5, 100, status="thinking",
                                    message="PageIndex agent starting tree search…")

        try:
            import litellm
        except ImportError:
            return self._error_result("litellm is not installed. Run: pip install litellm")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_document_structure",
                    "description": (
                        "Returns the document's table-of-contents tree. "
                        "Each node has: title, start_index (start page), end_index (end page). "
                        "Call this first to identify which pages to retrieve."
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_page_content",
                    "description": (
                        f"Retrieves raw text for specific pages. "
                        f"HARD LIMIT: max {MAX_PAGES_PER_CALL} pages per call. "
                        f"Use tight ranges: '5-7', '12', or '3,8'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pages": {
                                "type": "string",
                                "description": f"Pages to fetch (max {MAX_PAGES_PER_CALL}, e.g. '54-56').",
                            }
                        },
                        "required": ["pages"],
                    },
                },
            },
        ]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]

        citations:     List[Dict] = []
        tree_path:     List[str]  = []
        tool_log:      List[Dict] = []
        pages_fetched: List[str]  = []

        # Disable thinking for Gemini Lite/Flash — Pro mandates it and rejects budget=0
        extra_kwargs = {}
        if "gemini" in self.model.lower() and "pro" not in self.model.lower():
            extra_kwargs["extra_body"] = {
                "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}
            }

        final_answer       = ""
        fetched_page_content = False

        for round_num in range(self.max_tool_rounds + 1):

            if progress_tracker:
                pct = 10 + round_num * (70 // (self.max_tool_rounds + 1))
                progress_tracker.update(pct, 100, status="reasoning",
                                        message=f"Agent reasoning (round {round_num + 1})…")

            # Round 1 only: force a tool call so Gemini fetches pages instead of stopping.
            # Round 0 and round 2+ use auto so the final answer isn't forced into a tool.
            current_tool_choice = "required" if (round_num == 1 and not fetched_page_content) else "auto"

            try:
                resp = litellm.completion(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=current_tool_choice,
                    temperature=0,
                    **extra_kwargs,
                )
            except Exception as e:
                logger.error("LiteLLM completion error: %s", e)
                return self._error_result(f"LLM call failed: {e}")

            msg        = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                final_answer = msg.content or ""
                break

            # Serialize to plain dict — avoids Pydantic type mismatch on next call
            serialized_tcs = [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in tool_calls
            ]
            messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": serialized_tcs,
            })

            for stc in serialized_tcs:
                fn_name = stc["function"]["name"]
                if fn_name == "get_page_content":
                    fetched_page_content = True
                try:
                    fn_args = json.loads(stc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}

                tool_result = self._dispatch_tool(
                    fn_name, fn_args, doc, citations, tree_path, pages_fetched
                )

                preview = (tool_result[:300] + "…") if len(tool_result) > 300 else tool_result
                tool_log.append({"tool": fn_name, "args": fn_args, "result_preview": preview})

                messages.append({
                    "role":         "tool",
                    "tool_call_id": stc["id"],
                    "name":         fn_name,
                    "content":      tool_result,
                })

        else:
            # Exhausted rounds — force final answer from what was retrieved
            if progress_tracker:
                progress_tracker.update(88, 100, status="synthesising",
                                        message="Synthesising final answer…")
            messages.append({
                "role":    "user",
                "content": "Maximum tool calls reached. Give your best answer from the content retrieved so far.",
            })
            try:
                forced       = litellm.completion(model=self.model, messages=messages,
                                                  temperature=0, **extra_kwargs)
                final_answer = forced.choices[0].message.content or ""
            except Exception as e:
                final_answer = f"(Could not generate final answer: {e})"

        # Fallback: if answer is empty or explicitly says "not found", use keyword search
        # across ALL structure nodes (not just the slimmed top-2-levels sent to LLM).
        # This catches cases where Gemini navigates to wrong pages on large docs like Boeing.
        _not_found = ("not find", "cannot find", "not found", "does not contain",
                      "not available", "cannot answer", "not contain information",
                      "sorry", "does not have")
        _need_fallback = (
            not final_answer.strip()
            or any(p in final_answer.lower() for p in _not_found)
        )

        if _need_fallback:
            logger.info("Agentic answer empty/not-found — falling back to keyword structure search")
            from pageindex.retrieve import get_page_content as _gpc
            documents = {doc["id"]: doc}

            # Search ALL nodes (full structure, not slimmed) for keyword matches
            candidate_pages = self._search_structure(question, doc.get("structure", []), top_k=4)
            pages_str = _cap_page_range(",".join(str(p) for p in candidate_pages))
            page_text_json = _gpc(documents, doc["id"], pages_str)

            tool_log.append({"tool": "get_page_content[fallback]",
                             "args": {"pages": pages_str},
                             "result_preview": page_text_json[:200]})

            # Populate citations from fallback fetch
            try:
                page_data = json.loads(page_text_json)
                if isinstance(page_data, list):
                    for item in page_data:
                        if isinstance(item, dict) and "page" in item:
                            section = self._section_for_page(item["page"], doc.get("structure", []))
                            content = item.get("content", "")
                            citations.append({
                                "pages":   str(item["page"]),
                                "content": (content[:400] + "…") if len(content) > 400 else content,
                                "section": section,
                            })
            except Exception:
                pass

            try:
                synth = litellm.completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Answer only from the provided page excerpts. End with 'Sources: pages X'."},
                        {"role": "user",   "content": f"Question: {question}\n\nRetrieved pages:\n{page_text_json}"},
                    ],
                    temperature=0, **extra_kwargs,
                )
                fallback_answer = synth.choices[0].message.content or ""
                # Only use fallback if it's not also a "not found" answer
                if fallback_answer.strip() and not any(p in fallback_answer.lower() for p in _not_found):
                    final_answer = fallback_answer
            except Exception as e:
                logger.warning("Fallback synthesis failed: %s", e)

            # Vision fallback: if text extraction still can't answer (e.g. data locked in
            # charts/graphics), render candidate pages as images and call the vision LLM.
            # Gemini Flash Lite is multimodal — no extra cost beyond normal per-token pricing.
            _still_not_found = (
                not final_answer.strip()
                or any(p in final_answer.lower() for p in _not_found)
            )
            if _still_not_found:
                doc_path = doc.get("path", "")
                if doc_path and os.path.isfile(doc_path) and doc_path.lower().endswith(".pdf"):
                    # Prefer all pages already cited (agentic + text fallback) over re-searching —
                    # the agentic loop may have already landed on the right pages.
                    all_cited = sorted(set(
                        int(c["pages"]) for c in citations
                        if str(c.get("pages", "")).isdigit()
                    ))
                    vision_pages = (all_cited or candidate_pages)[:MAX_PAGES_PER_CALL]
                    logger.info(
                        "Text fallback insufficient — attempting vision fallback on pages %s",
                        vision_pages,
                    )
                    vision_answer = self._vision_fallback(
                        question, doc_path, vision_pages, extra_kwargs
                    )
                    if vision_answer and not any(p in vision_answer.lower() for p in _not_found):
                        final_answer = vision_answer
                        tool_log.append({
                            "tool":           "vision_fallback",
                            "args":           {"pages": candidate_pages},
                            "result_preview": vision_answer[:200],
                        })

        # Calculation pass: if question needs arithmetic and answer lacks numbers,
        # ask the LLM to compute explicitly from the already-retrieved page text.
        if self._needs_calculation(question) and not self._answer_has_numbers(final_answer):
            calc_pages = [c for c in citations if c.get("content")]
            if calc_pages:
                retrieved_text = "\n\n".join(
                    f"[Page {c['pages']}]\n{c['content']}" for c in calc_pages[:6]
                )
                try:
                    calc_resp = litellm.completion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content":
                                "You are a financial analyst. Using ONLY the provided page excerpts, "
                                "compute the exact answer. Show your arithmetic. "
                                "End with 'Sources: pages X'."},
                            {"role": "user", "content":
                                f"Question: {question}\n\nPage excerpts:\n{retrieved_text}"},
                        ],
                        temperature=0,
                        **extra_kwargs,
                    )
                    calc_answer = calc_resp.choices[0].message.content or ""
                    if calc_answer.strip() and not any(p in calc_answer.lower() for p in _not_found):
                        final_answer = calc_answer
                        tool_log.append({"tool": "calculation_pass", "args": {}, "result_preview": calc_answer[:200]})
                except Exception as e:
                    logger.warning("Calculation pass failed: %s", e)

        if progress_tracker:
            progress_tracker.update(100, 100, status="complete",
                                    message="PageIndex query complete")

        return {
            "success":   True,
            "answer":    final_answer.strip(),
            "citations": citations,
            "tree_path": tree_path,
            "tool_calls": tool_log,
            "doc_id":    target_id,
            "doc_name":  doc.get("doc_name", ""),
            "error":     None,
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return lightweight metadata for every document in the workspace."""
        meta = self._read_meta() or {}
        results = []
        for did, entry in meta.items():
            results.append({
                "doc_id":        did,
                "doc_name":      entry.get("doc_name", did),
                "doc_description": entry.get("doc_description", ""),
                "status":        entry.get("status", "unknown"),
                "page_count":    entry.get("page_count", 0),
                "section_count": entry.get("section_count", 0),
                "type":          entry.get("type", "pdf"),
                "indexed_at":    entry.get("indexed_at", ""),
            })
        # Most recently indexed first
        results.sort(key=lambda d: d["indexed_at"], reverse=True)
        return results

    def get_tree(self, doc_id: str) -> Optional[List[dict]]:
        """Return the tree structure for *doc_id* (used by the tree-viewer endpoint)."""
        doc = self._load_doc(doc_id)
        if not doc:
            return None
        return doc.get("structure", [])

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from workspace and in-memory store."""
        doc_file = self.workspace / f"{doc_id}.json"
        try:
            if doc_file.exists():
                doc_file.unlink()
            self._documents.pop(doc_id, None)
            meta = self._read_meta() or {}
            meta.pop(doc_id, None)
            _atomic_write_json(self.workspace / META_INDEX, meta)
            logger.info("Deleted PageIndex document %s", doc_id)
            return True
        except Exception as e:
            logger.error("Failed to delete doc %s: %s", doc_id, e)
            return False

    # ── Tool dispatcher ───────────────────────────────────────────────────────

    def _dispatch_tool(
        self,
        name:          str,
        args:          dict,
        doc:           dict,
        citations:     List[Dict],
        tree_path:     List[str],
        pages_fetched: List[str],
    ) -> str:
        from pageindex.retrieve import get_document_structure, get_page_content

        # Build a minimal documents dict in the format retrieve.py expects
        documents = {doc["id"]: doc}

        if name == "get_document_structure":
            result = get_document_structure(documents, doc["id"])
            # Keep only top 2 levels with title+pages — strips summaries and deep children.
            # Full AMD structure is 50K tokens; this brings it to ~1K tokens.
            # The LLM only needs section titles and page ranges to pick where to fetch.
            try:
                structure = json.loads(result)
                def _slim(nodes, depth=0):
                    out = []
                    for node in nodes:
                        t = node.get("title", "")
                        if t and t not in tree_path:
                            tree_path.append(t)
                        slim_node = {
                            "title":       t,
                            "start_index": node.get("start_index"),
                            "end_index":   node.get("end_index"),
                        }
                        # Keep summary for top-level only — enough context to pick section,
                        # without the ~50K token cost of full summaries at every level.
                        if depth == 0:
                            summary = node.get("summary", "")
                            if summary:
                                slim_node["summary"] = summary[:200]
                        if depth < 1 and node.get("nodes"):
                            slim_node["nodes"] = _slim(node["nodes"], depth + 1)
                        out.append(slim_node)
                    return out
                result = json.dumps(_slim(structure), ensure_ascii=False)
            except Exception:
                pass
            return result

        elif name == "get_page_content":
            raw_pages = args.get("pages", "")
            capped    = _cap_page_range(raw_pages)
            result    = get_page_content(documents, doc["id"], capped)

            # Parse result to build citations
            try:
                page_data = json.loads(result)
                if isinstance(page_data, list):
                    for item in page_data:
                        if isinstance(item, dict) and "page" in item:
                            # Find section title for this page
                            section = self._section_for_page(
                                item["page"], doc.get("structure", [])
                            )
                            citations.append({
                                "pages":   str(item["page"]),
                                "content": (item.get("content", "")[:400] + "…")
                                           if len(item.get("content", "")) > 400
                                           else item.get("content", ""),
                                "section": section,
                            })
            except Exception:
                pass

            pages_fetched.append(capped)
            return result

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    # ── Workspace helpers ─────────────────────────────────────────────────────

    def _load_workspace(self):
        """Populate self._documents from _meta.json (lightweight metadata only)."""
        meta = self._read_meta()
        if meta is None:
            meta = self._rebuild_meta()
        for doc_id, entry in meta.items():
            self._documents[doc_id] = dict(entry, id=doc_id)
        stale = [k for k, v in self._documents.items()
                 if v.get("status") == STATUS_INDEXING]
        if stale:
            logger.warning(
                "Found %d document(s) stuck in 'indexing' state — likely crashed: %s",
                len(stale), stale,
            )

    def _load_doc(self, doc_id: str) -> Optional[dict]:
        """Return a fully-populated document dict (lazy-loads JSON from disk)."""
        doc = self._documents.get(doc_id)
        if not doc:
            return None

        # If structure isn't loaded yet, pull from disk
        if "structure" not in doc:
            full = _read_json_safe(self.workspace / f"{doc_id}.json")
            if full:
                doc["structure"] = full.get("structure", [])
                if full.get("pages"):
                    doc["pages"] = full["pages"]
                # Ensure the id field is correct
                doc["id"] = doc_id
                self._documents[doc_id] = doc

        # Ensure 'id' key exists (retrieve.py expects documents[doc_id]['id'])
        doc.setdefault("id", doc_id)
        return doc

    def _read_meta(self) -> Optional[dict]:
        return _read_json_safe(self.workspace / META_INDEX)

    def _rebuild_meta(self) -> dict:
        """Scan individual doc JSON files and reconstruct _meta.json."""
        meta = {}
        for p in self.workspace.glob("*.json"):
            if p.name == META_INDEX:
                continue
            data = _read_json_safe(p)
            if isinstance(data, dict):
                doc_id = p.stem
                meta[doc_id] = {
                    "doc_id":      doc_id,
                    "status":      data.get("status", STATUS_READY),
                    "doc_name":    data.get("doc_name", p.stem),
                    "doc_description": data.get("doc_description", ""),
                    "page_count":  data.get("page_count") or data.get("line_count", 0),
                    "section_count": self._count_nodes(data.get("structure", [])),
                    "type":        data.get("type", "pdf"),
                    "indexed_at":  data.get("indexed_at", ""),
                    "path":        data.get("path", ""),
                }
        _atomic_write_json(self.workspace / META_INDEX, meta)
        return meta

    def _save_meta_entry(self, doc_id: str, entry: dict):
        meta = self._read_meta() or {}
        meta[doc_id] = entry
        _atomic_write_json(self.workspace / META_INDEX, meta)

    def _search_structure(self, question: str, structure: list, top_k: int = 3) -> List[int]:
        """
        Score every node in the structure tree by keyword overlap with the question.
        Returns a sorted list of page numbers (start..end, capped) for the top_k nodes.
        Zero LLM calls — pure local search.
        """
        q_tokens = set(re.findall(r'[a-z]{3,}', question.lower()))

        scored: List[tuple] = []

        def _score_node(node):
            text = " ".join([
                node.get("title", ""),
                node.get("summary", ""),
            ]).lower()
            node_tokens = set(re.findall(r'[a-z]{3,}', text))
            score = len(q_tokens & node_tokens)
            start = node.get("start_index", 0)
            end   = node.get("end_index",   start)
            if start > 0:
                scored.append((score, start, end, node.get("title", "")))
            for child in node.get("nodes", []):
                _score_node(child)

        for node in structure:
            _score_node(node)

        if not scored:
            return list(range(1, min(MAX_PAGES_PER_CALL + 1, 9)))

        scored.sort(key=lambda x: -x[0])
        pages: List[int] = []
        for score, start, end, title in scored[:top_k]:
            # Take up to 3 pages from each matching section
            for p in range(start, min(end + 1, start + 3)):
                if p not in pages:
                    pages.append(p)
            if len(pages) >= MAX_PAGES_PER_CALL:
                break

        return sorted(pages[:MAX_PAGES_PER_CALL])

    def _pick_ready_doc(self) -> Optional[str]:
        """Return the doc_id of the most recently indexed ready document."""
        meta = self._read_meta() or {}
        ready = [
            (k, v) for k, v in meta.items()
            if v.get("status") == STATUS_READY
        ]
        if not ready:
            return None
        ready.sort(key=lambda kv: kv[1].get("indexed_at", ""), reverse=True)
        return ready[0][0]

    def _route_doc(self, question: str) -> Optional[str]:
        """
        Pick the most relevant ready document for the given question.

        Scores each doc by counting overlapping tokens between the question
        and the doc's name + description. Falls back to most-recently-indexed
        if no document scores any match.
        """
        meta = self._read_meta() or {}
        ready = [(k, v) for k, v in meta.items() if v.get("status") == STATUS_READY]
        if not ready:
            return None
        if len(ready) == 1:
            return ready[0][0]

        # Tokenise question into lowercase words (3+ chars to skip noise)
        q_tokens = set(re.findall(r'[a-z]{3,}', question.lower()))

        best_id, best_score = None, -1
        for doc_id, entry in ready:
            # Filename tokens weighted 3x — "boeing" or "amd" in filename is a strong signal
            name_tokens = set(re.findall(r'[a-z]{3,}', entry.get("doc_name", "").lower()))
            desc_tokens = set(re.findall(r'[a-z]{3,}', entry.get("doc_description", "").lower()))
            score = 3 * len(q_tokens & name_tokens) + len(q_tokens & desc_tokens)
            if score > best_score:
                best_score, best_id = score, doc_id

        # If no keyword overlap at all, fall back to most recent
        if best_score == 0:
            ready.sort(key=lambda kv: kv[1].get("indexed_at", ""), reverse=True)
            return ready[0][0]

        logger.info("_route_doc: selected %s (score=%d) for question: %.80s",
                    best_id, best_score, question)
        return best_id

    # ── PageIndexClient lazy init ─────────────────────────────────────────────

    def _get_client(self):
        """Return a PageIndexClient, creating it on first call."""
        if self._client is None:
            from pageindex.client import PageIndexClient
            self._client = PageIndexClient(
                model=self.model.replace("anthropic/", "").replace("gemini/", ""),
                workspace=str(self.workspace / "client_workspace"),
            )
            # Override model so LiteLLM routes through the right provider prefix
            self._client.model = self.model
        return self._client

    # ── Retry helper ─────────────────────────────────────────────────────────

    def _retry_with_backoff(
        self,
        fn,
        max_attempts: int = 5,
        progress_tracker=None,
        progress_range=(10, 80),
    ):
        """
        Call fn() up to max_attempts times with exponential back-off.
        Raises the last exception if all attempts fail.
        """
        p_start, p_end = progress_range
        last_exc = None

        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as e:
                last_exc = e
                if attempt + 1 == max_attempts:
                    break

                wait = min(2 ** attempt, 60)   # 1, 2, 4, 8, 16 … max 60s
                logger.warning(
                    "PageIndex attempt %d/%d failed (%s). Retrying in %ds…",
                    attempt + 1, max_attempts, e, wait,
                )
                if progress_tracker:
                    pct = p_start + int((attempt / max_attempts) * (p_end - p_start))
                    progress_tracker.update(
                        pct, 100, status="retrying",
                        message=f"Rate limited or transient error — retrying in {wait}s "
                                f"(attempt {attempt + 1}/{max_attempts}): {e}",
                    )
                time.sleep(wait)

        raise last_exc

    # ── Misc helpers ──────────────────────────────────────────────────────────

    def _resolve_model(self) -> str:
        """
        Pick the best available LiteLLM model path based on configured API keys.
        Priority: Gemini → OpenAI env var.
        """
        # User may explicitly set a model in config
        explicit = self.config.get("pageindex_model", "").strip()
        if explicit:
            return explicit

        if self.config.get("gemini_api_key"):
            os.environ.setdefault("GEMINI_API_KEY", self.config["gemini_api_key"])
            return GEMINI_MODEL

        if os.environ.get("OPENAI_API_KEY"):
            return OPENAI_MODEL

        logger.warning(
            "No LLM API key found for PageIndex. Set gemini_api_key in config."
        )
        return ""

    def _check_deps(self):
        """Raise ImportError with a helpful message if pageindex or litellm are missing."""
        try:
            import pageindex  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"pageindex package not found: {e}. "
                "Make sure the pageindex/ directory is present in the SimpleRAGx project root "
                "(it should have been bundled — re-run: "
                "cp -r /tmp/pageindex_src/pageindex .  after cloning VectifyAI/PageIndex)."
            )
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"litellm not installed: {e}. Run: pip install litellm>=1.40.0"
            )

    @staticmethod
    def _count_nodes(structure: list) -> int:
        """Recursively count all nodes in the tree."""
        total = 0
        for node in structure:
            total += 1
            if node.get("nodes"):
                total += PageIndexService._count_nodes(node["nodes"])
        return total

    @staticmethod
    def _section_for_page(page_num: int, structure: list) -> str:
        """Walk the tree and return the title of the section that contains page_num."""
        for node in structure:
            start = node.get("start_index", 0)
            end   = node.get("end_index",   0)
            if start <= page_num <= end:
                # Recurse into children for a more precise match
                child_match = PageIndexService._section_for_page(
                    page_num, node.get("nodes", [])
                )
                return child_match if child_match else node.get("title", "")
        return ""

    @staticmethod
    def _needs_calculation(question: str) -> bool:
        """Detect questions that require arithmetic rather than pure text retrieval."""
        triggers = (
            "ratio", "calculate", "margin", "percentage", "effective tax",
            "quick ratio", "current ratio", "how much more", "how much less",
            "compared to", "growth rate", "increase by", "decrease by",
            "total", "net income", "earnings per", "return on",
        )
        q = question.lower()
        return any(t in q for t in triggers)

    @staticmethod
    def _answer_has_numbers(answer: str) -> bool:
        """Return True if the answer already contains numeric data."""
        return bool(re.search(r'\d+\.?\d*\s*%|\$\s*[\d,]+|\d[\d,]+\s*(million|billion)', answer, re.IGNORECASE))

    @staticmethod
    def _render_pages_as_images(pdf_path: str, page_nums: List[int]) -> List[tuple]:
        """Render PDF pages to base64 PNG using PyMuPDF. Returns [(page_num, b64), ...]."""
        results = []
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            mat = pymupdf.Matrix(1.5, 1.5)  # ~108 DPI — readable without huge token cost
            for pg in page_nums:
                idx = pg - 1
                if 0 <= idx < len(doc):
                    pix = doc[idx].get_pixmap(matrix=mat)
                    b64 = base64.b64encode(pix.tobytes("png")).decode()
                    results.append((pg, b64))
            doc.close()
        except Exception as e:
            logger.warning("Page image render failed: %s", e)
        return results

    def _vision_fallback(
        self,
        question:      str,
        pdf_path:      str,
        page_nums:     List[int],
        extra_kwargs:  dict,
    ) -> str:
        """
        Send rendered page images to the vision LLM when text extraction misses the answer.
        Handles data locked in charts, diagrams, and scanned tables.
        """
        import litellm
        images = self._render_pages_as_images(pdf_path, page_nums)
        if not images:
            return ""

        content: List[Any] = [
            {
                "type": "text",
                "text": (
                    f"Answer the question below using ONLY the document pages shown. "
                    f"Read all values directly from tables, charts, and graphics in the images. "
                    f"End your answer with 'Sources: pages X'.\n\nQuestion: {question}"
                ),
            }
        ]
        for pg, b64 in images:
            content.append({"type": "text", "text": f"[Page {pg}]"})
            content.append({
                "type":      "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        try:
            resp = litellm.completion(
                model=self.model,
                messages=[
                    {
                        "role":    "system",
                        "content": "You are a precise document analyst. Extract exact figures from the page images, including values shown only in charts or graphics.",
                    },
                    {"role": "user", "content": content},
                ],
                temperature=0,
                **extra_kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning("Vision fallback LLM call failed: %s", e)
            return ""

    @staticmethod
    def _error_result(msg: str) -> Dict[str, Any]:
        return {
            "success":    False,
            "answer":     f"PageIndex error: {msg}",
            "citations":  [],
            "tree_path":  [],
            "tool_calls": [],
            "doc_id":     None,
            "doc_name":   "",
            "error":      msg,
        }
