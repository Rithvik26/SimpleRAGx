"""
SimpleRAG Extensions - Rate Limiting, Caching, and Progress Indicators
"""

import time
import threading
import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional

PROGRESS_TTL_SECONDS = 3600   # evict trackers older than 1 hour
EMBEDDING_CACHE_MAX  = 10_000  # max cached embeddings; evict oldest 20% when full

# For rate limiting
class RateLimiter:
    """Implements rate limiting for API calls."""
    
    def __init__(self, calls_per_minute: int = 60):
        """Initialize rate limiter with specified calls per minute."""
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """
        Try to acquire permission to make an API call.
        Returns True if allowed, False if rate limit exceeded.
        """
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.calls = [t for t in self.calls if now - t < 60]
            
            # Check if we're under the limit
            if len(self.calls) < self.calls_per_minute:
                self.calls.append(now)
                return True
            return False
    
    def wait_for_permission(self, timeout: float = 60.0) -> bool:
        """
        Wait until call is allowed or timeout is reached.
        Returns True if permission granted, False if timeout reached.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.acquire():
                return True
            time.sleep(0.1)  # Sleep to avoid CPU spinning
        return False

# For caching embeddings
class EmbeddingCache:
    """Disk-backed cache for embeddings with a max-entry size bound."""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".simplerag", "cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _evict_if_needed(self) -> None:
        """Delete oldest 20% of files when cache exceeds EMBEDDING_CACHE_MAX."""
        files = sorted(self.cache_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if len(files) <= EMBEDDING_CACHE_MAX:
            return
        evict_n = max(1, EMBEDDING_CACHE_MAX // 5)
        for f in files[:evict_n]:
            try:
                f.unlink()
            except OSError:
                pass

    def get(self, text: str) -> Optional[List[float]]:
        cache_file = self.cache_dir / f"{self._get_cache_key(text)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def set(self, text: str, embedding: List[float]) -> None:
        cache_file = self.cache_dir / f"{self._get_cache_key(text)}.json"
        try:
            with self._lock:
                self._evict_if_needed()
                with open(cache_file, "w") as f:
                    json.dump(embedding, f)
        except IOError:
            pass

# Decorators for using rate limiting and caching
def rate_limited(limiter: RateLimiter):
    """Decorator to apply rate limiting to a function."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if limiter.wait_for_permission():
                return func(*args, **kwargs)
            else:
                raise RuntimeError("Rate limit exceeded and timeout reached")
        return wrapper
    return decorator

def cached_embedding(cache: EmbeddingCache):
    """Decorator to apply embedding caching to a function."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, text: str):
            # Try to get from cache first
            cached_result = cache.get(text)
            if cached_result is not None:
                return cached_result
            
            # If not in cache, call the original function
            result = func(self, text)
            
            # Store result in cache
            cache.set(text, result)
            return result
        return wrapper
    return decorator

# Progress tracking
class ProgressTracker:
    """Track progress of long-running operations."""

    _last_eviction: float = 0.0
    _eviction_interval: float = 60.0  # only scan for stale entries at most once per minute

    def __init__(self, session_id: str, operation: str):
        """Initialize progress tracker for a specific operation and session."""
        self.session_id = session_id
        self.operation = operation
        self.progress = 0
        self.total = 0
        self.status = "initializing"
        self.message = ""
        self.current_file = ""
        self.created_at: float = time.time()
        self._lock = threading.Lock()

        # In-memory store for all progress trackers
        if not hasattr(ProgressTracker, "trackers"):
            ProgressTracker.trackers = {}

        ProgressTracker.trackers[f"{session_id}_{operation}"] = self
    
    def update(self, progress: int, total: int, status: str = None, message: str = None, current_file: str = None):
        """Update progress information (thread-safe)."""
        with self._lock:
            self.progress = progress
            self.total = total
            if status:
                self.status = status
            if message:
                self.message = message
            if current_file:
                self.current_file = current_file
    
    def get_info(self) -> Dict[str, Any]:
        """Get a consistent snapshot of current progress (thread-safe)."""
        with self._lock:
            percentage = int((self.progress / self.total * 100) if self.total > 0 else 0)
            return {
                "session_id":   self.session_id,
                "operation":    self.operation,
                "progress":     self.progress,
                "total":        self.total,
                "percentage":   percentage,
                "status":       self.status,
                "message":      self.message,
                "current_file": self.current_file,
            }
    
    @staticmethod
    def _evict_stale() -> None:
        """Remove trackers older than PROGRESS_TTL_SECONDS (batched, max once/minute)."""
        now = time.time()
        if now - ProgressTracker._last_eviction < ProgressTracker._eviction_interval:
            return
        ProgressTracker._last_eviction = now
        if not hasattr(ProgressTracker, "trackers"):
            return
        cutoff = now - PROGRESS_TTL_SECONDS
        stale = [k for k, v in ProgressTracker.trackers.items() if v.created_at < cutoff]
        for k in stale:
            del ProgressTracker.trackers[k]

    @staticmethod
    def get_tracker(session_id: str, operation: str) -> Optional["ProgressTracker"]:
        """Get an existing progress tracker by session_id and operation."""
        if not hasattr(ProgressTracker, "trackers"):
            return None
        ProgressTracker._evict_stale()
        return ProgressTracker.trackers.get(f"{session_id}_{operation}")