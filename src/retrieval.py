"""Semantic search over the AFR news corpus.

**Blocked: BLK-2** (corpus not yet available).

Returns compact, bounded excerpts — evidence, not data dumps (FR-3.5).

Documented fallback (DEP-4, architecture.md §6): the embedding stack carries a
native dependency (``onnxruntime``) whose availability on the host's aarch64
architecture is confirmed in build step 0. If it cannot be satisfied there,
retrieval degrades to SQLite FTS5 keyword search — no native dependency, weaker
recall, same public interface, critical path unblocked.

TODO(build step 4): implement search over the vector index built by
``scripts/ingest.py``, with a bounded top-k and an optional date-range filter.
"""

from __future__ import annotations

# TODO(build step 4): async def search(query: str, k: int, ...) -> list[Excerpt]
