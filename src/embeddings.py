"""Embedding generation for AFR retrieval.

The encoder is synchronous and CPU-bound, so it is executed off the event loop
— otherwise one embedding call stalls the two other concurrent requests
(NFR-3.4). The model is loaded at import, before the service reports healthy,
never inside a request (NFR-1.4).

**Blocked: BLK-2.** See ``retrieval.py`` for the FTS5 fallback if the native
dependency cannot be satisfied on aarch64 (DEP-4).

TODO(build step 4): implement, mirroring the sync-encoder-wrapped-for-async
pattern from the reference project.
"""

from __future__ import annotations

# TODO(build step 4):
#   _model = <encoder>(EMBEDDING_MODEL_NAME, cache_dir=EMBEDDING_CACHE_DIR)
#   def embed(texts: list[str]) -> list[list[float]]      # sync, for indexing
#   async def aembed(texts: list[str]) -> list[list[float]]  # asyncio.to_thread
