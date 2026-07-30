"""Async SQLite access — read-only, one connection per operation.

Connections are scoped per operation rather than shared through a module-level
handle, so three concurrent requests cannot contend on one cursor (NFR-3.3).
Opened read-only, because no tool may modify the source datasets (FR-3.7,
CON-3).

**Blocked: BLK-2** (database is built by ``scripts/ingest.py`` from files not
yet available).

TODO(build step 4): implement ``aconnect()`` as an async context manager over
aiosqlite, opening DB_PATH with ``mode=ro`` via a file: URI and a row factory
that yields mappings.
"""

from __future__ import annotations

# TODO(build step 4): @asynccontextmanager async def aconnect() -> AsyncIterator[Connection]
