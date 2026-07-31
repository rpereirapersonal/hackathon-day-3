"""Async SQLite access to the AFR index — read-only, one connection per call.

Connections are scoped per operation rather than shared through a module-level
handle, so three concurrent requests cannot contend on one cursor (NFR-3.3).
Opened read-only through a ``file:`` URI, because no tool may modify the
ingested artifacts and none of them may touch the source datasets (FR-3.7,
CON-3).

Schema, written by ``src/ingest.py``:

``afr_body(id INTEGER PRIMARY KEY, body TEXT NOT NULL)``
    One row per AFR record. ``body`` is ``src.text.raw_text`` — the four text
    fields joined by newlines, unescaped. ``id`` is 1-based and aligned with
    ``afr_meta.parquet`` row order and with row ``id - 1`` of
    ``afr_vectors.npy``.

``afr_fts`` — FTS5 over ``body``, external content on ``afr_body``
    The tokenised index. Its ``unicode61`` tokenizer lowercases and splits on
    punctuation, which is exactly what ``src.text.norm_text`` does, so every
    word-bounded match is guaranteed to appear as a candidate. External
    content means the text is stored once rather than twice.

Substring terms deliberately bypass the index and scan ``afr_body`` with
``LIKE``: a substring can fall inside a token ("corporate cuts" contains
"rate cut") where no tokenised query can reach it. That scan runs in C over
the stored column in seconds, against the sixty a Python pass over the corpus
costs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from src.config import load_data_paths


class IndexMissingError(RuntimeError):
    """Raised when the ingested AFR index is absent.

    Actionable rather than a bare ``OperationalError``: the fix is always to
    run the ingest step, and a tool that says so lets the reasoning brain
    report a real limitation instead of retrying blindly.
    """


def afr_db_path() -> Path:
    return load_data_paths().afr_db


@asynccontextmanager
async def aconnect(path: Path | None = None) -> AsyncIterator[aiosqlite.Connection]:
    """Open a read-only connection to the AFR index for one operation.

    Args:
        path: Override the configured database location. Tests pass a fixture
            database here.

    Yields:
        An open connection with a mapping row factory.

    Raises:
        IndexMissingError: if the database file does not exist.
    """
    resolved = path if path is not None else afr_db_path()
    if not resolved.exists():
        raise IndexMissingError(
            f"AFR index not found at {resolved}. Build it once with "
            "`python -m src.ingest` before serving."
        )

    connection = await aiosqlite.connect(
        f"file:{resolved.as_posix()}?mode=ro", uri=True
    )
    try:
        connection.row_factory = aiosqlite.Row
        yield connection
    finally:
        await connection.close()
