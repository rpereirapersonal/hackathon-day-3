"""Build the SQLite tables and the vector index from the source files.

**Blocked: BLK-2.**

This script is the single adapter boundary between raw data and the tool layer.
All knowledge of raw file layout — column names, instrument identifiers, date
formats — lives here and nowhere else, so when the real schemas arrive this
script changes and ``src/tools/`` above it does not.

Runs ahead of serving, never inside a request (NFR-1.4). Reads the sources
without modifying them (CON-3).

TODO(build step 4): implement.
"""

from __future__ import annotations

# TODO(build step 4):
#   1. Read the RBA decision file  -> normalised read-only SQLite table.
#   2. Read the ASX price file(s)  -> normalised read-only SQLite table,
#      indexed on (company, date) so bounded lookups stay inside the latency
#      budget.
#   3. Read the AFR corpus -> chunk, embed, write the vector index (or the FTS5
#      table if the native dependency is unavailable on aarch64 — DEP-4).
#   4. Print a summary: row counts, date ranges, distinct companies. This is the
#      artifact that finally settles the open schema questions.
