"""AFR news tools — semantic retrieval for sentiment and market direction.

**Blocked: BLK-2** (corpus not yet available). Covers FR-4.3.

Scope discipline matters here more than anywhere else in the tool layer. This
group answers questions about *tone, narrative and direction*. It must never
be the route to a count, a rate level or a computed price change — that
substitution is the brief's §10 example 1, and it scores zero. The tool
docstrings say so explicitly, so the reasoning brain is steered away from it by
the tool descriptions and not only by the system prompt.

Planned tools, in build order — see docs/tool-backlog.md:

1. ``afr_search``          — top-k semantically relevant articles for a query,
                             bounded k, returning compact excerpts rather than
                             full article text.
2. ``afr_search_window``   — the same, scoped to a date range, so "coverage in
                             the week after <event>" is answerable.
3. ``afr_article_context`` — a longer bounded excerpt for one already-retrieved
                             article, when a headline is not enough.
"""

from __future__ import annotations

# TODO(build step 4): implement over src/retrieval.py. If the embedding stack's
# native dependency cannot be satisfied on the host's aarch64 architecture
# (DEP-4), these tools fall back to SQLite FTS5 keyword search — weaker recall,
# same signatures, critical path unblocked (architecture.md §6).
