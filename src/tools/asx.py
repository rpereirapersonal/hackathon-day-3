"""ASX price tools — structured lookups and deterministic price movements.

**Blocked: BLK-2** (real schema and instrument identifiers unknown).
Covers FR-4.2 and FR-4.4.

Percentage change, absolute change and rankings are computed in SQL or Python
and returned as finished values. Neither model performs arithmetic (FR-3.4).

Planned tools, in build order — see docs/tool-backlog.md:

1. ``asx_price_change``   — computed change between two dates for one company:
                            absolute, percentage, and both endpoint prices.
2. ``asx_price_on``       — price on or nearest before a given date.
3. ``asx_top_movers``     — ranked best/worst movers over a window, bounded
                            result limit.
4. ``asx_price_series``   — bounded price series for a short window, for
                            questions about direction within a period.
5. ``asx_resolve_company``— map a company name or partial ticker to the
                            dataset's own identifier, so a near-miss name does
                            not silently return nothing.
"""

from __future__ import annotations

# TODO(build step 4): implement against the real schema. Every query is
# filtered and limited — no unbounded scans, which the brief flags as a
# guaranteed timeout (FR-3.5). Read-only connections only (FR-3.7, CON-3).
