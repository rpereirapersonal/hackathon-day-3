"""RBA cash-rate decision tools — structured queries and derived statistics.

**Blocked: BLK-2** (real schema unknown). Covers FR-4.1 and FR-4.4.

The highest-priority capability group in the whole tool layer: both of the
brief's §10 worked failures are RBA questions, and both fail the same way — a
statistic that was never actually computed. Every number these tools return is
computed in SQL, never left to a model (FR-3.4).

Planned tools, in build order — see docs/tool-backlog.md for the full rationale:

1. ``rba_decision_stats``   — counts over a date range: decisions, changes,
                              increases, decreases, holds. Answers §10 ex. 1.
2. ``rba_decisions``        — bounded chronological listing for a date range,
                              with first/last framing.
3. ``rba_hold_runs``        — longest / ranked runs of unchanged rates, with
                              start and end dates. Answers §10 ex. 2.
4. ``rba_rate_at``          — the cash rate in force on a given date, plus the
                              decision that set it.
"""

from __future__ import annotations

# TODO(build step 4): implement against the real schema. Each tool is an
# `async def` decorated with `@tool`, with a model-facing docstring stating what
# it is for, when to reach for it, and what each argument means. Numeric
# arguments carry declared min/max so out-of-range values are rejected at the
# schema layer before any data access (FR-3.1). Connections are opened
# read-only (FR-3.7, CON-3).
