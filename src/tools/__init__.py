"""Tool registry — the single list bound to the reasoning brain.

**Blocked: BLK-2.** No tool is implemented yet, because no column name,
instrument identifier or date format is asserted anywhere in this repository
until the real RBA / ASX / AFR files are in hand. ``TOOLS`` is deliberately
empty rather than speculative: an empty registry makes the gap obvious at
startup (``orchestrator.build_orchestrator`` logs a warning), whereas a tool
built against a guessed schema would fail silently at evaluation time.

Build order and rationale for each planned tool live in
``docs/tool-backlog.md``. The settled design principles — async, compact
evidence rather than data dumps, bounded arguments, deterministic maths in SQL
or Python, no unbounded scans, errors returned as results, read-only access —
are in ``architecture.md`` §5 and will not change when the data arrives.
"""

from __future__ import annotations

from typing import Any

# TODO(build step 4): populate from src/tools/rba.py, asx.py and afr.py once
# the dataset schemas are known (BLK-2). Ordering matters only for the model's
# reading of the tool list; keep the highest-value capability first.
TOOLS: tuple[Any, ...] = ()

__all__ = ["TOOLS"]
