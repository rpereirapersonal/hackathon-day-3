"""Tool registry — the single list bound to the reasoning brain.

Ordering is the model's reading order, so the highest-value capability comes
first within each dataset group, and the coverage tool sits at the top: it is
the cheapest way to establish whether a question is answerable at all before
anything else is attempted.

Twelve tools is already past the point where routing stays comfortable under a
tight call budget, which is why near-duplicates were deliberately not built.
There is no separate "price on a date" tool alongside ``asx_return``, no generic
SQL passthrough, and no calculator — if a model needs a calculator, the wrong
tool was called, because the statistic belongs in the tool that already has the
rows. The pressure runs the other way too: ``asx_return`` takes several window
start dates in one call, and ``rba_decision_stats`` groups by year, precisely so
multi-period questions do not need a call apiece.
Design rules for all of them are in ``architecture.md`` §5; the ranked backlog of
what was built and what was deliberately deferred is in ``tool-backlog.md``.
"""

from __future__ import annotations

from typing import Any

from src.tools.afr import AFR_TOOLS
from src.tools.asx import ASX_TOOLS
from src.tools.meta import META_TOOLS
from src.tools.rba import RBA_TOOLS

TOOLS: tuple[Any, ...] = (*META_TOOLS, *RBA_TOOLS, *ASX_TOOLS, *AFR_TOOLS)

__all__ = ["TOOLS"]
