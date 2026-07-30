"""Per-request runtime context.

``create_agent`` accepts a ``context_schema`` dataclass supplied at
``ainvoke`` time and exposed to tools and middleware as ``runtime.context``.
It carries the request-scoped values the model must not be able to see or
forge: the correlation id, the hard deadline, and the remaining tool budget.

Nothing here is module-level mutable state — a fresh instance is built per
request in ``api.py`` (NFR-3.2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class QueryContext:
    """Request-scoped runtime passed into the graph.

    Attributes:
        request_id: Correlation id, echoed into logs so a failed request can be
            traced end to end (FR-6.3).
        deadline: Monotonic clock value after which the request must stop
            working and return a degraded but valid answer (NFR-1.2).
        tool_budget: Hard cap on tool calls for this request (NFR-2.1/2.2).
        tool_calls_used: Number of tool calls already executed. Maintained by
            the budget middleware, not by the model.
    """

    request_id: str
    deadline: float
    tool_budget: int
    tool_calls_used: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def seconds_remaining(self) -> float:
        """Time left before the hard deadline. Negative once it has passed."""
        return self.deadline - time.monotonic()

    @property
    def budget_remaining(self) -> int:
        """Tool calls still permitted for this request."""
        return max(0, self.tool_budget - self.tool_calls_used)
