"""Cross-cutting concerns for the reasoning loop.

Three wrappers, applied outermost-first in ``DEFAULT_MIDDLEWARE``:

* ``deadline_guard`` — refuses further tool calls once the request's hard
  deadline has passed, so the loop degrades instead of overrunning the latency
  band (NFR-1.2, AC-8).
* ``tool_budget`` — hard cap on tool calls, enforced in code rather than
  requested in a prompt (NFR-2.2, AC-6).
* ``trace_recorder`` — appends every execution, including failures and
  refusals, to the ``tool_trace`` channel (FR-3.3, FR-6.2).

Keeping these out of the tools means a new tool inherits budget, deadline and
tracing behaviour for free, and cannot forget to implement them.

All three short-circuit or record with a ``ToolMessage`` rather than raising,
so a breach never propagates out of the graph (FR-3.6). Every termination path
flows on to synthesis; none is an error path (FR-2.5, NFR-2.3).

Pattern follows https://docs.langchain.com/oss/python/langchain/middleware/custom.

TODO(build step 3): these are pass-through stubs. They wire the stack up and
log, but do not yet enforce or record. Implement alongside ``synthesis.py``.
Evaluate whether ``langchain.agents.middleware.ToolCallLimitMiddleware`` can
replace ``tool_budget`` outright before hand-rolling it.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain.agents.middleware import wrap_tool_call

from src.context import QueryContext

if TYPE_CHECKING:
    from langchain.agents.middleware import ToolCallRequest
    from langchain.messages import ToolMessage
    from langgraph.types import Command

logger = logging.getLogger(__name__)

Handler = Callable[["ToolCallRequest"], Awaitable["ToolMessage | Command"]]


def _context_of(request: "ToolCallRequest") -> QueryContext | None:
    """The request-scoped runtime, or None when invoked without one (tests)."""
    return getattr(request.runtime, "context", None)


@wrap_tool_call
async def deadline_guard(
    request: "ToolCallRequest", handler: Handler
) -> "ToolMessage | Command":
    """Stop starting new tool work once the request deadline has passed."""
    # TODO(build step 3): when ctx.seconds_remaining <= 0, short-circuit with an
    # error ToolMessage so the loop closes and synthesis runs on the evidence
    # gathered so far. Also enforce a per-call timeout inside the remaining
    # budget rather than only checking before dispatch.
    return await handler(request)


@wrap_tool_call
async def tool_budget(
    request: "ToolCallRequest", handler: Handler
) -> "ToolMessage | Command":
    """Enforce the hard per-request cap on tool calls."""
    # TODO(build step 3): increment ctx.tool_calls_used, and once
    # ctx.budget_remaining is 0, refuse with an error ToolMessage explaining the
    # cap. Exhaustion proceeds to synthesis; it must not raise (NFR-2.3).
    return await handler(request)


@wrap_tool_call
async def trace_recorder(
    request: "ToolCallRequest", handler: Handler
) -> "ToolMessage | Command":
    """Append this execution to ``tool_trace`` and log its timing."""
    # TODO(build step 3): time the call and return a Command that appends a
    # TraceEntry to the tool_trace channel — name, args, and a truncated result
    # summary. Record refused and failed calls too (FR-6.2). Never log
    # credentials (NFR-6.4).
    return await handler(request)


# Outermost first. ``trace_recorder`` wraps the others so that a call refused by
# the deadline guard or the budget cap is still recorded in the trace (FR-6.2) —
# an inner recorder would never see a short-circuited call. The deadline then
# outranks the budget: once time is up, budget accounting is moot.
DEFAULT_MIDDLEWARE = (trace_recorder, deadline_guard, tool_budget)
