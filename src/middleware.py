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
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

from src.context import QueryContext

if TYPE_CHECKING:
    from langchain.agents.middleware import ToolCallRequest
    from langgraph.types import Command

logger = logging.getLogger(__name__)

Handler = Callable[["ToolCallRequest"], Awaitable["ToolMessage | Command"]]

#: Trace results are returned in the HTTP response and fed to synthesis, so
#: they are summaries, not data dumps (FR-3.3).
_MAX_TRACE_RESULT = 2000


def _context_of(request: "ToolCallRequest") -> QueryContext | None:
    """The request-scoped runtime, or None when invoked without one (tests)."""
    return getattr(request.runtime, "context", None)


def _call_of(request: "ToolCallRequest") -> tuple[str, dict[str, Any], str]:
    """The tool call's name, arguments and id, however the call is shaped."""
    call = request.tool_call
    if isinstance(call, dict):
        return (
            call.get("name", "unknown"),
            call.get("args", {}) or {},
            call.get("id", "") or "",
        )
    return (
        getattr(call, "name", "unknown"),
        getattr(call, "args", {}) or {},
        getattr(call, "id", "") or "",
    )


def _refusal(request: "ToolCallRequest", reason: str) -> ToolMessage:
    """A refused call, shaped as a normal tool result.

    An error ToolMessage rather than an exception: the model reads it, stops
    asking, and the loop closes into synthesis instead of unwinding (FR-3.6,
    NFR-2.3).
    """
    name, _, call_id = _call_of(request)
    return ToolMessage(content=reason, tool_call_id=call_id, name=name, status="error")


@wrap_tool_call
async def deadline_guard(
    request: "ToolCallRequest", handler: Handler
) -> "ToolMessage | Command":
    """Stop starting new tool work once the request deadline has passed."""
    ctx = _context_of(request)
    if ctx is None:
        return await handler(request)

    remaining = ctx.seconds_remaining
    if remaining <= 0:
        logger.warning("Deadline passed; refusing further tool calls.")
        return _refusal(
            request,
            "Time budget for this request is exhausted. No further data can be "
            "gathered. Stop calling tools and finish with the evidence already "
            "collected.",
        )

    # A per-call ceiling inside the remaining budget. Without it a single slow
    # tool consumes the whole request and the deadline check above never gets
    # another chance to fire.
    try:
        return await asyncio.wait_for(handler(request), timeout=remaining)
    except asyncio.TimeoutError:
        name, _, _ = _call_of(request)
        logger.warning("Tool %s exceeded the remaining %.1fs budget.", name, remaining)
        return _refusal(
            request,
            f"The {name} call exceeded the remaining time budget and was "
            "stopped. Finish with the evidence already collected.",
        )


@wrap_tool_call
async def tool_budget(
    request: "ToolCallRequest", handler: Handler
) -> "ToolMessage | Command":
    """Enforce the hard per-request cap on tool calls."""
    ctx = _context_of(request)
    if ctx is None:
        return await handler(request)

    if ctx.budget_remaining <= 0:
        logger.warning(
            "Tool budget of %d exhausted; refusing further calls.", ctx.tool_budget
        )
        return _refusal(
            request,
            f"The limit of {ctx.tool_budget} tool calls for this request has "
            "been reached. Stop calling tools and finish with the evidence "
            "already collected.",
        )

    # Counted on dispatch, not on success. A failed call has already cost its
    # share of the latency budget, so charging only successes would let a
    # failing tool be retried without limit.
    ctx.tool_calls_used += 1
    return await handler(request)


@wrap_tool_call
async def trace_recorder(
    request: "ToolCallRequest", handler: Handler
) -> "ToolMessage | Command":
    """Append this execution to ``tool_trace`` and log its timing."""
    from langgraph.types import Command

    name, args, _ = _call_of(request)
    started = time.monotonic()

    try:
        result = await handler(request)
    except Exception as exc:
        # A tool that raises is evidence about the tool, not a request failure
        # (FR-3.6). Record it and hand the model a readable error.
        elapsed = time.monotonic() - started
        logger.warning("Tool %s raised after %.2fs: %s", name, elapsed, exc)
        summary = f"{name} failed: {exc}"
        return Command(
            update={
                "messages": [_refusal(request, summary)],
                "tool_trace": [
                    {"tool": name, "args": dict(args), "result": summary[:_MAX_TRACE_RESULT]}
                ],
            }
        )

    elapsed = time.monotonic() - started
    logger.info("Tool %s completed in %.2fs", name, elapsed)

    entry = {
        "tool": name,
        "args": dict(args),
        "result": _summarise(result),
    }

    # A tool returning a Command already carries its own state update; merge
    # the trace entry into it rather than discarding either.
    if isinstance(result, Command):
        update = dict(result.update or {}) if isinstance(result.update, dict) else {}
        update["tool_trace"] = [*(update.get("tool_trace") or []), entry]
        return Command(update=update, goto=result.goto)

    return Command(update={"messages": [result], "tool_trace": [entry]})


def _summarise(result: Any) -> str:
    """A truncated string form of a tool result, safe to return in the trace."""
    content = getattr(result, "content", result)
    text = content if isinstance(content, str) else str(content)
    text = text.strip()
    if len(text) > _MAX_TRACE_RESULT:
        return text[:_MAX_TRACE_RESULT] + " ... [truncated]"
    return text


# Outermost first. ``trace_recorder`` wraps the others so that a call refused by
# the deadline guard or the budget cap is still recorded in the trace (FR-6.2) —
# an inner recorder would never see a short-circuited call. The deadline then
# outranks the budget: once time is up, budget accounting is moot.
DEFAULT_MIDDLEWARE = (trace_recorder, deadline_guard, tool_budget)
