"""Budget and deadline tests — AC-6, AC-8.

Covers NFR-1.2 and NFR-2. Both are enforced in code rather than requested in a
prompt, so both are testable without a live model.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from src.context import QueryContext
from src.middleware import deadline_guard, tool_budget, trace_recorder


class _Request:
    """A minimal ``ToolCallRequest`` stand-in.

    The middleware reads only ``tool_call`` and ``runtime``, so a full request
    object would add coupling to internals these tests are not about.
    """

    def __init__(self, runtime, name: str = "rba_rate_at", args: dict | None = None):
        self.tool_call = {"name": name, "args": args or {}, "id": "c1"}
        self.runtime = runtime
        self.tool = None
        self.state = {}


def _handler(content: str = "4.35 percent"):
    async def handle(request):
        return ToolMessage(content=content, tool_call_id="c1", name="rba_rate_at")

    return handle


def _wrapped(middleware):
    """The async callable a middleware decorator exposes."""
    return middleware.awrap_tool_call


# ---------------------------------------------------------------------------
# Budget — AC-6, NFR-2.2, NFR-2.3
# ---------------------------------------------------------------------------
def test_budget_permits_calls_up_to_the_cap(fake_runtime):
    runtime = fake_runtime(tool_budget=3)
    wrapped = _wrapped(tool_budget)
    for _ in range(3):
        result = asyncio.run(wrapped(_Request(runtime), _handler()))
        assert result.status != "error"
    assert runtime.context.tool_calls_used == 3


def test_budget_refuses_past_the_cap_without_raising(fake_runtime):
    """Exhaustion is a normal termination path, not an error (NFR-2.3)."""
    runtime = fake_runtime(tool_budget=1)
    wrapped = _wrapped(tool_budget)
    asyncio.run(wrapped(_Request(runtime), _handler()))

    refused = asyncio.run(wrapped(_Request(runtime), _handler()))
    assert refused.status == "error"
    assert "limit of 1 tool calls" in refused.content
    # The cap holds: the refused call did not reach the handler.
    assert runtime.context.tool_calls_used == 1


def test_budget_counts_failed_calls(fake_runtime):
    """A failing tool still consumes budget, or it can be retried forever."""
    runtime = fake_runtime(tool_budget=2)

    async def failing(request):
        raise RuntimeError("tool exploded")

    with pytest.raises(RuntimeError):
        asyncio.run(_wrapped(tool_budget)(_Request(runtime), failing))
    assert runtime.context.tool_calls_used == 1


# ---------------------------------------------------------------------------
# Deadline — AC-8, NFR-1.2
# ---------------------------------------------------------------------------
def test_deadline_guard_refuses_once_time_is_up(fake_runtime):
    runtime = fake_runtime(deadline_in=-1.0)
    result = asyncio.run(_wrapped(deadline_guard)(_Request(runtime), _handler()))
    assert result.status == "error"
    assert "time budget" in result.content.lower()


def test_deadline_guard_stops_a_slow_tool(fake_runtime):
    """A tool slower than the remaining budget is cut off, not awaited."""
    runtime = fake_runtime(deadline_in=0.2)

    async def slow(request):
        await asyncio.sleep(5)
        return ToolMessage(content="too late", tool_call_id="c1", name="rba_rate_at")

    started = time.monotonic()
    result = asyncio.run(_wrapped(deadline_guard)(_Request(runtime), slow))
    assert time.monotonic() - started < 2
    assert result.status == "error"


def test_deadline_guard_passes_through_with_time_left(fake_runtime):
    runtime = fake_runtime(deadline_in=30.0)
    result = asyncio.run(_wrapped(deadline_guard)(_Request(runtime), _handler()))
    assert result.status != "error"
    assert "4.35" in result.content


def test_middleware_is_inert_without_a_runtime():
    """Direct invocation in tests must not require a QueryContext."""

    class _Bare:
        context = None

    for middleware in (deadline_guard, tool_budget):
        result = asyncio.run(_wrapped(middleware)(_Request(_Bare()), _handler()))
        assert "4.35" in result.content


# ---------------------------------------------------------------------------
# Trace — FR-6.2
# ---------------------------------------------------------------------------
def test_trace_records_a_successful_call(fake_runtime):
    runtime = fake_runtime()
    result = asyncio.run(
        _wrapped(trace_recorder)(
            _Request(runtime, args={"as_of": "2024-06-18"}), _handler()
        )
    )
    assert isinstance(result, Command)
    entry = result.update["tool_trace"][0]
    assert entry["tool"] == "rba_rate_at"
    assert entry["args"] == {"as_of": "2024-06-18"}
    assert "4.35" in entry["result"]


def test_trace_records_a_raising_tool_without_propagating(fake_runtime):
    """A tool that raises is evidence, not a request failure (FR-3.6)."""
    runtime = fake_runtime()

    async def failing(request):
        raise RuntimeError("database is missing")

    result = asyncio.run(_wrapped(trace_recorder)(_Request(runtime), failing))
    assert isinstance(result, Command)
    assert "database is missing" in result.update["tool_trace"][0]["result"]
    assert result.update["messages"][0].status == "error"


def test_trace_records_a_refused_call(fake_runtime):
    """A call refused by the budget cap still appears in the trace (FR-6.2).

    This is why ``trace_recorder`` is outermost in ``DEFAULT_MIDDLEWARE``.
    """
    runtime = fake_runtime(tool_budget=0)

    async def through_budget(request):
        return await _wrapped(tool_budget)(request, _handler())

    result = asyncio.run(_wrapped(trace_recorder)(_Request(runtime), through_budget))
    assert isinstance(result, Command)
    assert "limit of 0 tool calls" in result.update["tool_trace"][0]["result"]


def test_trace_truncates_a_large_result(fake_runtime):
    """The trace is a diagnostic summary, not a data dump (FR-3.3)."""
    runtime = fake_runtime()
    result = asyncio.run(
        _wrapped(trace_recorder)(_Request(runtime), _handler("x" * 50_000))
    )
    recorded = result.update["tool_trace"][0]["result"]
    assert len(recorded) < 5_000
    assert recorded.endswith("[truncated]")
