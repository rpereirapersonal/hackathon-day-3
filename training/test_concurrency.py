"""Concurrency test — AC-7.

The harness sends up to three hidden questions concurrently, so this covers a
scored failure mode, not a theoretical one (NFR-3).

The graph is driven with stubbed reason/synthesize nodes rather than a live
model: what is under test is that the state channels and their reducers keep
three interleaved requests apart, and a real gateway would make that assertion
slower without making it stronger (NFR-5.4).
"""

from __future__ import annotations

import asyncio
import time

import pytest
from langgraph.graph import END, START, StateGraph

from src.context import QueryContext
from src.graph import _package
from src.state import GraphState

QUESTIONS = [
    "What was the RBA cash rate in June 2024?",
    "How did BHP shares move in March 2023?",
    "What did AFR coverage say about inflation in 2022?",
]


def _graph(reason, synthesize):
    """The real outer topology with the two model-bound nodes stubbed."""
    builder = StateGraph(GraphState, context_schema=QueryContext)
    builder.add_node("reason", reason)
    builder.add_node("synthesize", synthesize)
    builder.add_node("package", _package)
    builder.add_edge(START, "reason")
    builder.add_edge("reason", "synthesize")
    builder.add_edge("synthesize", "package")
    builder.add_edge("package", END)
    return builder.compile()


def _context(request_id: str = "x") -> QueryContext:
    return QueryContext(
        request_id=request_id, deadline=time.monotonic() + 30, tool_budget=5
    )


def test_query_context_instances_do_not_share_state():
    """Budget accounting is per request, not process-wide (NFR-3.2)."""
    first = QueryContext(request_id="a", deadline=time.monotonic() + 10, tool_budget=5)
    second = QueryContext(request_id="b", deadline=time.monotonic() + 10, tool_budget=5)

    first.tool_calls_used += 3

    assert first.budget_remaining == 2
    assert second.budget_remaining == 5
    assert second.tool_calls_used == 0


def test_trace_channels_stay_separate_across_concurrent_runs():
    """Three concurrent runs must not cross-contaminate ``tool_trace`` (AC-7)."""

    async def reason(state, runtime=None):
        # Index rather than a text slice: `_package` strips its answer, and a
        # slice ending in a space would fail on the strip, not on contamination.
        marker = str(QUESTIONS.index(state["question"]))
        # Yield control so the three runs genuinely interleave.
        await asyncio.sleep(0.01)
        return {
            "tool_trace": [{"tool": "t", "args": {}, "result": f"evidence:{marker}"}],
            "steps": 1,
        }

    async def synthesize(state, runtime=None):
        await asyncio.sleep(0.01)
        return {
            "answer": " | ".join(e["result"] for e in state.get("tool_trace", []))
        }

    graph = _graph(reason, synthesize)

    async def run_all():
        return await asyncio.gather(
            *(
                graph.ainvoke({"question": q}, context=_context(f"r{i}"))
                for i, q in enumerate(QUESTIONS)
            )
        )

    results = asyncio.run(run_all())

    assert len(results) == 3
    for index, result in enumerate(results):
        trace = result["tool_trace"]
        # Exactly its own evidence, and none of any other request's.
        assert len(trace) == 1
        assert trace[0]["result"] == f"evidence:{index}"
        assert result["answer"] == f"evidence:{index}"


def test_responses_match_their_own_question():
    """Answers must not be shuffled between concurrent callers (AC-7)."""

    async def reason(state, runtime=None):
        await asyncio.sleep(0.01)
        return {"tool_trace": [{"tool": "t", "args": {}, "result": state["question"]}]}

    async def synthesize(state, runtime=None):
        return {"answer": state["tool_trace"][0]["result"]}

    graph = _graph(reason, synthesize)

    async def run_all():
        return await asyncio.gather(
            *(graph.ainvoke({"question": q}, context=_context()) for q in QUESTIONS)
        )

    assert [r["answer"] for r in asyncio.run(run_all())] == QUESTIONS


def test_repeated_concurrent_rounds_are_identical():
    """The same three requests twice give the same results (NFR-3.2).

    The clearest evidence that no mutable module-level state participates.
    """

    async def reason(state, runtime=None):
        await asyncio.sleep(0.01)
        return {"tool_trace": [{"tool": "t", "args": {}, "result": state["question"]}]}

    async def synthesize(state, runtime=None):
        return {"answer": state["tool_trace"][0]["result"]}

    graph = _graph(reason, synthesize)

    async def run_round():
        results = await asyncio.gather(
            *(graph.ainvoke({"question": q}, context=_context()) for q in QUESTIONS)
        )
        return [r["answer"] for r in results]

    assert asyncio.run(run_round()) == asyncio.run(run_round()) == QUESTIONS


@pytest.mark.parametrize("budget", [0, 1, 5])
def test_budget_remaining_never_goes_negative(budget):
    """Over-spend must clamp, not wrap into an effectively unlimited budget."""
    ctx = QueryContext(
        request_id="a", deadline=time.monotonic() + 10, tool_budget=budget
    )
    ctx.tool_calls_used = budget + 10
    assert ctx.budget_remaining == 0
