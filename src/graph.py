"""The outer graph: reason -> synthesize -> package.

Three nodes, one linear path. The linearity is the point — ``synthesize`` is
terminal apart from its edge to ``package``, so no synthesis output can route
back into the reasoning loop, and ``package`` reads ``answer`` only from the
synthesis node's output (FR-5.2, CON-7, AC-4).

The reasoning agent is a compiled subgraph used as a node rather than being the
whole graph, which makes the Qwen -> Nemotron hand-off a visible edge and keeps
deadline and packaging concerns out of the agent's control flow
(architecture.md §4).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from src.context import QueryContext
from src.orchestrator import build_orchestrator
from src.state import GraphState
from src.synthesis import synthesize

logger = logging.getLogger(__name__)


async def _reason(state: GraphState, runtime: Any = None) -> dict[str, Any]:
    """Run the Qwen reasoning subgraph over the question.

    Seeds ``messages`` from ``question`` so the agent is driven by the request
    field rather than requiring the caller to pre-build a message list, and
    counts one reasoning step per model turn for the ``steps`` field (FR-6.1).
    """
    agent = build_orchestrator()

    messages = list(state.get("messages") or [])
    if not messages:
        messages = [HumanMessage(content=state.get("question", ""))]

    try:
        result = await agent.ainvoke(
            {"messages": messages},
            context=getattr(runtime, "context", None),
        )
    except Exception:
        # The reasoning brain failing is a degradation, not a request failure:
        # the edge to synthesize still runs, so any evidence gathered before
        # the fault is still written up rather than discarded (FR-2.5, FR-1.6).
        logger.exception("Reasoning failed; continuing to synthesis with what we have.")
        return {}

    produced = list(result.get("messages") or [])
    update: dict[str, Any] = {
        # Only the messages the subgraph added: the seed is already in state,
        # and re-emitting it would duplicate it in the reducer.
        "messages": produced[len(messages) :],
        # `steps` is an additive channel, so this contributes the count rather
        # than setting it. AIMessages are the reasoning turns; tool results are
        # not iterations of the loop.
        "steps": sum(1 for m in produced if m.__class__.__name__ == "AIMessage"),
    }

    # The subgraph owns the trace channel; carry through whatever it recorded.
    trace = result.get("tool_trace")
    if trace:
        update["tool_trace"] = list(trace)
    return update


def _package(state: GraphState) -> dict[str, Any]:
    """Normalise the response fields.

    Reads ``answer`` from state, which only ``synthesize`` writes — never from
    the reasoning messages (FR-2.4, CON-7). The empty-answer guard is the last
    line of defence for FR-1.4: synthesis already degrades rather than
    returning nothing, so reaching this branch means an unexpected path, and
    returning an empty string would score zero.
    """
    answer = (state.get("answer") or "").strip()
    if not answer:
        logger.error("Reached package with no answer; emitting a stated limitation.")
        answer = (
            "The available data does not support an answer to this question."
        )
    return {"answer": answer}


def build_graph():
    """Compile the outer graph.

    A factory as well as a module-level instance: tests need to build the graph
    without inheriting the process-wide one, and ``langgraph.json`` needs the
    instance.
    """
    builder = StateGraph(GraphState, context_schema=QueryContext)

    builder.add_node("reason", _reason)
    builder.add_node("synthesize", synthesize)
    builder.add_node("package", _package)

    builder.add_edge(START, "reason")
    builder.add_edge("reason", "synthesize")
    builder.add_edge("synthesize", "package")
    builder.add_edge("package", END)

    return builder.compile(name="market-signal-agent")


#: The compiled graph referenced by ``langgraph.json``.
graph = build_graph()
