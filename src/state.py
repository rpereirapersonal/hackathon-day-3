"""Graph state channels.

All per-request state lives here rather than in module-level variables, so
three concurrent requests cannot bleed into each other (NFR-3.1, NFR-3.2).

``tool_trace`` is append-only and becomes the ``tool_trace`` field of the
response contract (FR-1.3, FR-6.2). ``steps`` reports reasoning iterations
actually taken (FR-6.1).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, TypedDict

from langchain.agents import AgentState


class TraceEntry(TypedDict):
    """One recorded tool execution, shaped exactly as the contract requires.

    ``result`` is a *summary* string, not a data dump — the trace is a
    diagnostic artifact and is returned in the HTTP response (FR-3.3).
    """

    tool: str
    args: dict[str, Any]
    result: str


class ReasoningState(AgentState):
    """State of the Qwen reasoning subgraph.

    Extends the built-in agent state (which owns ``messages``) with the two
    diagnostic channels the response contract needs.
    """

    tool_trace: NotRequired[Annotated[list[TraceEntry], operator.add]]
    steps: NotRequired[Annotated[int, operator.add]]


class GraphState(ReasoningState):
    """State of the outer graph: reasoning state plus the synthesis output.

    ``question`` is retained verbatim so the synthesis step never has to
    reconstruct it from message history, and ``answer`` is written only by the
    ``synthesize`` node — ``package`` reads the response's ``answer`` from
    here and never from the reasoning messages (FR-2.4, CON-7).
    """

    question: NotRequired[str]
    answer: NotRequired[str]
