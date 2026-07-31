"""The single definition of the scored JSON contract.

Every response — success, degraded, or internal error — is serialised through
these models, so no path can drift out of shape (FR-1.3, FR-1.5, CON-4).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """The evaluator sends exactly one field (FR-1.2)."""

    question: str


class TraceEntry(BaseModel):
    """One recorded tool execution.

    Mirrors ``state.TraceEntry``, which is the ``TypedDict`` the graph channel
    accumulates. Two declarations of one shape is a real cost, but the channel
    needs a ``TypedDict`` for ``operator.add`` reduction and the response needs
    a pydantic model for serialisation, and coupling them would drag pydantic
    into the state layer for no gain.
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str


class QueryResponse(BaseModel):
    """The scored response body.

    ``answer`` is required and must never be empty — an empty answer scores
    zero regardless of what the trace shows (FR-1.4).
    """

    answer: str
    steps: int = 0
    tool_trace: list[TraceEntry] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Body of the health gate.

    ``domain_predict_mode`` is echoed so a submission accidentally left in
    ``mock`` is visible at a glance rather than discovered at scoring
    (FR-5.5, AC-12).
    """

    status: str
    domain_predict_mode: str
