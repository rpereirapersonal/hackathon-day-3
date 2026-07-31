"""Shared test fixtures.

Tests never contact a live model (NFR-5.4), so they are fast and work while the
gateway credentials are outstanding (BLK-3, BLK-4).

TODO(build step 4): tmp_db — a small fixture database with known values, for
the determinism tests (AC-5).
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.context import QueryContext


class FakeChatModel(BaseChatModel):
    """A chat model that replays a scripted sequence of responses.

    Makes graph structure, tool sequencing and loop termination testable with
    no gateway. Each call returns the next scripted message; once the script is
    exhausted the final message repeats, so a loop that fails to terminate does
    not hang the suite — it stops making progress and trips the budget cap,
    which is itself what several tests assert.

    ``bind_tools`` records what it was given and returns ``self``. Tests that
    assert a model has *no* tools bound (AC-4) read ``bound_tools``.
    """

    responses: list[AIMessage] = []
    calls: list[list[Any]] = []
    bound_tools: list[Any] | None = None

    def __init__(self, responses: list[AIMessage] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        # BaseChatModel is a pydantic model; these are per-instance mutable
        # scratch space rather than declared config, so they are set directly.
        object.__setattr__(self, "responses", list(responses or []))
        object.__setattr__(self, "calls", [])
        object.__setattr__(self, "bound_tools", None)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeChatModel":
        object.__setattr__(self, "bound_tools", list(tools))
        return self

    def _next(self, messages: list[Any]) -> ChatResult:
        self.calls.append(list(messages))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        message = (
            self.responses[index]
            if self.responses
            else AIMessage(content="no scripted response")
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages: list[Any], stop=None, run_manager=None, **kwargs):
        return self._next(messages)

    async def _agenerate(
        self, messages: list[Any], stop=None, run_manager=None, **kwargs
    ):
        return self._next(messages)


class FakeRuntime:
    """Minimal stand-in exposing ``.context``.

    Lets tools and middleware be invoked directly. Direct invocation bypasses
    argument validation, so argument bounds are asserted at the schema layer
    instead.
    """

    def __init__(self, context: QueryContext):
        self.context = context


def tool_call_message(
    name: str, args: dict[str, Any], call_id: str = "c1"
) -> AIMessage:
    """An assistant message requesting one tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


@pytest.fixture
def fake_model():
    """Factory for a scripted fake chat model."""

    def _build(responses: list[AIMessage] | None = None) -> FakeChatModel:
        return FakeChatModel(responses)

    return _build


@pytest.fixture
def fake_runtime():
    """Factory for a runtime carrying a fresh :class:`QueryContext`."""

    def _build(
        *, deadline_in: float = 50.0, tool_budget: int = 5, request_id: str = "test"
    ) -> FakeRuntime:
        return FakeRuntime(
            QueryContext(
                request_id=request_id,
                deadline=time.monotonic() + deadline_in,
                tool_budget=tool_budget,
            )
        )

    return _build
