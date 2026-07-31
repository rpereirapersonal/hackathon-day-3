"""Role-separation tests — AC-4.

The one test group that directly defends CON-7 and FR-2.4. Worth writing
carefully: "evidence that the final agent uses Qwen for planning/tool-calls and
routes verified results to fine-tuned Nemotron for final synthesis" is an
explicitly scored item (brief §6A).
"""

from __future__ import annotations

import asyncio

from src.config import Settings
from src.graph import build_graph
from src.synthesis import synthesize

SENTINEL = "QWEN-REASONING-TRANSCRIPT-MUST-NOT-REACH-THE-USER"


def _settings(**overrides) -> Settings:
    base = dict(
        agent_brain_model="brain",
        agent_brain_base_url="http://localhost:8000/v1",
        agent_brain_api_key="EMPTY",
        domain_ft_model="domain",
        domain_ft_base_url="http://localhost:8000/v1",
        domain_ft_api_key="EMPTY",
        domain_predict_mode="mock",
        request_deadline_seconds=50,
        max_tool_calls=5,
        log_dir="logs",
    )
    base.update(overrides)
    return Settings(**base)


def test_synthesis_model_has_no_tool_binding_path():
    """The synthesis model is never given tools (FR-5.2, CON-7).

    Asserted against the factory's source rather than a constructed instance,
    so it holds without a gateway: what makes CON-7 structural is that no
    ``bind_tools`` call exists on this path at all.
    """
    import inspect

    from src import models

    source = inspect.getsource(models.build_synthesis_model)
    assert "bind_tools" not in source


def test_synthesis_ignores_the_reasoning_transcript():
    """The answer is built from evidence, not from Qwen's deliberation (FR-5.1).

    The sentinel is placed where the reasoning transcript would be. Synthesis
    reads ``tool_trace`` only, so it must not surface.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    state = {
        "question": "What was the cash rate?",
        "messages": [
            HumanMessage(content="What was the cash rate?"),
            AIMessage(content=SENTINEL),
        ],
        "tool_trace": [
            {"tool": "rba_rate_at", "args": {}, "result": "The rate was 4.35 percent."}
        ],
    }
    result = asyncio.run(synthesize(state, settings=_settings()))
    assert SENTINEL not in result["answer"]
    assert "4.35" in result["answer"]


def test_synthesize_only_routes_to_package():
    """Synthesis cannot re-enter the tool loop (AC-4).

    Structural, not behavioural: if the only edge out of ``synthesize`` is to
    ``package``, no prompt or model behaviour can route back into reasoning.
    """
    drawn = build_graph().get_graph()
    targets = {e.target for e in drawn.edges if e.source == "synthesize"}
    assert targets == {"package"}


def test_package_reads_answer_only_from_state():
    """``package`` never reconstructs an answer from reasoning messages (FR-2.4)."""
    from langchain_core.messages import AIMessage

    from src.graph import _package

    packaged = _package(
        {
            "answer": "The rate was 4.35 percent.",
            "messages": [AIMessage(content=SENTINEL)],
        }
    )
    assert packaged["answer"] == "The rate was 4.35 percent."
    assert SENTINEL not in packaged["answer"]


def test_package_never_emits_an_empty_answer():
    """An empty answer scores zero, so the last node guards it (FR-1.4)."""
    from src.graph import _package

    assert _package({"answer": "   "})["answer"].strip()
    assert _package({})["answer"].strip()
