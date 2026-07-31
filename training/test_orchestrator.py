"""Orchestrator construction and tool-routing tests.

The prompt in ``src/orchestrator.py`` and the tool descriptions in
``src/tools/`` are both doing scored work — together they are what steers the
reasoning brain away from the brief's §10 failure modes — so what they encode is
asserted here rather than assumed.

The descriptions matter as much as the prompt. A model routes by reading them,
and the failure this suite guards against is a counting question sent to
semantic search: article relevance can never yield a record count, so a count
taken from search results is wrong however good the query was.

TODO(build step 3): the behavioural half of this module needs the scripted fake
chat model from ``conftest.py`` — that a counting question actually drives
``afr_count_matches``, that a one-call question does not consume the budget, that
a cross-dataset question composes within the target, and that a structured tool
error is adapted to rather than retried identically (FR-3.6). Those land with
the middleware, which is still a pass-through stub.
"""

from __future__ import annotations

import pytest

from src.tools import TOOLS


def test_registry_is_populated():
    """An empty registry means the agent answers from priors — the worst outcome."""
    assert len(TOOLS) >= 10


def test_tool_names_are_unique():
    names = [tool.name for tool in TOOLS]
    assert len(names) == len(set(names))


def test_every_tool_describes_itself():
    """Routing is done by reading descriptions, so a bare one is a routing bug."""
    for tool in TOOLS:
        assert tool.description and len(tool.description) > 80, tool.name


def test_expected_capabilities_are_present():
    names = {tool.name for tool in TOOLS}
    assert {
        "dataset_coverage",
        "rba_rate_at",
        "rba_decision_stats",
        "rba_decisions",
        "rba_hold_runs",
        "asx_return",
        "asx_summary_stats",
        "asx_resolve_company",
        "afr_count_matches",
        "afr_article_lookup",
        "afr_sentiment_evidence",
        "afr_search",
    } <= names


def _description(name: str) -> str:
    """The tool's description, lowercased with line wrapping collapsed.

    Docstrings are wrapped for readability, so a phrase can straddle a newline.
    These assertions are about what the description says, not how it is laid out.
    """
    raw = next(tool.description for tool in TOOLS if tool.name == name)
    return " ".join(raw.lower().split())


def test_search_description_disclaims_counting():
    """Semantic search must name the counting tool, at the tool layer.

    Steering this in the system prompt alone is weaker: the prompt cannot know
    which tools were actually bound, and the model reads the description last.
    """
    description = _description("afr_search")
    assert "cannot count" in description
    assert "afr_count_matches" in description


def test_counting_description_disclaims_tone():
    description = _description("afr_count_matches")
    assert "cannot judge tone or sentiment" in description


def test_sentiment_tool_does_not_promise_a_label():
    """The label belongs to synthesis, not to a tool (FR-5.2, CON-7)."""
    description = _description("afr_sentiment_evidence")
    assert "evidence" in description
    assert "do not decide the label yourself" in description


@pytest.mark.parametrize(
    "name", ["afr_count_matches", "afr_article_lookup", "asx_return", "rba_rate_at"]
)
def test_tools_declare_argument_schemas(name):
    """Arguments are validated before any data access (FR-3.1)."""
    tool = next(t for t in TOOLS if t.name == name)
    schema = tool.args_schema.model_json_schema()
    assert schema.get("properties")
