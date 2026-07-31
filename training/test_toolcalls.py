"""Tool-call recovery tests.

Covers the workaround in ``src/toolcalls.py`` for the supplied server's
parser mismatch. Worth testing carefully: when recovery fails the agent looks
like it simply chose not to use tools, and answers from priors — the brief's
§10 failure mode, arriving with no error to notice.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from src.toolcalls import parse_xml_tool_calls, recover_tool_calls

# The exact shape observed from the supplied Qwen3.6 build.
XML_CALL = (
    "<tool_call>\n<function=rba_rate_at>\n"
    "<parameter=as_of>\n2024-06-18\n</parameter>\n"
    "</function>\n</tool_call>"
)


def test_parses_the_observed_xml_form():
    calls = parse_xml_tool_calls(XML_CALL)
    assert len(calls) == 1
    assert calls[0]["name"] == "rba_rate_at"
    assert calls[0]["args"] == {"as_of": "2024-06-18"}
    assert calls[0]["type"] == "tool_call"
    assert calls[0]["id"]


def test_parses_multiple_parameters():
    xml = (
        "<tool_call><function=asx_return>"
        "<parameter=ticker>BHP</parameter>"
        "<parameter=start>2023-03-01</parameter>"
        "<parameter=end>2023-03-31</parameter>"
        "</function></tool_call>"
    )
    assert parse_xml_tool_calls(xml)[0]["args"] == {
        "ticker": "BHP",
        "start": "2023-03-01",
        "end": "2023-03-31",
    }


def test_coerces_json_typed_values():
    """Tool schemas are typed, so "5" where an int is declared fails validation."""
    xml = (
        "<tool_call><function=afr_search>"
        "<parameter=query>inflation</parameter>"
        "<parameter=limit>5</parameter>"
        "<parameter=strict>true</parameter>"
        "</function></tool_call>"
    )
    args = parse_xml_tool_calls(xml)[0]["args"]
    assert args["limit"] == 5
    assert args["strict"] is True
    # A bare word is not JSON and must survive as text.
    assert args["query"] == "inflation"


def test_parses_multiple_calls():
    xml = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )
    calls = parse_xml_tool_calls(xml)
    assert [c["name"] for c in calls] == ["a", "b"]


def test_ignores_content_with_no_call():
    assert parse_xml_tool_calls("The cash rate was 4.35 percent.") == []
    assert parse_xml_tool_calls("") == []


def test_recovery_attaches_calls_and_strips_the_scaffolding():
    """The XML must not survive into content — it would reach the user (FR-5.6)."""
    recovered = recover_tool_calls(AIMessage(content=XML_CALL))
    assert len(recovered.tool_calls) == 1
    assert recovered.tool_calls[0]["name"] == "rba_rate_at"
    assert "<tool_call>" not in recovered.content
    assert "<function=" not in recovered.content


def test_recovery_is_a_no_op_when_the_parser_worked():
    """Recovery only ever adds calls the server missed."""
    original = AIMessage(
        content="",
        tool_calls=[
            {"name": "real", "args": {"a": 1}, "id": "x", "type": "tool_call"}
        ],
    )
    recovered = recover_tool_calls(original)
    assert recovered is original
    assert len(recovered.tool_calls) == 1
    assert recovered.tool_calls[0]["name"] == "real"


def test_recovery_leaves_a_plain_answer_untouched():
    """A final answer with no tool call must pass through unchanged.

    This is what lets the loop terminate: if recovery invented a call here, the
    agent could never stop.
    """
    message = AIMessage(content="The cash rate was 4.35 percent.")
    recovered = recover_tool_calls(message)
    assert recovered is message
    assert not recovered.tool_calls
