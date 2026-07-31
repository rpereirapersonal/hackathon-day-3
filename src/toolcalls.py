"""Recovery for tool calls the server-side parser drops.

The supplied vLLM instance is started with ``--tool-call-parser hermes``, which
expects a JSON object inside ``<tool_call>``. The served Qwen3.6 build instead
emits the pseudo-XML form::

    <tool_call>
    <function=rba_rate_at>
    <parameter=on>2024-06-18</parameter>
    </function>
    </tool_call>

The parser does not recognise it, so vLLM returns ``tool_calls: []`` and the
call text arrives as ordinary message content. To the agent loop that looks
like the model declining to use tools, and it answers from priors instead —
the exact failure the brief's §10 warns about, arriving silently.

The clean fix is server-side (a matching parser, or a build that emits the
JSON form). Until whoever owns the server can restart it, this module reads the
XML form back out of the content and reattaches it as a proper tool call.

Recovery only ever *adds* tool calls that the server missed: when the parser
worked, ``response.tool_calls`` is already populated and the content is left
untouched.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)

_TOOL_CALL_BLOCK = re.compile(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>")
_FUNCTION = re.compile(r"(?s)<function=([^>\s]+)\s*>(.*?)</function>")
_PARAMETER = re.compile(r"(?s)<parameter=([^>\s]+)\s*>(.*?)</parameter>")


def _coerce(raw: str) -> Any:
    """Best-effort typing of an XML parameter value.

    The XML form carries no types, so every value arrives as a string. Tool
    argument schemas are typed, and passing "5" where an int is declared is a
    validation error, so JSON-decode what decodes and leave the rest as text.
    """
    text = raw.strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def parse_xml_tool_calls(content: str) -> list[dict[str, Any]]:
    """Extract tool calls from the pseudo-XML form, if any are present.

    Returns a list of LangChain tool-call dicts, empty when the content holds
    no recognisable call.
    """
    if not content or "<tool_call>" not in content:
        return []

    calls: list[dict[str, Any]] = []
    for block in _TOOL_CALL_BLOCK.findall(content):
        for name, body in _FUNCTION.findall(block):
            args = {key: _coerce(value) for key, value in _PARAMETER.findall(body)}
            calls.append(
                {
                    "name": name.strip(),
                    "args": args,
                    "id": f"call_{uuid.uuid4().hex[:16]}",
                    "type": "tool_call",
                }
            )
    return calls


def recover_tool_calls(message: AIMessage) -> AIMessage:
    """Reattach tool calls the server-side parser failed to extract.

    A no-op when the parser succeeded, or when the content holds no call.
    """
    if getattr(message, "tool_calls", None):
        return message

    content = message.content if isinstance(message.content, str) else ""
    calls = parse_xml_tool_calls(content)
    if not calls:
        return message

    logger.warning(
        "Recovered %d tool call(s) the server parser dropped: %s. The vLLM "
        "tool-call parser does not match this model's output format.",
        len(calls),
        ", ".join(c["name"] for c in calls),
    )

    # The call text is scaffolding, not answer content — strip it so it cannot
    # reach synthesis or the user (FR-5.6).
    remainder = _TOOL_CALL_BLOCK.sub("", content).strip()
    return message.model_copy(update={"content": remainder, "tool_calls": calls})
