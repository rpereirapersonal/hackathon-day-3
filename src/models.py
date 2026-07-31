"""Model factories — one per role, and the two are never interchangeable.

The reasoning brain is bound to tools. The synthesis model is constructed with
no tool-binding path at all, which is what makes CON-7 ("Nemotron must not be
the primary tool-calling model") structural rather than aspirational.

Both models are reached through an OpenAI-compatible gateway, so the same
client class serves both with different aliases, base URLs and credentials
(BLK-5 has no code impact — a gateway is a base URL plus a key either way).
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from src.config import Settings, load_settings
from src.toolcalls import recover_tool_calls

logger = logging.getLogger(__name__)

# Planning and synthesis both want reproducible output, not creative variation.
# Determinism is enforced in code (FR-3.4), but a near-zero temperature keeps
# tool-argument selection stable across identical questions.
_TEMPERATURE = 0.0

# The served Qwen3.6 build emits a chain-of-thought preamble regardless of the
# server's ``--override-generation-config {"enable_thinking": false}``; the
# switch that actually takes effect is this per-request chat-template argument.
# It matters twice over. In synthesis the preamble is internal scaffolding that
# must not reach the answer (FR-5.6), and measured at ~700 tokens against ~21
# for the answer itself. In reasoning it crowds out the tool call entirely — with
# thinking on, the model spends its output deliberating and returns no
# tool_calls at all.
#
# Harmless if the gateway ignores unknown template kwargs, so it is safe to send
# to a non-Qwen endpoint — which the fine-tuned model may well be.
_NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


class _ToolCallRepairChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` that repairs tool calls the server-side parser dropped.

    Implemented as a subclass rather than ``model | recover`` because piping
    returns a ``RunnableSequence``, which has no ``bind_tools`` — and
    ``create_agent`` needs to bind the tool set. Overriding generation keeps
    this a ``BaseChatModel``, so the agent, the middleware and the tests all
    treat it as an ordinary chat model.

    See ``src/toolcalls.py`` for the parser mismatch this works around.
    """

    def _repair(self, result):
        for generation in result.generations:
            message = getattr(generation, "message", None)
            if isinstance(message, AIMessage):
                generation.message = recover_tool_calls(message)
        return result

    def _generate(self, *args, **kwargs):
        return self._repair(super()._generate(*args, **kwargs))

    async def _agenerate(self, *args, **kwargs):
        return self._repair(await super()._agenerate(*args, **kwargs))


def build_reasoning_model(settings: Settings | None = None) -> ChatOpenAI:
    """The supplied Qwen ``agent-brain``. Plans and requests tools (FR-2.1).

    Never fine-tuned (FR-2.3, CON-8).
    """
    cfg = settings or load_settings()
    return _ToolCallRepairChatOpenAI(
        model=cfg.agent_brain_model,
        base_url=cfg.agent_brain_base_url,
        api_key=cfg.agent_brain_api_key,
        temperature=_TEMPERATURE,
        extra_body=dict(_NO_THINKING),
        # TODO(build step 3): per-stage timeout and a shared async httpx
        # client, once the gateway's real latency is known (BLK-3, NFR-1.3).
    )


def build_synthesis_model(settings: Settings | None = None) -> ChatOpenAI:
    """The team's fine-tuned Nemotron. Writes the final answer (FR-5.1).

    Bound with no tools, by construction. Callers must not attach any
    (FR-5.2, CON-7) — the role-separation test asserts this (AC-4).
    """
    cfg = settings or load_settings()
    return ChatOpenAI(
        model=cfg.domain_ft_model,
        base_url=cfg.domain_ft_base_url,
        api_key=cfg.domain_ft_api_key,
        temperature=_TEMPERATURE,
        extra_body=dict(_NO_THINKING),
        # max_tokens is bound at the call site in synthesis.py, which owns the
        # ~15s slice of the latency budget (NFR-1.3).
    )
