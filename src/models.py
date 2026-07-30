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

from langchain_openai import ChatOpenAI

from src.config import Settings, load_settings

logger = logging.getLogger(__name__)

# Planning and synthesis both want reproducible output, not creative variation.
# Determinism is enforced in code (FR-3.4), but a near-zero temperature keeps
# tool-argument selection stable across identical questions.
_TEMPERATURE = 0.0


def build_reasoning_model(settings: Settings | None = None) -> ChatOpenAI:
    """The supplied Qwen ``agent-brain``. Plans and requests tools (FR-2.1).

    Never fine-tuned (FR-2.3, CON-8).
    """
    cfg = settings or load_settings()
    return ChatOpenAI(
        model=cfg.agent_brain_model,
        base_url=cfg.agent_brain_base_url,
        api_key=cfg.agent_brain_api_key,
        temperature=_TEMPERATURE,
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
        # TODO(build step 3): bounded max_tokens to hold the synthesis stage
        # inside its ~15s slice of the latency budget (NFR-1.3).
    )
