"""Environment configuration.

Every value is environment-supplied. ``.env.example`` carries placeholders
only; real credentials are exported in the shell and never committed
(NFR-6.1, NFR-6.2, CON-6).

No defaults exist for model aliases or base URLs. A silent default there would
point the submission at the wrong model, which is a scored failure rather than
an inconvenience.

TODO(build step 1): validate on import and fail fast, per architecture.md §9.
Currently validation happens in ``load_settings()`` so that importing this
module — and therefore importing the orchestrator in tests — does not require
a populated environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or unusable."""


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set. See .env.example; export credentials in the "
            "shell rather than committing them."
        )
    return value


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one process."""

    # Reasoning brain — the supplied Qwen agent-brain (BLK-3).
    agent_brain_model: str
    agent_brain_base_url: str
    agent_brain_api_key: str

    # Domain synthesis — the team's fine-tuned Nemotron (BLK-4).
    domain_ft_model: str
    domain_ft_base_url: str
    domain_ft_api_key: str

    # ``mock`` for pre-adapter integration testing, ``llm`` for real
    # inference. Shipping in ``mock`` forfeits the fine-tuned-model evidence
    # entirely (FR-5.5, AC-12).
    domain_predict_mode: str

    request_deadline_seconds: int
    max_tool_calls: int

    # TODO(build step 4): data-layer paths — DATA_DIR, DB_PATH, INDEX_PATH,
    # EMBEDDING_MODEL_NAME, EMBEDDING_CACHE_DIR (BLK-2).

    log_dir: str

    @property
    def is_mock_synthesis(self) -> bool:
        return self.domain_predict_mode.lower() == "mock"


def load_settings() -> Settings:
    """Read and validate configuration from the environment.

    Raises:
        ConfigError: if a required variable is missing or malformed.
    """
    return Settings(
        agent_brain_model=_required("AGENT_BRAIN_MODEL"),
        agent_brain_base_url=_required("AGENT_BRAIN_BASE_URL"),
        agent_brain_api_key=_required("AGENT_BRAIN_API_KEY"),
        domain_ft_model=_required("DOMAIN_FT_MODEL"),
        domain_ft_base_url=_required("DOMAIN_FT_BASE_URL"),
        domain_ft_api_key=_required("DOMAIN_FT_API_KEY"),
        domain_predict_mode=os.getenv("DOMAIN_PREDICT_MODE", "mock").strip(),
        request_deadline_seconds=_int("REQUEST_DEADLINE_SECONDS", 50),
        max_tool_calls=_int("MAX_TOOL_CALLS", 3),
        log_dir=os.getenv("LOG_DIR", "logs").strip() or "logs",
    )
