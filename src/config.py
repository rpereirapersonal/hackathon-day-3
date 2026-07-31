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
from pathlib import Path


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or unusable."""


#: Organizer-supplied environment names accepted in place of our own (DEP-8).
#:
#: The supplied cluster exports these before starting the agent, and failing
#: fast on a variable that *is* set — merely under a different name — would be
#: a self-inflicted outage on a correctly provisioned host. Our own name is
#: checked first, so an explicit local override is never silently ignored.
ALIASES: dict[str, tuple[str, ...]] = {
    "AGENT_BRAIN_MODEL": ("BRAIN_MODEL",),
    "AGENT_BRAIN_BASE_URL": ("LITELLM_BASE_URL", "LITELLM_URL"),
    "AGENT_BRAIN_API_KEY": ("LITELLM_KEY",),
    # Both models are reached through the one gateway unless told otherwise.
    "DOMAIN_FT_BASE_URL": ("LITELLM_BASE_URL", "LITELLM_URL"),
    "DOMAIN_FT_API_KEY": ("LITELLM_KEY",),
    "MAX_TOOL_CALLS": ("MAX_AGENT_STEPS",),
}


def _lookup(name: str) -> str:
    """The value of ``name``, or of the first organizer alias that is set."""
    for candidate in (name, *ALIASES.get(name, ())):
        value = os.getenv(candidate, "").strip()
        if value:
            return value
    return ""


def _required(name: str) -> str:
    value = _lookup(name)
    if not value:
        alternatives = ALIASES.get(name, ())
        also = f" (or {' / '.join(alternatives)})" if alternatives else ""
        raise ConfigError(
            f"{name}{also} is not set. See .env.example; export credentials in "
            "the shell rather than committing them."
        )
    return value


def _int(name: str, default: int) -> int:
    raw = _lookup(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class DataPaths:
    """Locations of the source datasets and the ingested artifacts.

    Separate from :class:`Settings` on purpose. ``python -m src.ingest`` runs
    before any model gateway exists and must not be blocked by a missing
    ``AGENT_BRAIN_API_KEY``, so the data layer resolves its own configuration
    without touching the credential-bearing settings.
    """

    source_dir: Path
    data_dir: Path
    embedding_model_name: str
    embedding_cache_dir: Path

    # Artifact names are fixed rather than configurable: they are written and
    # read by this repository only, and a mismatch between the two would be a
    # silent empty-result failure at evaluation time.
    @property
    def afr_db(self) -> Path:
        return self.data_dir / "afr.sqlite"

    @property
    def afr_meta(self) -> Path:
        return self.data_dir / "afr_meta.parquet"

    @property
    def afr_vectors(self) -> Path:
        return self.data_dir / "afr_vectors.npy"

    @property
    def asx(self) -> Path:
        return self.data_dir / "asx.parquet"

    @property
    def rba(self) -> Path:
        return self.data_dir / "rba.parquet"

    @property
    def coverage(self) -> Path:
        return self.data_dir / "coverage.parquet"

    @property
    def afr_source(self) -> Path:
        return self.source_dir / "AFR"

    @property
    def asx_source(self) -> Path:
        return self.source_dir / "ASX"

    @property
    def rba_source(self) -> Path:
        return self.source_dir / "RBA Rates" / "RBA-rates.csv"


def load_data_paths() -> DataPaths:
    """Resolve data-layer paths from the environment, with usable defaults.

    Unlike the model settings, these default rather than fail fast: a wrong
    path surfaces immediately as a missing-file error at ingest or import,
    whereas a wrong model alias would silently produce a scored answer from
    the wrong model.
    """
    return DataPaths(
        source_dir=Path(os.getenv("SOURCE_DATA_DIR", "./data set")),
        data_dir=Path(os.getenv("DATA_DIR", "./data")),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"
        ).strip(),
        embedding_cache_dir=Path(
            os.getenv("EMBEDDING_CACHE_DIR", "./.cache/fastembed")
        ),
    )


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

    # Backstop, not the target. The orchestrator prompt aims at <=3 calls, but
    # two of the public questions legitimately need three different datasets
    # (article + cash rate + basket return), which leaves a cap of 3 with no
    # room for the single adaptive retry FR-3.6 promises. The brief's warning
    # is about exceeding 5.
    max_tool_calls: int

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
        max_tool_calls=_int("MAX_TOOL_CALLS", 5),
        log_dir=os.getenv("LOG_DIR", "logs").strip() or "logs",
    )
