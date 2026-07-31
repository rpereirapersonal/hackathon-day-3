"""Pandas frames over the ingested artifacts, loaded once per process.

Every structured fact the tools return is computed from one of these frames,
in pandas, deterministically (FR-3.4). The frames are small enough to hold
resident: RBA is 175 rows, ASX is 31,932, and the AFR frame carries metadata
only — headline, date and identifiers — with the 780 MB of article text left
in SQLite where ``LIKE`` and FTS5 can reach it without loading it into memory.

Concurrency: these are read-only lookups, loaded before the service reports
healthy (NFR-1.4) and never reassigned or mutated afterwards, so they are not
the mutable module-level request state NFR-3.2 prohibits. Tools must not write
to them — take a copy first if a derived column is needed.

Row alignment across the three AFR artifacts is load-bearing and asserted at
ingest: ``afr_meta`` row *i* describes ``afr_body.id == i + 1`` and
``afr_vectors[i]``.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_data_paths

logger = logging.getLogger(__name__)


class ArtifactMissingError(RuntimeError):
    """Raised when an ingested artifact is absent.

    Names the rebuild command, because that is always the fix and a tool that
    says so lets the reasoning brain report a real limitation rather than
    retry blindly.
    """


def _require(path: Path) -> Path:
    if not path.exists():
        raise ArtifactMissingError(
            f"Ingested artifact not found at {path}. Build it once with "
            "`python -m src.ingest` before serving."
        )
    return path


@lru_cache(maxsize=1)
def afr_meta() -> pd.DataFrame:
    """AFR record metadata: ``id, headline, headline_norm, pub_date, year, month``.

    219,538 rows, one per record, in file order. Not deduplicated — the corpus
    holds 37,048 repeated ``(headline, date)`` pairs and every published
    reference count includes them.
    """
    return pd.read_parquet(_require(load_data_paths().afr_meta))


@lru_cache(maxsize=1)
def asx() -> pd.DataFrame:
    """ASX daily prices: ``ticker, company, date, open, high, low, close, volume``.

    All 18 instruments concatenated and sorted by ``(ticker, date)``.
    """
    return pd.read_parquet(_require(load_data_paths().asx))


@lru_cache(maxsize=1)
def rba() -> pd.DataFrame:
    """RBA decisions: ``effective_date, change_pp, cash_rate_target, direction``.

    Sorted ascending by date. ``direction`` is one of ``increase``,
    ``decrease`` or ``hold``, derived from the sign of ``change_pp``.
    """
    return pd.read_parquet(_require(load_data_paths().rba))


@lru_cache(maxsize=1)
def coverage() -> pd.DataFrame:
    """Per-dataset ``dataset, row_count, min_date, max_date``.

    The grounding for coverage questions: one public question turns entirely
    on AFR and ASX ending in 2021 while RBA runs to late 2023, and answering
    it from the model's priors instead of this frame is a fabrication.
    """
    return pd.read_parquet(_require(load_data_paths().coverage))


@lru_cache(maxsize=1)
def afr_vectors() -> np.ndarray:
    """Memmapped ``float32[n_records, dim]`` article embeddings, L2-normalised.

    Memmapped read-only, so the 337 MB is paged in by the OS rather than
    copied into the process, and normalised at ingest so cosine similarity is
    a single matmul with no per-query division.
    """
    path = _require(load_data_paths().afr_vectors)
    return np.load(path, mmap_mode="r")


def warm() -> None:
    """Load every artifact.

    Called at application import so indexing and page-in costs are paid before
    ``/health`` reports 200, never inside a request (NFR-1.4). Failures are
    logged rather than raised: a missing artifact must degrade ``/query``
    answers, not take down the health gate (NFR-4.2).
    """
    for name, loader in (
        ("afr_meta", afr_meta),
        ("asx", asx),
        ("rba", rba),
        ("coverage", coverage),
        ("afr_vectors", afr_vectors),
    ):
        try:
            loaded = loader()
        except ArtifactMissingError as exc:
            logger.warning("Artifact %s unavailable: %s", name, exc)
        else:
            logger.info("Artifact %s loaded: %d rows", name, len(loaded))
