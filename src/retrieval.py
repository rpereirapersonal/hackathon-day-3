"""Reads over the AFR index — exact counting, article lookup, semantic search.

The engine layer. ``src/tools/afr.py`` wraps these functions with bounded,
model-facing schemas; the ingest self-check calls them directly to prove the
index reproduces the published reference counts. One implementation, two
callers, so a tool can never disagree with what ingest verified.

Counting is two-stage, because neither mechanism is sufficient alone. The FTS5
index narrows word-bounded terms — its tokenizer lowercases and splits on
punctuation, so every bounded match is guaranteed to be a candidate — while
substring terms scan with ``LIKE``, since a substring can sit inside a token
where no tokenised query reaches it. Both stages only ever *narrow*: the
``\\b``-anchored regex from ``src/text.py`` decides the final count, matched
against the unescaped article text exactly as the graded convention requires.

Semantic search is a single normalised matmul over the memmapped vectors. At
219,538 x 384 that is tens of milliseconds, which is why there is no FAISS
dependency and no second native library to satisfy on aarch64.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from src import frames
from src.db import aconnect
from src.text import (
    UNDATED,
    TermSpec,
    build_confirm_regex,
    describe_convention,
    fts_match_query,
    like_predicates,
    norm_text,
    period_key,
)

logger = logging.getLogger(__name__)

#: Refuse rather than silently truncate. A term broad enough to select this
#: many candidates is a routing mistake, and answering from a truncated set
#: would report a wrong count with no indication anything was dropped.
MAX_CANDIDATES = 80_000

#: Characters of leading article text returned as evidence. Enough to carry the
#: headline, standfirst and opening argument, bounded so results stay evidence
#: rather than data dumps (FR-3.5).
EXCERPT_CHARS = 1_200

#: Characters scanned for sentiment cues — more than the excerpt, because tone
#: often turns in the second half of a column, still far short of a full body.
CUE_SCAN_CHARS = 6_000


@dataclass(frozen=True)
class CountResult:
    """Outcome of an exact-count query."""

    total: int
    groups: list[dict[str, int | str]] = field(default_factory=list)
    convention: str = ""
    candidates_examined: int = 0


@dataclass(frozen=True)
class Excerpt:
    """One bounded piece of article evidence."""

    headline: str
    publication_date: str
    excerpt: str
    score: float | None = None


def _as_yyyymmdd(value: date | str | None) -> int | None:
    """Normalise a date bound to the integer form stored in ``afr_body``."""
    if value is None:
        return None
    if isinstance(value, date):
        return value.year * 10_000 + value.month * 100 + value.day
    digits = str(value).replace("-", "").strip()
    if len(digits) != 8 or not digits.isdigit():
        raise ValueError(f"unparseable date bound: {value!r}")
    return int(digits)


def _date_clause(
    date_from: date | str | None, date_to: date | str | None, column: str
) -> tuple[str, list[int]]:
    clauses: list[str] = []
    params: list[int] = []
    lower = _as_yyyymmdd(date_from)
    upper = _as_yyyymmdd(date_to)
    if lower is not None:
        clauses.append(f"{column} >= ?")
        params.append(lower)
    if upper is not None:
        clauses.append(f"{column} <= ?")
        params.append(upper)
    return " AND ".join(clauses), params


async def count_matches(
    specs: Sequence[TermSpec],
    *,
    group_by: str = "none",
    date_from: date | str | None = None,
    date_to: date | str | None = None,
    top_n: int = 5,
    db_path: Path | None = None,
) -> CountResult:
    """Count AFR records matching ``specs``, once per record.

    Args:
        specs: Parsed search terms. Bounded terms narrow through FTS5;
            substring terms scan with ``LIKE``.
        group_by: ``"none"``, ``"year"`` or ``"month"``.
        date_from: Inclusive lower bound on publication date.
        date_to: Inclusive upper bound on publication date.
        top_n: How many groups to return, ranked by count descending.
        db_path: Override the index location. Used by the ingest self-check.

    Returns:
        A :class:`CountResult` whose ``total`` is the record count and whose
        ``groups`` are the ``top_n`` largest periods when grouping.

    Raises:
        ValueError: if ``specs`` is empty or the candidate set is too broad.
    """
    if not specs:
        raise ValueError("at least one search term is required")

    fts_query = fts_match_query(specs)
    like_sql, like_params = like_predicates(specs)
    date_sql, date_params = _date_clause(date_from, date_to, "b.pub_date")

    candidate_ids: set[int] = set()
    async with aconnect(db_path) as conn:
        if fts_query is not None:
            sql = (
                "SELECT b.id FROM afr_fts f JOIN afr_body b ON b.id = f.rowid "
                "WHERE afr_fts MATCH ?"
            )
            params: list[object] = [fts_query]
            if date_sql:
                sql += f" AND {date_sql}"
                params.extend(date_params)
            async with conn.execute(sql, params) as cursor:
                async for row in cursor:
                    candidate_ids.add(row[0])

        if like_sql:
            # Substring terms cannot use the index. The date bound is applied
            # in the same statement so SQLite restricts rows before running the
            # scan, which is what keeps a year-scoped question cheap.
            sql = "SELECT b.id FROM afr_body b WHERE "
            conditions = [f"({like_sql})"]
            params = list(like_params)
            if date_sql:
                conditions.append(date_sql)
                params.extend(date_params)
            async with conn.execute(sql + " AND ".join(conditions), params) as cursor:
                async for row in cursor:
                    candidate_ids.add(row[0])

        if len(candidate_ids) > MAX_CANDIDATES:
            raise ValueError(
                f"{len(candidate_ids):,} candidate records for these terms, "
                f"above the {MAX_CANDIDATES:,} limit. Narrow the date range or "
                "use a more specific term."
            )

        regex = build_confirm_regex(specs)
        total = 0
        per_period: dict[str, int] = {}
        for chunk in _chunked(sorted(candidate_ids), 2_000):
            placeholders = ",".join("?" * len(chunk))
            async with conn.execute(
                f"SELECT pub_date, body FROM afr_body WHERE id IN ({placeholders})",
                chunk,
            ) as cursor:
                async for row in cursor:
                    if regex.search(row["body"]) is None:
                        continue
                    total += 1
                    # An undated record still counts toward the total but
                    # cannot honestly be placed in a period.
                    if group_by != "none" and row["pub_date"] != UNDATED:
                        stamp = str(row["pub_date"])
                        as_date = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
                        key = period_key(as_date, group_by)
                        per_period[key] = per_period.get(key, 0) + 1

    groups: list[dict[str, int | str]] = []
    if group_by != "none":
        ranked = sorted(per_period.items(), key=lambda kv: (-kv[1], kv[0]))
        groups = [{"period": k, "count": v} for k, v in ranked[:top_n]]

    return CountResult(
        total=total,
        groups=groups,
        convention=describe_convention(specs),
        candidates_examined=len(candidate_ids),
    )


def _chunked(items: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def fetch_body(record_id: int, *, db_path: Path | None = None) -> str:
    """Return the full stored text of one record, by id."""
    async with aconnect(db_path) as conn:
        async with conn.execute(
            "SELECT body FROM afr_body WHERE id = ?", (record_id,)
        ) as cursor:
            row = await cursor.fetchone()
    return "" if row is None else row["body"]


async def lookup_article(
    headline: str, publication_date: date | str, *, db_path: Path | None = None
) -> dict[str, object] | None:
    """Fetch one article by exact headline and publication date.

    Headlines are compared after normalisation, so case, punctuation and smart
    quotes do not decide whether a question finds its article.

    Returns:
        A mapping with the headline, date, excerpt, character count and
        ``duplicate_rows`` — the corpus holds 37,048 repeated
        ``(headline, date)`` pairs, so the count is reported rather than
        hidden. ``None`` when nothing matches.
    """
    meta = frames.afr_meta()
    stamp = _as_yyyymmdd(publication_date)
    target = norm_text(headline).strip()
    hits = meta[(meta["headline_norm"] == target) & (meta["pub_date"] == stamp)]
    if hits.empty:
        return None

    first = hits.iloc[0]
    body = await fetch_body(int(first["id"]), db_path=db_path)
    return {
        "headline": str(first["headline"]),
        "publication_date": _iso(int(first["pub_date"])),
        "excerpt": body[:EXCERPT_CHARS],
        "char_count": len(body),
        "duplicate_rows": int(len(hits)),
        "record_id": int(first["id"]),
    }


def suggest_headlines(headline: str, limit: int = 5) -> list[dict[str, str]]:
    """Return near matches for a headline that did not resolve exactly.

    A miss must not look like absent data. Scoring is token overlap against the
    normalised headline — cheap, and enough to surface a date typo or a
    truncated title.
    """
    meta = frames.afr_meta()
    wanted = set(norm_text(headline).split())
    if not wanted:
        return []

    # Prefilter on the longest token before scoring. Scoring all 219,538
    # headlines with a per-row Python callable costs seconds, and this is the
    # miss path — the caller is already one failed lookup deep.
    anchor = max(wanted, key=len)
    candidates = meta[meta["headline_norm"].str.contains(anchor, regex=False)]
    if candidates.empty:
        return []

    # Not `_overlap`: itertuples renames any leading-underscore column to a
    # positional `_1`, and the attribute access below would fail.
    scored = candidates.assign(
        overlap_score=candidates["headline_norm"].map(
            lambda h: len(wanted & set(h.split())) / len(wanted)
        )
    )
    return [
        {
            "headline": str(row.headline),
            "publication_date": _iso(int(row.pub_date)),
            "overlap": f"{row.overlap_score:.2f}",
        }
        for row in scored.nlargest(limit, "overlap_score").itertuples()
        if row.overlap_score > 0
    ]


async def sentiment_evidence(
    headline: str, publication_date: date | str, *, db_path: Path | None = None
) -> dict[str, object] | None:
    """Return an article plus the directional language it contains.

    Deliberately returns evidence and no label: assigning
    positive / negative / mixed and the market direction is the fine-tuned
    synthesis model's job, not a tool's (FR-5.2, CON-7).
    """
    from src.text import HEDGE_CUES, NEGATIVE_CUES, POSITIVE_CUES, find_cues

    article = await lookup_article(headline, publication_date, db_path=db_path)
    if article is None:
        return None

    body = await fetch_body(int(article["record_id"]), db_path=db_path)
    window = body[:CUE_SCAN_CHARS]
    positive = find_cues(window, POSITIVE_CUES)
    negative = find_cues(window, NEGATIVE_CUES)
    return {
        **article,
        "positive_cues": positive,
        "negative_cues": negative,
        "hedge_cues": find_cues(window, HEDGE_CUES),
        "cue_balance": len(positive) - len(negative),
        "scanned_chars": len(window),
    }


async def search(
    query: str,
    *,
    k: int = 5,
    date_from: date | str | None = None,
    date_to: date | str | None = None,
    db_path: Path | None = None,
) -> list[Excerpt]:
    """Rank articles by semantic similarity to ``query``.

    The date window is applied as a mask *before* scoring, so a windowed query
    scores a subset rather than the whole corpus.

    Returns:
        Up to ``k`` bounded excerpts, most similar first.
    """
    from src.embeddings import aembed_query

    meta = frames.afr_meta()
    vectors = frames.afr_vectors()

    mask = np.ones(len(meta), dtype=bool)
    lower = _as_yyyymmdd(date_from)
    upper = _as_yyyymmdd(date_to)
    stamps = meta["pub_date"].to_numpy()
    if lower is not None:
        mask &= stamps >= lower
    if upper is not None:
        mask &= stamps <= upper
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return []

    vector = await aembed_query(norm_text(query).strip())
    scores = await asyncio.to_thread(
        lambda: np.asarray(vectors[positions] @ vector, dtype=np.float32)
    )
    take = min(k, scores.size)
    top = positions[np.argsort(scores)[::-1][:take]]

    results: list[Excerpt] = []
    for position in top:
        row = meta.iloc[int(position)]
        body = await fetch_body(int(row["id"]), db_path=db_path)
        results.append(
            Excerpt(
                headline=str(row["headline"]),
                publication_date=_iso(int(row["pub_date"])),
                excerpt=body[:EXCERPT_CHARS],
                score=float(scores[np.flatnonzero(positions == position)[0]]),
            )
        )
    return results


def _iso(stamp: int) -> str:
    text = str(stamp)
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
