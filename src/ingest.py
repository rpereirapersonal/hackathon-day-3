"""Build the AFR index and the structured frames from the source files.

The single adapter boundary between raw data and the tool layer. Every fact
about raw file layout — field names, filename conventions, the RBA date format,
the signed change column, the empty AFR file — lives here and nowhere else, so
the tools above it never parse a source file.

Runs ahead of serving, never inside a request (NFR-1.4). Reads the sources
without modifying them (CON-3).

Usage::

    python -m src.ingest                     # everything
    python -m src.ingest --stage structured  # RBA + ASX + coverage only, seconds
    python -m src.ingest --stage afr         # AFR body, FTS5 index and metadata
    python -m src.ingest --stage embeddings  # the slow pass, resumable
    python -m src.ingest --verify            # re-run the reference-count check

Stages are separable because they cost wildly different amounts: the structured
frames build in seconds, the AFR index in a couple of minutes, and encoding
219,538 articles takes tens of minutes on the target host. Splitting them means
a failed embedding run never costs the rest of the work, and the embedding pass
resumes from wherever it stopped.

The run ends by asserting the four published reference counts against the index
it just built, and exits non-zero if any of them is wrong. Ingest that silently
applies a different matching convention is the most expensive failure available
here — every downstream number inherits it, with nothing to signal the error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DataPaths, load_data_paths
from src.text import UNDATED, norm_text, parse_terms, pub_date_stamp, raw_text

logger = logging.getLogger("ingest")

#: Records inserted per transaction batch. Large enough to keep SQLite from
#: fsyncing constantly, small enough that peak memory stays flat.
INSERT_BATCH = 5_000

#: Leading characters of the stored body used as embedding input — the
#: headline, standfirst and opening sentences. One vector per article, derived
#: from the stored text rather than the source file, so the embedding pass can
#: resume from the database alone and row alignment cannot drift.
#:
#: Measured on the development host: encoder throughput is roughly inverse to
#: this length, at 32 articles/s over 600 characters against 66 over 320. Since
#: the whole corpus must be encoded, that is the difference between a two-hour
#: pass and a one-hour one, and the lead of an AFR article carries its subject.
#: Semantic search is insurance for topical questions here, not the path any
#: exact answer depends on, so the shorter window is the right trade.
EMBED_INPUT_CHARS = 320

#: Rows encoded between memmap flushes and progress lines.
EMBED_CHUNK = 2_000

#: The published reference counts, as (terms, scope, expected). These are
#: assertions about the *convention*, keyed by search arguments — no question id
#: appears here, and none may (CON-9, AC-13).
REFERENCE_COUNTS: tuple[tuple[list[str], str, dict[str, object], int], ...] = (
    (["unemployment"], "peak year 2020", {"date_from": "20200101", "date_to": "20201231"}, 1_452),
    (["unemployment"], "peak month May 2020", {"date_from": "20200501", "date_to": "20200531"}, 218),
    (["QBE"], "2021", {"date_from": "20210101", "date_to": "20211231"}, 369),
    (
        ["interest rate*", "cash rate*", "rate cut*", "rate hike*", "RBA"],
        "rate pattern 2019",
        {"date_from": "20190101", "date_to": "20191231"},
        3_181,
    ),
)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def check_fts5() -> None:
    """Fail early if the interpreter's SQLite lacks FTS5.

    FTS5 is a compile-time option, not a guarantee. Discovering its absence
    here — with a message naming the cause — is far better than a bare
    ``OperationalError`` partway through a multi-minute build.
    """
    probe = sqlite3.connect(":memory:")
    try:
        probe.execute("CREATE VIRTUAL TABLE probe USING fts5(body)")
    except sqlite3.OperationalError as exc:
        raise SystemExit(
            "This Python's SQLite was built without FTS5, which the AFR index "
            f"requires ({exc}). Install a build with FTS5 enabled, or rebuild "
            "SQLite with -DSQLITE_ENABLE_FTS5."
        ) from exc
    finally:
        probe.close()


def check_sources(paths: DataPaths) -> None:
    missing = [
        str(p)
        for p in (paths.afr_source, paths.asx_source, paths.rba_source)
        if not p.exists()
    ]
    if missing:
        raise SystemExit(
            "Source data not found: " + ", ".join(missing) + ". Set "
            "SOURCE_DATA_DIR to the directory holding AFR/, ASX/ and RBA Rates/."
        )


# ---------------------------------------------------------------------------
# AFR
# ---------------------------------------------------------------------------
def iter_afr_records(afr_dir: Path) -> Iterator[dict[str, object]]:
    """Yield every AFR record in a stable order.

    Files are visited in sorted filename order and lines in file order, which
    is what makes ``afr_body.id``, ``afr_meta`` row order and ``afr_vectors``
    row order agree. Blank lines are skipped — most files end with one, and one
    file (``AFR_20150201-20150201.jsonl``) contains nothing else.
    """
    for path in sorted(afr_dir.glob("AFR_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"{path.name} line {line_number} is not valid JSON: {exc}"
                    ) from exc


def build_afr(paths: DataPaths) -> int:
    """Build ``afr.sqlite`` and ``afr_meta.parquet``. Returns the record count.

    Written to a temporary path and moved into place on success, so a run that
    dies halfway can never leave a partially indexed database to be served.
    """
    target = paths.afr_db
    staging = target.with_suffix(".sqlite.building")
    staging.unlink(missing_ok=True)

    logger.info("Building AFR index from %s", paths.afr_source)
    connection = sqlite3.connect(staging)
    ids: list[int] = []
    headlines: list[str] = []
    headlines_norm: list[str] = []
    stamps: list[int] = []
    char_counts: list[int] = []

    try:
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;
            CREATE TABLE afr_body(
                id       INTEGER PRIMARY KEY,
                pub_date INTEGER NOT NULL,
                body     TEXT    NOT NULL
            );
            """
        )

        batch: list[tuple[int, int, str]] = []
        undated = 0
        for index, record in enumerate(iter_afr_records(paths.afr_source), start=1):
            body = raw_text(record)
            stamp = pub_date_stamp(record.get("PUBLICATIONDATE"))
            undated += stamp == UNDATED
            headline = "" if record.get("HEADLINE") is None else str(record["HEADLINE"])

            batch.append((index, stamp, body))
            ids.append(index)
            headlines.append(headline)
            headlines_norm.append(norm_text(headline).strip())
            stamps.append(stamp)
            char_counts.append(len(body))

            if len(batch) >= INSERT_BATCH:
                connection.executemany("INSERT INTO afr_body VALUES (?,?,?)", batch)
                batch.clear()
                if index % 50_000 == 0:
                    logger.info("  %d records inserted", index)

        if batch:
            connection.executemany("INSERT INTO afr_body VALUES (?,?,?)", batch)

        if undated:
            logger.warning(
                "%d of %d records carry no usable PUBLICATIONDATE. They are kept "
                "and count toward unscoped totals, but no date-bounded or "
                "grouped query can include them.",
                undated,
                len(ids),
            )

        logger.info("Indexing %d records (FTS5 + date index)", len(ids))
        connection.executescript(
            """
            CREATE INDEX afr_body_pub_date ON afr_body(pub_date);
            CREATE VIRTUAL TABLE afr_fts USING fts5(
                body, content='afr_body', content_rowid='id'
            );
            INSERT INTO afr_fts(afr_fts) VALUES('rebuild');
            """
        )
        connection.commit()
    finally:
        connection.close()

    target.unlink(missing_ok=True)
    staging.replace(target)

    meta = pd.DataFrame(
        {
            "id": pd.array(ids, dtype="int32"),
            "headline": headlines,
            "headline_norm": headlines_norm,
            "pub_date": pd.array(stamps, dtype="int32"),
            "year": pd.array([s // 10_000 for s in stamps], dtype="int16"),
            "month": pd.array([(s // 100) % 100 for s in stamps], dtype="int8"),
            "char_count": pd.array(char_counts, dtype="int32"),
        }
    )
    meta.to_parquet(paths.afr_meta, index=False, compression="zstd")
    logger.info("Wrote %s (%d rows)", paths.afr_meta, len(meta))
    return len(meta)


def build_embeddings(paths: DataPaths) -> int:
    """Encode one vector per article into a memmapped ``.npy``. Resumable.

    Progress is tracked in a sidecar file holding the number of completed rows,
    so a run interrupted at 80% resumes there rather than starting over. The
    encoded text is the leading ``EMBED_INPUT_CHARS`` of the stored body, read
    back from SQLite by id — deriving it from the database rather than the
    source files is what guarantees row *i* of the vector file describes
    ``afr_body.id == i + 1``.
    """
    from src.embeddings import embed_documents, embedding_dim

    meta = pd.read_parquet(paths.afr_meta, columns=["id"])
    total = len(meta)
    dim = embedding_dim()
    progress_path = paths.afr_vectors.with_suffix(".progress")

    done = 0
    if paths.afr_vectors.exists() and progress_path.exists():
        existing = np.load(paths.afr_vectors, mmap_mode="r")
        if existing.shape == (total, dim):
            done = int(progress_path.read_text().strip() or 0)
            logger.info("Resuming embeddings at row %d of %d", done, total)
        else:
            logger.warning(
                "Existing vector file has shape %s, expected %s - rebuilding",
                existing.shape,
                (total, dim),
            )
        del existing

    if done == 0:
        # np.lib.format.open_memmap writes the header up front, so the file is
        # a valid .npy from the first byte and can be reopened on resume.
        vectors = np.lib.format.open_memmap(
            paths.afr_vectors, mode="w+", dtype=np.float32, shape=(total, dim)
        )
    else:
        vectors = np.lib.format.open_memmap(paths.afr_vectors, mode="r+")

    connection = sqlite3.connect(f"file:{paths.afr_db.as_posix()}?mode=ro", uri=True)
    try:
        while done < total:
            upper = min(done + EMBED_CHUNK, total)
            rows = connection.execute(
                "SELECT substr(body, 1, ?) FROM afr_body "
                "WHERE id > ? AND id <= ? ORDER BY id",
                (EMBED_INPUT_CHARS, done, upper),
            ).fetchall()
            texts = [row[0] or " " for row in rows]
            vectors[done : done + len(texts)] = embed_documents(texts)
            done += len(texts)
            vectors.flush()
            progress_path.write_text(str(done))
            logger.info("  embedded %d/%d (%.1f%%)", done, total, 100 * done / total)
    finally:
        connection.close()
        del vectors

    progress_path.unlink(missing_ok=True)
    logger.info("Wrote %s (%d x %d)", paths.afr_vectors, total, dim)
    return total


# ---------------------------------------------------------------------------
# ASX and RBA
# ---------------------------------------------------------------------------
def build_asx(paths: DataPaths) -> pd.DataFrame:
    """Concatenate the per-company price files into one frame.

    The company name comes from the filename and the instrument identifier from
    the ``ticker`` field, because they differ — ``Qantas-ASX-…`` holds
    ``QAN.AX``. Both are kept so a question naming either resolves.
    """
    records = []
    for path in sorted(paths.asx_source.glob("*.jsonl")):
        company = path.stem.split("-ASX-")[0]
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                row["company"] = company
                records.append(row)

    frame = pd.DataFrame.from_records(records)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame.sort_values(["ticker", "date"], ignore_index=True)
    frame = frame[
        ["ticker", "company", "date", "open", "high", "low", "close", "volume"]
    ]
    frame.to_parquet(paths.asx, index=False, compression="zstd")
    logger.info(
        "Wrote %s (%d rows, %d tickers)", paths.asx, len(frame), frame.ticker.nunique()
    )
    return frame


def build_rba(paths: DataPaths) -> pd.DataFrame:
    """Parse the RBA decision CSV into a typed frame.

    The CSV is preferred over the sibling JSONL, which carries a UTF-8 BOM that
    corrupts its first key. ``Change % points`` arrives as a signed string
    (``+0.25``, ``0.00``, ``-0.25``); ``direction`` is derived from its sign so
    no downstream code re-decides what counts as a hold.
    """
    frame = pd.read_csv(paths.rba_source)
    frame = frame.rename(
        columns={
            "Effective Date": "effective_date",
            "Change % points": "change_pp",
            "Cash rate target%": "cash_rate_target",
        }
    )
    frame["effective_date"] = pd.to_datetime(
        frame["effective_date"], format="%d %b %Y"
    ).dt.date
    frame["change_pp"] = (
        frame["change_pp"].astype(str).str.replace("+", "", regex=False).astype(float)
    )
    frame["cash_rate_target"] = frame["cash_rate_target"].astype(float)
    frame["direction"] = np.select(
        [frame["change_pp"] > 0, frame["change_pp"] < 0],
        ["increase", "decrease"],
        default="hold",
    )
    frame = frame.sort_values("effective_date", ignore_index=True)
    frame.to_parquet(paths.rba, index=False, compression="zstd")
    logger.info("Wrote %s (%d decisions)", paths.rba, len(frame))
    return frame


def build_coverage(
    paths: DataPaths, *, afr_rows: int, asx: pd.DataFrame, rba: pd.DataFrame
) -> pd.DataFrame:
    """Record each dataset's row count, entity count and date span.

    This is the evidence behind any answer about what the data can and cannot
    support. Without it a coverage question gets answered from the model's
    priors, which is a fabrication whichever way it lands.

    ``entity_count`` and ``rows_per_entity`` exist because a dimensions
    question asks how many instruments there are and how many rows each holds,
    not merely how many rows there are in total. ``rows_per_entity`` is the
    common per-entity row count when every entity has the same number of rows,
    and ``None`` when they differ — a claim of uniformity has to be measured,
    not assumed.
    """
    meta = pd.read_parquet(paths.afr_meta, columns=["pub_date"])
    dated = meta.pub_date[meta.pub_date != UNDATED]
    afr_min, afr_max = int(dated.min()), int(dated.max())

    per_ticker = asx.groupby("ticker").size().unique()
    frame = pd.DataFrame(
        [
            {
                "dataset": "AFR",
                "row_count": afr_rows,
                "entity_count": None,
                "rows_per_entity": None,
                "min_date": _iso(afr_min),
                "max_date": _iso(afr_max),
            },
            {
                "dataset": "ASX",
                "row_count": len(asx),
                "entity_count": int(asx.ticker.nunique()),
                "rows_per_entity": (
                    int(per_ticker[0]) if len(per_ticker) == 1 else None
                ),
                "min_date": str(asx.date.min()),
                "max_date": str(asx.date.max()),
            },
            {
                "dataset": "RBA",
                "row_count": len(rba),
                "entity_count": None,
                "rows_per_entity": None,
                "min_date": str(rba.effective_date.min()),
                "max_date": str(rba.effective_date.max()),
            },
        ]
    )
    frame.to_parquet(paths.coverage, index=False, compression="zstd")
    return frame


def _iso(stamp: int) -> str:
    text = str(stamp)
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
async def _verify(paths: DataPaths) -> list[str]:
    from src.retrieval import count_matches

    failures = []
    for terms, scope, window, expected in REFERENCE_COUNTS:
        result = await count_matches(
            parse_terms(terms), db_path=paths.afr_db, **window
        )
        status = "ok" if result.total == expected else "MISMATCH"
        logger.info(
            "  %-24s %s expected %d, got %d (%d candidates)",
            scope,
            status,
            expected,
            result.total,
            result.candidates_examined,
        )
        if result.total != expected:
            failures.append(
                f"{scope}: expected {expected}, got {result.total} for {terms}"
            )
    return failures


def verify(paths: DataPaths) -> list[str]:
    """Assert the index reproduces every published reference count."""
    logger.info("Verifying the matching convention against published counts")
    return asyncio.run(_verify(paths))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        choices=("all", "structured", "afr", "embeddings"),
        default="all",
        help="Which part to build. Default: all.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only re-run the reference-count check against an existing index.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )
    paths = load_data_paths()
    paths.data_dir.mkdir(parents=True, exist_ok=True)

    if args.verify:
        failures = verify(paths)
        return _report(failures)

    check_fts5()
    check_sources(paths)

    afr_rows = 0
    if args.stage in ("all", "afr"):
        afr_rows = build_afr(paths)
    elif paths.afr_meta.exists():
        afr_rows = len(pd.read_parquet(paths.afr_meta, columns=["id"]))

    if args.stage in ("all", "structured"):
        asx = build_asx(paths)
        rba = build_rba(paths)
        if paths.afr_meta.exists():
            coverage = build_coverage(paths, afr_rows=afr_rows, asx=asx, rba=rba)
            logger.info("Coverage:\n%s", coverage.to_string(index=False))
        else:
            logger.warning(
                "Skipped coverage: %s not built yet. Re-run --stage structured "
                "after the AFR stage.",
                paths.afr_meta,
            )

    if args.stage in ("all", "embeddings"):
        build_embeddings(paths)

    failures = verify(paths) if paths.afr_db.exists() else ["AFR index not built"]
    return _report(failures)


def _report(failures: list[str]) -> int:
    if failures:
        logger.error(
            "Reference-count verification FAILED. The matching convention is "
            "wrong and every downstream number inherits it:"
        )
        for failure in failures:
            logger.error("  %s", failure)
        return 1
    logger.info("All reference counts reproduced. Index is ready to serve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
