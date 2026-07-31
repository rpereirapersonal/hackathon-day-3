"""Dataset coverage — what the supplied data can and cannot support.

One tool, and among the highest-value in the layer. Some questions ask about
periods or entities the data does not reach, and the correct answer is to say so
and explain why. Without a coverage tool the reasoning brain has no grounded way
to establish the boundary: it either invents observations for a period that is
absent, or refuses for the wrong reason. Both score zero on a question that is
otherwise straightforward.

Reads ``coverage.parquet``, written by ``src/ingest.py`` from the ingested row
counts and date extremes — measured, not asserted.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain.tools import tool

from src import frames
from src.frames import ArtifactMissingError

logger = logging.getLogger(__name__)


@tool(parse_docstring=True)
async def dataset_coverage(
    dataset: Literal["RBA", "ASX", "AFR", "all"] = "all",
) -> dict[str, Any]:
    """Report each dataset's size, shape and date span.

    Use this before answering any question about a period, and always when a
    question asks whether the data can support an analysis. If the question
    reaches beyond a dataset's date span, that gap is the answer — state it
    rather than reasoning about what the numbers would have been.

    Also use it for "what are the dimensions of the dataset" questions. For
    ASX it reports how many instruments there are (`entity_count`) and how many
    rows each one holds (`rows_per_entity`), as well as the total row count —
    a dimensions question usually wants the first two, not the total.

    Args:
        dataset: Which dataset to report, or all three.
    """
    try:
        frame = frames.coverage()
    except ArtifactMissingError as exc:
        logger.warning("dataset_coverage failed: %s", exc)
        return {"error": str(exc), "tool": "dataset_coverage"}

    if dataset != "all":
        frame = frame[frame["dataset"] == dataset]
        if frame.empty:
            return {"error": f"Unknown dataset {dataset!r}", "tool": "dataset_coverage"}

    counts = ("row_count", "entity_count", "rows_per_entity")
    rows = []
    for record in frame.to_dict(orient="records"):
        # Datasets without entities leave the per-entity columns null, which
        # makes the whole column a float. Drop the nulls rather than report a
        # value that reads as "measured, and empty", and restore the survivors
        # to integers so the answer says 18 rather than 18.0.
        clean = {k: v for k, v in record.items() if v is not None and v == v}
        rows.append(
            {k: int(v) if k in counts else v for k, v in clean.items()}
        )

    return {
        "coverage": rows,
        "note": (
            "Dates are the extremes actually present in the ingested data. A "
            "question outside a span cannot be answered from that dataset. "
            "`entity_count` is the number of distinct instruments and "
            "`rows_per_entity` the rows each one holds, where every entity "
            "holds the same number."
        ),
    }


META_TOOLS = (dataset_coverage,)
