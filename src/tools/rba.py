"""RBA cash-rate decision tools — structured queries and derived statistics.

The highest-priority capability group in the tool layer: both of the brief's
§10 worked failures are RBA questions, and both fail the same way — a statistic
that was never actually computed. Every number here is computed in pandas and
returned finished, so neither model ever counts, sums or does date arithmetic
(FR-3.4).

``rba_rate_at`` is also the join key for the cross-dataset questions: "use the
cash-rate target in force on <date>" appears in every article-plus-rate
question, and guessing it from the month is a scored error.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

import pandas as pd
from langchain.tools import tool
from pydantic import Field

from src import frames
from src.frames import ArtifactMissingError

logger = logging.getLogger(__name__)


def _failure(tool_name: str, message: str) -> dict[str, Any]:
    logger.warning("%s failed: %s", tool_name, message)
    return {"error": message, "tool": tool_name}


def _window(frame: pd.DataFrame, date_from: str | None, date_to: str | None):
    selected = frame
    if date_from:
        selected = selected[selected["effective_date"] >= pd.to_datetime(date_from).date()]
    if date_to:
        selected = selected[selected["effective_date"] <= pd.to_datetime(date_to).date()]
    return selected


@tool(parse_docstring=True)
async def rba_rate_at(as_of: str) -> dict[str, Any]:
    """Get the RBA cash-rate target in force on a given date.

    Use this whenever a question says "the cash-rate target in force on <date>",
    or anchors anything to a rate at a point in time. The rate in force is set
    by the most recent decision on or before that date, which is usually not the
    same month — do not infer it from the date yourself.

    Returns the target in force, the decision that set it, and that decision's
    change in percentage points.

    Args:
        as_of: The date to evaluate, YYYY-MM-DD.
    """
    try:
        frame = frames.rba()
        target = pd.to_datetime(as_of).date()
    except ArtifactMissingError as exc:
        return _failure("rba_rate_at", str(exc))
    except (ValueError, TypeError) as exc:
        return _failure("rba_rate_at", f"unparseable date {as_of!r}: {exc}")

    eligible = frame[frame["effective_date"] <= target]
    if eligible.empty:
        first = frame["effective_date"].min()
        return _failure(
            "rba_rate_at",
            f"No RBA decision on or before {as_of}. The data starts at {first}.",
        )

    row = eligible.iloc[-1]
    return {
        "as_of": as_of,
        "cash_rate_target_pct": float(row["cash_rate_target"]),
        "set_by_decision_dated": str(row["effective_date"]),
        "that_decision_change_pp": float(row["change_pp"]),
        "that_decision_direction": str(row["direction"]),
    }


@tool(parse_docstring=True)
async def rba_decision_stats(
    date_from: str | None = None,
    date_to: str | None = None,
    group_by: Literal["none", "year"] = "none",
) -> dict[str, Any]:
    """Count RBA decisions over a period and total the change in the rate.

    Use this for "how many decisions changed the rate", "how many increases
    versus decreases", "how many cuts in <period>", "how far did the target
    fall", or a year-end target. It returns finished counts and totals computed
    over the decision records — never a list for you to count.

    Also returns the target in force immediately before the first change in the
    period, which is what "took the target from X to Y" questions need.

    Args:
        date_from: Inclusive start date, YYYY-MM-DD. Omit for the first record.
        date_to: Inclusive end date, YYYY-MM-DD. Omit for the last record.
        group_by: Break the counts down per calendar year as well as in total.
    """
    try:
        frame = frames.rba()
        selected = _window(frame, date_from, date_to)
    except ArtifactMissingError as exc:
        return _failure("rba_decision_stats", str(exc))
    except (ValueError, TypeError) as exc:
        return _failure("rba_decision_stats", f"unparseable date bound: {exc}")

    if selected.empty:
        return _failure(
            "rba_decision_stats", "No RBA decisions fall in that date range."
        )

    changes = selected[selected["change_pp"] != 0]
    payload: dict[str, Any] = {
        "window": {
            "from": str(selected["effective_date"].iloc[0]),
            "to": str(selected["effective_date"].iloc[-1]),
        },
        "decision_records": int(len(selected)),
        "changed_the_rate": int(len(changes)),
        "increases": int((selected["direction"] == "increase").sum()),
        "decreases": int((selected["direction"] == "decrease").sum()),
        "holds": int((selected["direction"] == "hold").sum()),
        "cumulative_change_pp": round(float(selected["change_pp"].sum()), 4),
        "target_at_end_pct": float(selected["cash_rate_target"].iloc[-1]),
    }

    # The level "before the first cut" is set by the preceding decision, which
    # may sit outside the requested window.
    if not changes.empty:
        first_change_date = changes["effective_date"].iloc[0]
        prior = frame[frame["effective_date"] < first_change_date]
        payload["first_change_dated"] = str(first_change_date)
        payload["target_before_first_change_pct"] = (
            float(prior["cash_rate_target"].iloc[-1]) if not prior.empty else None
        )

    if group_by == "year":
        by_year = []
        for year, group in selected.groupby(
            selected["effective_date"].map(lambda d: d.year)
        ):
            by_year.append(
                {
                    "year": int(year),
                    "increases": int((group["direction"] == "increase").sum()),
                    "decreases": int((group["direction"] == "decrease").sum()),
                    "holds": int((group["direction"] == "hold").sum()),
                    "cumulative_change_pp": round(float(group["change_pp"].sum()), 4),
                    "target_at_year_end_pct": float(group["cash_rate_target"].iloc[-1]),
                }
            )
        payload["by_year"] = by_year

    return payload


@tool(parse_docstring=True)
async def rba_decisions(
    date_from: str | None = None,
    date_to: str | None = None,
    direction: Literal["any", "increase", "decrease", "hold", "change"] = "any",
    order: Literal["asc", "desc"] = "asc",
    limit: Annotated[int, Field(ge=1, le=60)] = 20,
) -> dict[str, Any]:
    """List individual RBA decisions with their dates and resulting targets.

    Use this when a question needs the decisions themselves rather than a
    count: the date a cut took effect, the target level it produced, or the
    sequence of moves across a period. "The three 2019 cuts and the targets
    they set" is exactly this tool, filtered to decreases.

    Prefer ``rba_decision_stats`` when the question only wants totals — that is
    one call instead of a list you would otherwise have to tally, and tallying
    is not something you do.

    Args:
        date_from: Inclusive start date, YYYY-MM-DD. Omit for the first record.
        date_to: Inclusive end date, YYYY-MM-DD. Omit for the last record.
        direction: Keep only increases, decreases, holds, or `change` for any
            non-zero move. `any` keeps everything.
        order: `asc` for oldest first, `desc` for most recent first.
        limit: Maximum decisions returned. The number matching the filter is
            reported separately so a truncated result is visible.
    """
    try:
        selected = _window(frames.rba(), date_from, date_to)
    except ArtifactMissingError as exc:
        return _failure("rba_decisions", str(exc))
    except (ValueError, TypeError) as exc:
        return _failure("rba_decisions", f"unparseable date bound: {exc}")

    if direction == "change":
        selected = selected[selected["change_pp"] != 0]
    elif direction != "any":
        selected = selected[selected["direction"] == direction]

    matched = int(len(selected))
    if matched == 0:
        return _failure(
            "rba_decisions", f"No {direction} decisions fall in that date range."
        )

    if order == "desc":
        selected = selected.iloc[::-1]
    shown = selected.head(limit)

    return {
        "filter": {"direction": direction, "order": order},
        "matched": matched,
        "returned": int(len(shown)),
        "truncated": matched > len(shown),
        "decisions": [
            {
                "effective_date": str(row.effective_date),
                "change_pp": float(row.change_pp),
                "cash_rate_target_pct": float(row.cash_rate_target),
                "direction": str(row.direction),
            }
            for row in shown.itertuples()
        ],
    }


@tool(parse_docstring=True)
async def rba_hold_runs(
    top_n: Annotated[int, Field(ge=1, le=10)] = 3,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Find the longest stretches where the RBA left the cash rate unchanged.

    Use this for "the longest period rates were held unchanged", or any question
    about how long a rate level persisted. The run lengths and their start and
    end dates are computed here — that is date arithmetic, which you do not do.

    A run starts at the decision that set a level and ends at the last decision
    before the level changed.

    Args:
        top_n: How many of the longest runs to return, longest first.
        date_from: Inclusive start date, YYYY-MM-DD.
        date_to: Inclusive end date, YYYY-MM-DD.
    """
    try:
        selected = _window(frames.rba(), date_from, date_to)
    except ArtifactMissingError as exc:
        return _failure("rba_hold_runs", str(exc))
    except (ValueError, TypeError) as exc:
        return _failure("rba_hold_runs", f"unparseable date bound: {exc}")

    if selected.empty:
        return _failure("rba_hold_runs", "No RBA decisions fall in that date range.")

    # Gaps and islands: a new run begins wherever the target differs from the
    # previous record's target.
    target = selected["cash_rate_target"].to_numpy()
    dates = selected["effective_date"].tolist()
    runs = []
    start = 0
    for position in range(1, len(target) + 1):
        if position == len(target) or target[position] != target[start]:
            runs.append(
                {
                    "cash_rate_target_pct": float(target[start]),
                    "start_date": str(dates[start]),
                    "end_date": str(dates[position - 1]),
                    "days": int((dates[position - 1] - dates[start]).days),
                    "decisions_in_run": int(position - start),
                }
            )
            start = position

    runs.sort(key=lambda run: (-run["days"], run["start_date"]))
    return {"longest_runs": runs[:top_n], "runs_found": len(runs)}


RBA_TOOLS = (rba_rate_at, rba_decision_stats, rba_decisions, rba_hold_runs)
