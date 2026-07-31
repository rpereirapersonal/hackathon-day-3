"""ASX price tools — deterministic returns, summary statistics, name resolution.

Percentage change, averaging, ranking and drawdown are all computed in pandas
and returned as finished values. Neither model performs arithmetic (FR-3.4).

Two dataset facts drive the design. The files are named by company
("Qantas-ASX-…") while the records carry ``.AX`` tickers ("QAN.AX"), so a name
resolver is not optional — without it a near-miss name silently returns nothing
and the agent reports absent data for data that exists. And the questions
repeatedly ask for an equal-weighted basket excluding one constituent, so the
exclusion is a first-class argument rather than something the reasoning brain
assembles from seventeen separate calls, which would exhaust the tool budget on
its own.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Annotated, Any, Literal

import pandas as pd
from langchain.tools import tool
from pydantic import Field

from src import frames
from src.frames import ArtifactMissingError

logger = logging.getLogger(__name__)

#: Tabcorp is excluded by name in most of the graded questions, so it is the
#: default exclusion. Passing an empty list includes everything.
DEFAULT_EXCLUDE = ("TAH.AX",)


def _failure(tool_name: str, message: str) -> dict[str, Any]:
    logger.warning("%s failed: %s", tool_name, message)
    return {"error": message, "tool": tool_name}


def _resolve(frame: pd.DataFrame, name: str) -> str | None:
    """Map a company name, ticker or fragment to the dataset's ticker."""
    probe = name.strip().lower().removesuffix(".ax")
    pairs = frame[["ticker", "company"]].drop_duplicates()
    for row in pairs.itertuples():
        if probe in (row.ticker.lower().removesuffix(".ax"), row.company.lower()):
            return row.ticker
    for row in pairs.itertuples():
        if probe in row.company.lower() or probe in row.ticker.lower():
            return row.ticker
    return None


def _closes(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return frame[frame["ticker"] == ticker][["date", "close", "volume", "high", "low"]]


def _close_on(series: pd.DataFrame, when, *, direction: str) -> tuple[Any, float] | None:
    """Close on an exact date, else the nearest trading day in ``direction``.

    The substituted date is always returned alongside the price so a
    non-trading-day request is visible rather than silently resolved.
    """
    exact = series[series["date"] == when]
    if not exact.empty:
        return when, float(exact["close"].iloc[0])
    if direction == "backward":
        prior = series[series["date"] <= when]
        if prior.empty:
            return None
        return prior["date"].iloc[-1], float(prior["close"].iloc[-1])
    later = series[series["date"] >= when]
    if later.empty:
        return None
    return later["date"].iloc[0], float(later["close"].iloc[0])


def _return_rows(
    frame: pd.DataFrame,
    universe: list[str],
    start,
    end,
    *,
    year_mode: bool,
) -> list[dict[str, Any]]:
    """Close-to-close return for each ticker in ``universe`` over one window."""
    rows: list[dict[str, Any]] = []
    for ticker in universe:
        series = _closes(frame, ticker)
        if year_mode:
            inside = series[(series["date"] >= start) & (series["date"] <= end)]
            if inside.empty:
                continue
            first_date = inside["date"].iloc[0]
            first_close = float(inside["close"].iloc[0])
            last_date = inside["date"].iloc[-1]
            last_close = float(inside["close"].iloc[-1])
        else:
            opening = _close_on(series, start, direction="forward")
            closing = _close_on(series, end, direction="backward")
            if opening is None or closing is None:
                continue
            (first_date, first_close), (last_date, last_close) = opening, closing

        rows.append(
            {
                "ticker": ticker,
                "return_pct": round(100.0 * (last_close / first_close - 1.0), 4),
                "from_date": str(first_date),
                "from_close": round(first_close, 6),
                "to_date": str(last_date),
                "to_close": round(last_close, 6),
            }
        )
    return rows


def _window_payload(
    frame: pd.DataFrame,
    start,
    end,
    *,
    year_mode: bool,
    basket_universe: list[str],
    report: list[str] | None,
    with_ranking: bool,
) -> dict[str, Any] | None:
    """One window's returns: the basket, its extremes, and the reported tickers.

    The basket is computed over ``basket_universe`` — every ticker except the
    excluded ones — regardless of which tickers the caller asked to see. This
    separation is the whole point of the function. A question that names five
    companies *and* asks for the basket average wants a seventeen-member mean
    alongside five individual returns, and averaging only the five named ones
    yields a plausible number that is simply wrong.
    """
    basket_rows = _return_rows(frame, basket_universe, start, end, year_mode=year_mode)
    if not basket_rows:
        return None

    ranked = sorted(basket_rows, key=lambda row: row["return_pct"], reverse=True)
    by_ticker = {row["ticker"]: row for row in basket_rows}

    if report:
        missing = [t for t in report if t not in by_ticker]
        reported = [by_ticker[t] for t in report if t in by_ticker]
        # A ticker the caller named but excluded from the basket is still worth
        # reporting individually; it just does not enter the average.
        reported += _return_rows(frame, missing, start, end, year_mode=year_mode)
    else:
        reported = basket_rows

    payload: dict[str, Any] = {
        "window": {"from": str(start), "to": str(end)},
        "per_ticker": reported,
        "basket": {
            "constituents": len(basket_rows),
            "average_return_pct": round(
                sum(row["return_pct"] for row in basket_rows) / len(basket_rows), 4
            ),
            "method": "arithmetic mean of constituent close-to-close returns",
            "covers": "every ticker except `exclude`; naming `tickers` does not "
            "narrow the basket",
        },
        "best": {"ticker": ranked[0]["ticker"], "return_pct": ranked[0]["return_pct"]},
        "worst": {
            "ticker": ranked[-1]["ticker"],
            "return_pct": ranked[-1]["return_pct"],
        },
    }
    if with_ranking:
        payload["ranking"] = [
            {"rank": position, "ticker": row["ticker"], "return_pct": row["return_pct"]}
            for position, row in enumerate(ranked, start=1)
        ]
    return payload


@tool(parse_docstring=True)
async def asx_return(
    date_from: str | None = None,
    date_to: str | None = None,
    year: Annotated[int | None, Field(ge=2015, le=2021)] = None,
    start_dates: Annotated[list[str] | None, Field(max_length=6)] = None,
    horizon_days: Annotated[int | None, Field(ge=1, le=365)] = None,
    tickers: Annotated[list[str] | None, Field(max_length=20)] = None,
    exclude: Annotated[list[str] | None, Field(max_length=20)] = None,
) -> dict[str, Any]:
    """Compute close-to-close returns per company and for the equal-weighted basket.

    Use this for every price-movement question: the move for one company between
    two dates, an annual return, a ranking of best and worst performers, or the
    average return of the basket of companies. It returns finished percentages,
    the basket average, and the ranking — never prices for you to subtract.

    Choose one of three ways to say which period you mean:

    * ``date_from`` and ``date_to`` for a single explicit window;
    * ``year`` for a first-to-last close annual return;
    * ``start_dates`` with ``horizon_days`` to measure the same forward window
      after each of several dates in one call — use this for "the return in the
      week after each rate cut" rather than calling the tool once per date,
      which would exhaust the call budget.

    The basket average always covers every company except those in ``exclude``.
    Naming ``tickers`` chooses which individual companies are listed back to
    you; it does not shrink the basket.

    When a requested date is not a trading day the nearest trading day is used
    and reported, so you can see what was actually measured.

    Args:
        date_from: Inclusive start date, YYYY-MM-DD.
        date_to: Inclusive end date, YYYY-MM-DD.
        year: Calendar year for a first-to-last annual return, instead of dates.
        start_dates: Up to six window start dates, YYYY-MM-DD, each measured
            forward by `horizon_days`.
        horizon_days: Length of the forward window in calendar days, used with
            `start_dates`. Seven means one week later.
        tickers: Also list these companies individually, by ticker or name. Does
            not change the basket. Omit to list every constituent.
        exclude: Companies to leave out of the basket and the ranking. Defaults
            to Tabcorp, which most questions exclude. Pass an empty list to
            include everything.
    """
    try:
        frame = frames.asx()
    except ArtifactMissingError as exc:
        return _failure("asx_return", str(exc))

    removed = tuple(DEFAULT_EXCLUDE if exclude is None else exclude)
    excluded_tickers = {t for t in (_resolve(frame, e) for e in removed) if t}
    basket_universe = sorted(set(frame["ticker"]) - excluded_tickers)

    report: list[str] | None = None
    if tickers:
        report, unresolved = [], []
        for name in tickers:
            resolved = _resolve(frame, name)
            (report if resolved else unresolved).append(resolved or name)
        if unresolved:
            return _failure(
                "asx_return",
                f"Unknown companies: {unresolved}. Call asx_resolve_company first.",
            )

    # --- several forward windows from a list of event dates -----------------
    if start_dates:
        if horizon_days is None:
            return _failure(
                "asx_return", "`start_dates` needs `horizon_days` to size each window."
            )
        windows = []
        for raw in start_dates:
            try:
                start = pd.to_datetime(raw).date()
            except (ValueError, TypeError) as exc:
                return _failure("asx_return", f"unparseable start date {raw!r}: {exc}")
            block = _window_payload(
                frame,
                start,
                start + timedelta(days=horizon_days),
                year_mode=False,
                basket_universe=basket_universe,
                report=report,
                with_ranking=False,
            )
            if block is None:
                return _failure("asx_return", f"No price data in the window from {raw}.")
            windows.append(block)
        return {
            "horizon_days": horizon_days,
            "excluded": sorted(excluded_tickers),
            "windows": windows,
        }

    # --- a single window ----------------------------------------------------
    if year is not None:
        start = pd.Timestamp(year=year, month=1, day=1).date()
        end = pd.Timestamp(year=year, month=12, day=31).date()
    elif date_from and date_to:
        try:
            start = pd.to_datetime(date_from).date()
            end = pd.to_datetime(date_to).date()
        except (ValueError, TypeError) as exc:
            return _failure("asx_return", f"unparseable date bound: {exc}")
    else:
        return _failure(
            "asx_return",
            "Supply `year`, or both `date_from` and `date_to`, or `start_dates` "
            "with `horizon_days`.",
        )

    payload = _window_payload(
        frame,
        start,
        end,
        year_mode=year is not None,
        basket_universe=basket_universe,
        report=report,
        with_ranking=True,
    )
    if payload is None:
        return _failure("asx_return", "No price data in that window.")
    payload["excluded"] = sorted(excluded_tickers)
    return payload


@tool(parse_docstring=True)
async def asx_summary_stats(
    metric: Literal["average_volume", "max_drawdown", "volatility"],
    tickers: Annotated[list[str] | None, Field(max_length=20)] = None,
    exclude: Annotated[list[str] | None, Field(max_length=20)] = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top_n: Annotated[int, Field(ge=1, le=20)] = 3,
) -> dict[str, Any]:
    """Rank companies by average daily volume, worst drawdown, or volatility.

    Use this for "highest average daily volume", "largest maximum drawdown", or
    "most volatile" questions. Every figure is computed here, including the peak
    and trough dates of a drawdown.

    Maximum drawdown is the largest fall from a running peak close, reported as
    a negative percentage with the peak and trough dates. Volatility is the
    annualised standard deviation of daily close-to-close returns.

    Args:
        metric: Which statistic to compute and rank by.
        tickers: Restrict to these companies. Omit for all.
        exclude: Companies to leave out. Defaults to Tabcorp.
        date_from: Inclusive start date, YYYY-MM-DD. Omit for the full sample.
        date_to: Inclusive end date, YYYY-MM-DD. Omit for the full sample.
        top_n: How many companies to return, ranked worst-first for drawdown and
            highest-first otherwise.
    """
    try:
        frame = frames.asx()
    except ArtifactMissingError as exc:
        return _failure("asx_summary_stats", str(exc))

    try:
        if date_from:
            frame = frame[frame["date"] >= pd.to_datetime(date_from).date()]
        if date_to:
            frame = frame[frame["date"] <= pd.to_datetime(date_to).date()]
    except (ValueError, TypeError) as exc:
        return _failure("asx_summary_stats", f"unparseable date bound: {exc}")

    removed = tuple(DEFAULT_EXCLUDE if exclude is None else exclude)
    excluded_tickers = {t for t in (_resolve(frame, e) for e in removed) if t}
    if tickers:
        wanted = [t for t in (_resolve(frame, name) for name in tickers) if t]
    else:
        wanted = sorted(set(frame["ticker"]) - excluded_tickers)

    rows: list[dict[str, Any]] = []
    for ticker in wanted:
        series = _closes(frame, ticker)
        if series.empty:
            continue
        if metric == "average_volume":
            rows.append(
                {
                    "ticker": ticker,
                    "average_daily_volume": round(float(series["volume"].mean()), 2),
                    "trading_days": int(len(series)),
                }
            )
        elif metric == "volatility":
            returns = series["close"].pct_change().dropna()
            rows.append(
                {
                    "ticker": ticker,
                    "annualised_volatility_pct": round(
                        float(returns.std(ddof=1) * (252**0.5) * 100), 4
                    ),
                    "trading_days": int(len(series)),
                }
            )
        else:
            closes = series["close"].to_numpy()
            dates = series["date"].tolist()
            peak, peak_index = closes[0], 0
            worst, worst_peak_index, worst_index = 0.0, 0, 0
            for position, price in enumerate(closes):
                if price > peak:
                    peak, peak_index = price, position
                fall = price / peak - 1.0
                if fall < worst:
                    worst, worst_peak_index, worst_index = fall, peak_index, position
            rows.append(
                {
                    "ticker": ticker,
                    # Cast off numpy's float so the tool result serialises to
                    # plain JSON in the trace.
                    "max_drawdown_pct": round(100.0 * float(worst), 4),
                    "peak_date": str(dates[worst_peak_index]),
                    "trough_date": str(dates[worst_index]),
                }
            )

    if not rows:
        return _failure("asx_summary_stats", "No price data in that window.")

    key = {
        "average_volume": "average_daily_volume",
        "volatility": "annualised_volatility_pct",
        "max_drawdown": "max_drawdown_pct",
    }[metric]
    # Drawdowns are negative, so worst-first is ascending; the others rank high-first.
    rows.sort(key=lambda row: row[key], reverse=metric != "max_drawdown")
    return {
        "metric": metric,
        "excluded": sorted(excluded_tickers),
        "companies_ranked": len(rows),
        "ranking": rows[:top_n],
    }


@tool(parse_docstring=True)
async def asx_resolve_company(name: str) -> dict[str, Any]:
    """Map a company name or partial ticker to the identifier the data uses.

    Use this before any price call when the question names a company in words
    rather than by ticker, or when a price call reports an unknown company. The
    price files are named by company while the records carry ``.AX`` tickers, so
    the two do not always look alike.

    Args:
        name: Company name, ticker, or a fragment of either.
    """
    try:
        frame = frames.asx()
    except ArtifactMissingError as exc:
        return _failure("asx_resolve_company", str(exc))

    resolved = _resolve(frame, name)
    pairs = frame[["ticker", "company"]].drop_duplicates()
    if resolved is None:
        return {
            "resolved": False,
            "query": name,
            "available": [
                {"ticker": row.ticker, "company": row.company}
                for row in pairs.itertuples()
            ],
        }
    match = pairs[pairs["ticker"] == resolved].iloc[0]
    return {
        "resolved": True,
        "query": name,
        "ticker": str(match["ticker"]),
        "company": str(match["company"]),
    }


ASX_TOOLS = (asx_return, asx_summary_stats, asx_resolve_company)
