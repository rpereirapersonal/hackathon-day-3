"""Deterministic-calculation tests — AC-5.

No model in the loop. Each tool is called directly and its output compared with
a value published in ``Participant_Package/public_questions.jsonl``, because
this is precisely where the brief's §10 examples fail: a statistic that was
never actually computed (FR-3.4, FR-4.4).

Every expectation below is a published reference value, not a figure this
implementation produced and then enshrined. That direction matters — a test
written against its own output proves only that the code is consistent, not that
it is right.

Expectations are keyed by *tool arguments*, never by question id, and no
question id appears in this file (CON-9, AC-13).

Tolerances follow the published grading notes: dates, tickers, counts, rankings
and RBA values are exact; returns, drawdowns and volatility allow 0.02
percentage points; average volume allows one share.

Skipped as a whole when the ingested artifacts are absent, so a clean checkout
still collects and passes the rest of the suite.
"""

from __future__ import annotations

import pytest

from src.db import IndexMissingError
from src.frames import ArtifactMissingError
from src.text import parse_terms

PCT = 0.02  # percentage-point tolerance for computed returns
SHARE = 1.0  # average-volume tolerance, in shares


@pytest.fixture(scope="module", autouse=True)
def _require_ingest():
    """Skip the module unless the ingested artifacts are present."""
    from src import frames

    try:
        frames.asx()
        frames.rba()
        frames.coverage()
        frames.afr_meta()
    except ArtifactMissingError as exc:
        pytest.skip(f"ingest artifacts missing: {exc}")


# ---------------------------------------------------------------------------
# AFR — the matching convention
# ---------------------------------------------------------------------------
# Exact counts. These four are the reason src/text.py is shaped the way it is:
# they are the only evidence that the convention matches the grader's, and they
# are reproduced by ingest as well, so a drift shows up in two places.
@pytest.mark.parametrize(
    ("terms", "window", "expected"),
    [
        (["unemployment"], ("20200101", "20201231"), 1_452),
        (["unemployment"], ("20200501", "20200531"), 218),
        (["QBE"], ("20210101", "20211231"), 369),
        (
            ["interest rate*", "cash rate*", "rate cut*", "rate hike*", "RBA"],
            ("20190101", "20191231"),
            3_181,
        ),
    ],
)
async def test_afr_counts_match_published_values(terms, window, expected):
    from src.retrieval import count_matches

    try:
        result = await count_matches(
            parse_terms(terms), date_from=window[0], date_to=window[1]
        )
    except IndexMissingError as exc:
        pytest.skip(str(exc))
    assert result.total == expected


async def test_afr_month_grouping_ranks_the_peak_first():
    """Grouping must rank by count, not by period order."""
    from src.retrieval import count_matches

    try:
        result = await count_matches(
            parse_terms(["unemployment"]), group_by="month", top_n=1
        )
    except IndexMissingError as exc:
        pytest.skip(str(exc))
    assert result.groups[0] == {"period": "2020-05", "count": 218}


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"terms": ["QBE"], "year": 2021}, 369),
        (
            {
                "terms": ["interest rate*", "cash rate*", "rate cut*", "rate hike*", "RBA"],
                "year": 2019,
            },
            3_181,
        ),
    ],
)
async def test_counts_survive_the_tool_wrapper(args, expected):
    """Same values through the model-facing surface, not just the engine.

    The wrapper parses the terms and turns `year` into a date window, and both
    are places a bug could change a count without the engine ever being wrong.
    """
    from src.tools.afr import afr_count_matches

    result = await afr_count_matches.ainvoke(args)
    assert "error" not in result, result
    assert result["total_records"] == expected


async def test_year_shortcut_matches_explicit_bounds():
    from src.tools.afr import afr_count_matches

    shortcut = await afr_count_matches.ainvoke({"terms": ["QBE"], "year": 2021})
    explicit = await afr_count_matches.ainvoke(
        {"terms": ["QBE"], "date_from": "2021-01-01", "date_to": "2021-12-31"}
    )
    assert shortcut["total_records"] == explicit["total_records"]


@pytest.mark.parametrize(
    ("headline", "date"),
    [
        ("Travel stocks take off on vaccine rollout", "2021-02-23"),
        ("Why investors don't believe the RBA on interest rates", "2021-11-25"),
        ("Energy stocks shine as vaccines fuel oil rally", "2020-11-28"),
    ],
)
async def test_named_articles_resolve_by_headline_and_date(headline, date):
    """The sentiment questions quote a headline — the lookup has to be exact."""
    from src.tools.afr import afr_article_lookup

    result = await afr_article_lookup.ainvoke(
        {"headline": headline, "publication_date": date}
    )
    assert result["found"] is True, result
    assert result["publication_date"] == date
    assert result["excerpt"]


async def test_sentiment_evidence_returns_cues_and_no_label():
    """Evidence, not a verdict: the label belongs to synthesis (FR-5.2, CON-7)."""
    from src.tools.afr import afr_sentiment_evidence

    result = await afr_sentiment_evidence.ainvoke(
        {
            "headline": "Energy stocks shine as vaccines fuel oil rally",
            "publication_date": "2020-11-28",
        }
    )
    assert result["found"] is True
    assert result["positive_cues"] and result["negative_cues"]
    assert result["cue_balance"] == len(result["positive_cues"]) - len(
        result["negative_cues"]
    )
    assert not {"label", "sentiment", "direction"} & set(result)


async def test_a_missed_headline_offers_near_matches():
    """A miss must not read as absent data."""
    from src.tools.afr import afr_article_lookup

    result = await afr_article_lookup.ainvoke(
        {
            "headline": "Travel stocks take off on vaccine rollout",
            "publication_date": "2021-02-24",  # one day late
        }
    )
    assert result["found"] is False
    assert result["near_matches"]


async def test_substring_terms_differ_from_bounded_terms():
    """The asterisk has to change the count, or it is not doing anything.

    Bounded ``rate cut`` cannot match "rate cuts"; the substring form must.
    """
    from src.retrieval import count_matches

    try:
        bounded = await count_matches(
            parse_terms(["rate cut"]), date_from="20190101", date_to="20191231"
        )
        substring = await count_matches(
            parse_terms(["rate cut*"]), date_from="20190101", date_to="20191231"
        )
    except IndexMissingError as exc:
        pytest.skip(str(exc))
    assert substring.total > bounded.total


# ---------------------------------------------------------------------------
# RBA
# ---------------------------------------------------------------------------
async def test_rba_decision_counts_over_the_full_record():
    from src.tools.rba import rba_decision_stats

    result = await rba_decision_stats.ainvoke({})
    assert result["decision_records"] == 175
    assert result["changed_the_rate"] == 41
    assert result["increases"] == 20
    assert result["decreases"] == 21


async def test_rba_easing_cycle_totals_and_endpoints():
    from src.tools.rba import rba_decision_stats

    result = await rba_decision_stats.ainvoke(
        {"date_from": "2011-01-01", "date_to": "2013-12-31", "group_by": "year"}
    )
    assert result["decreases"] == 8
    assert result["cumulative_change_pp"] == pytest.approx(-2.25)
    assert result["target_before_first_change_pct"] == pytest.approx(4.75)
    assert result["target_at_end_pct"] == pytest.approx(2.50)
    assert [(y["year"], y["decreases"]) for y in result["by_year"]] == [
        (2011, 2),
        (2012, 4),
        (2013, 2),
    ]


@pytest.mark.parametrize(
    "as_of", ["2021-02-23", "2021-11-25", "2020-11-28"]
)
async def test_rate_in_force_is_the_preceding_decision(as_of):
    """The rate in force is set by an earlier decision, not the current month."""
    from src.tools.rba import rba_rate_at

    result = await rba_rate_at.ainvoke({"as_of": as_of})
    assert result["cash_rate_target_pct"] == pytest.approx(0.10)
    assert result["set_by_decision_dated"] < as_of


async def test_2019_easing_cycle_end_state():
    from src.tools.rba import rba_decision_stats

    result = await rba_decision_stats.ainvoke(
        {"date_from": "2019-01-01", "date_to": "2019-12-31"}
    )
    assert result["decreases"] == 3
    assert result["cumulative_change_pp"] == pytest.approx(-0.75)
    assert result["target_at_end_pct"] == pytest.approx(0.75)


async def test_the_three_2019_cuts_and_the_targets_they_set():
    """The decisions themselves, not a count — targets 1.25, 1.00, 0.75."""
    from src.tools.rba import rba_decisions

    result = await rba_decisions.ainvoke(
        {"date_from": "2019-01-01", "date_to": "2019-12-31", "direction": "decrease"}
    )
    assert result["matched"] == 3
    assert result["truncated"] is False
    assert [d["effective_date"] for d in result["decisions"]] == [
        "2019-06-05",
        "2019-07-03",
        "2019-10-02",
    ]
    assert [d["cash_rate_target_pct"] for d in result["decisions"]] == [1.25, 1.00, 0.75]


async def test_decision_listing_carries_each_cut_and_the_target_it_set():
    """Counts alone cannot answer "the cuts took the target to X, Y and Z".

    A statistics call reports three cuts in 2019 but not which dates they fell
    on or what each one produced, and those are separately graded components.
    """
    from src.tools.rba import rba_decisions

    result = await rba_decisions.ainvoke(
        {"date_from": "2019-01-01", "date_to": "2019-12-31", "direction": "decrease"}
    )
    assert result["matched"] == 3
    assert result["truncated"] is False
    assert [
        (row["effective_date"], row["cash_rate_target_pct"])
        for row in result["decisions"]
    ] == [("2019-06-05", 1.25), ("2019-07-03", 1.00), ("2019-10-02", 0.75)]


async def test_hold_runs_are_ordered_longest_first():
    from src.tools.rba import rba_hold_runs

    result = await rba_hold_runs.ainvoke({"top_n": 3})
    days = [run["days"] for run in result["longest_runs"]]
    assert days == sorted(days, reverse=True)
    for run in result["longest_runs"]:
        assert run["start_date"] <= run["end_date"]


# ---------------------------------------------------------------------------
# ASX
# ---------------------------------------------------------------------------
async def test_dataset_dimensions():
    from src import frames

    asx = frames.asx()
    assert asx["ticker"].nunique() == 18
    assert set(asx.groupby("ticker").size().unique()) == {1_774}
    assert str(asx["date"].min()) == "2015-01-02"
    assert str(asx["date"].max()) == "2021-12-30"


@pytest.mark.parametrize(
    ("date_from", "date_to", "expected_basket"),
    [
        ("2019-06-05", "2019-06-12", 2.88),
        ("2019-07-03", "2019-07-10", 0.24),
        ("2019-10-02", "2019-10-09", -2.17),
        ("2020-11-30", "2020-12-07", 2.37),
    ],
)
async def test_basket_return_over_a_window(date_from, date_to, expected_basket):
    from src.tools.asx import asx_return

    result = await asx_return.ainvoke({"date_from": date_from, "date_to": date_to})
    assert result["basket"]["constituents"] == 17
    assert result["basket"]["average_return_pct"] == pytest.approx(
        expected_basket, abs=PCT
    )


async def test_per_ticker_returns_over_a_window():
    from src.tools.asx import asx_return

    result = await asx_return.ainvoke(
        {"date_from": "2019-06-05", "date_to": "2019-06-12"}
    )
    returns = {row["ticker"]: row["return_pct"] for row in result["per_ticker"]}
    for ticker, expected in (
        ("CBA.AX", 0.60),
        ("NAB.AX", 1.39),
        ("ANZ.AX", 0.89),
        ("BHP.AX", 5.89),
        ("RIO.AX", 2.91),
    ):
        assert returns[ticker] == pytest.approx(expected, abs=PCT)


async def test_naming_tickers_does_not_shrink_the_basket():
    """The trap in the multi-part price questions.

    A question that names five companies *and* asks for the basket average wants
    a seventeen-member mean alongside five individual returns. Averaging only the
    five named ones yields a plausible number that is simply wrong.
    """
    from src.tools.asx import asx_return

    result = await asx_return.ainvoke(
        {
            "date_from": "2019-06-05",
            "date_to": "2019-06-12",
            "tickers": ["CBA.AX", "NAB.AX", "ANZ.AX", "BHP.AX", "RIO.AX"],
        }
    )
    assert len(result["per_ticker"]) == 5
    assert result["basket"]["constituents"] == 17
    assert result["basket"]["average_return_pct"] == pytest.approx(2.88, abs=PCT)


async def test_several_forward_windows_in_one_call():
    """One call for all three post-decision weeks, not three calls.

    Three separate calls would consume the whole tool budget on a question that
    also needs the decision targets.
    """
    from src.tools.asx import asx_return

    result = await asx_return.ainvoke(
        {
            "start_dates": ["2019-06-05", "2019-07-03", "2019-10-02"],
            "horizon_days": 7,
        }
    )
    observed = [w["basket"]["average_return_pct"] for w in result["windows"]]
    for actual, expected in zip(observed, (2.88, 0.24, -2.17), strict=True):
        assert actual == pytest.approx(expected, abs=PCT)


async def test_annual_returns_and_ranking():
    from src.tools.asx import asx_return

    result_2018 = await asx_return.ainvoke({"year": 2018})
    assert result_2018["best"]["ticker"] == "BHP.AX"
    assert result_2018["best"]["return_pct"] == pytest.approx(22.17, abs=PCT)
    assert result_2018["worst"]["ticker"] == "AMP.AX"
    assert result_2018["worst"]["return_pct"] == pytest.approx(-50.04, abs=PCT)

    result_2021 = await asx_return.ainvoke({"year": 2021})
    assert result_2021["best"]["ticker"] == "QBE.AX"
    assert result_2021["best"]["return_pct"] == pytest.approx(35.57, abs=PCT)

    result_2019 = await asx_return.ainvoke({"year": 2019})
    assert result_2019["basket"]["average_return_pct"] == pytest.approx(
        20.11, abs=PCT
    )


async def test_tabcorp_is_excluded_by_default_and_includable_on_request():
    """The default exclusion is load-bearing: it changes the volume ranking."""
    from src.tools.asx import asx_summary_stats

    default = await asx_summary_stats.ainvoke({"metric": "average_volume", "top_n": 1})
    assert default["ranking"][0]["ticker"] == "AMP.AX"
    assert default["ranking"][0]["average_daily_volume"] == pytest.approx(
        11_635_671.71, abs=SHARE
    )

    everything = await asx_summary_stats.ainvoke(
        {"metric": "average_volume", "exclude": [], "top_n": 1}
    )
    assert everything["ranking"][0]["ticker"] == "TAH.AX"


async def test_worst_drawdowns_with_peak_and_trough_dates():
    from src.tools.asx import asx_summary_stats

    result = await asx_summary_stats.ainvoke({"metric": "max_drawdown", "top_n": 3})
    expected = [
        ("AMP.AX", -82.45, "2015-03-20", "2021-12-17"),
        ("AGL.AX", -76.24, "2017-04-10", "2021-11-16"),
        ("QAN.AX", -71.08, "2019-12-19", "2020-03-19"),
    ]
    for row, (ticker, drawdown, peak, trough) in zip(
        result["ranking"], expected, strict=True
    ):
        assert row["ticker"] == ticker
        assert row["max_drawdown_pct"] == pytest.approx(drawdown, abs=PCT)
        assert row["peak_date"] == peak
        assert row["trough_date"] == trough


@pytest.mark.parametrize(
    ("name", "ticker"),
    [
        ("Qantas", "QAN.AX"),
        ("Rio", "RIO.AX"),
        ("Tabcorp", "TAH.AX"),
        ("TAH", "TAH.AX"),
        ("CBA.AX", "CBA.AX"),
    ],
)
async def test_company_names_resolve_to_dataset_tickers(name, ticker):
    """Files are named by company, records carry .AX tickers — both must resolve."""
    from src.tools.asx import asx_resolve_company

    result = await asx_resolve_company.ainvoke({"name": name})
    assert result["resolved"] is True
    assert result["ticker"] == ticker


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
async def test_coverage_reports_the_2021_boundary():
    """The grounding for refusing an out-of-range analysis.

    AFR and ASX both stop in 2021 while RBA continues well past it. A question
    about a later period has to be refused on this evidence, not answered from
    the model's priors.
    """
    from src.tools.meta import dataset_coverage

    result = await dataset_coverage.ainvoke({})
    spans = {row["dataset"]: row for row in result["coverage"]}
    assert spans["AFR"]["max_date"].startswith("2021")
    assert spans["ASX"]["max_date"].startswith("2021")
    assert spans["RBA"]["max_date"] > "2022"
    assert spans["AFR"]["row_count"] == 219_538
    assert spans["RBA"]["row_count"] == 175


async def test_coverage_reports_asx_dimensions_not_just_a_row_total():
    """A dimensions question wants instruments and rows-per-instrument.

    18 tickers of 1,774 rows each is the published answer; 31,932 is the total
    and answers a different question.
    """
    from src.tools.meta import dataset_coverage

    result = await dataset_coverage.ainvoke({"dataset": "ASX"})
    asx = result["coverage"][0]
    assert asx["entity_count"] == 18
    assert asx["rows_per_entity"] == 1_774
    assert asx["min_date"] == "2015-01-02"
    assert asx["max_date"] == "2021-12-30"


async def test_coverage_omits_per_entity_keys_where_they_do_not_apply():
    """A null would read as "measured, and empty" rather than "not applicable"."""
    from src.tools.meta import dataset_coverage

    result = await dataset_coverage.ainvoke({"dataset": "AFR"})
    assert "entity_count" not in result["coverage"][0]
