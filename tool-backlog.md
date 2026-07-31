# Tool surface — what is built, and what was deliberately left out

The capability surface in `src/tools/`, as built against the real dataset files.
Requirement ids refer to `requirements.md`.

This replaces the pre-data backlog. The ranking that drove build order combined
how much of the scored question surface a tool unlocks with how cheap it is to
build correctly, breaking ties toward the capability the brief names in its §10
failure examples. What survived contact with the data differs from that plan in
two ways worth recording, because both were assumptions the data overturned.

**Semantic search was ranked too high.** It was #7 of 12 and the only AFR tool
specified, on the assumption that AFR questions are sentiment questions and
sentiment needs retrieval. In fact the published AFR questions split three ways
— exact term counting, article lookup by a headline the question already quotes,
and dataset coverage — and *none* of them needs vector similarity. `afr_search`
is now insurance for hidden questions that name a topic rather than an article.

**Near-duplicate price tools collapsed into one.** `asx_price_change`,
`asx_price_on`, `asx_top_movers` and `asx_price_series` were four separate
entries. They are one tool, `asx_return`, because the questions ask for a
window, a per-company breakdown, a basket average and a ranking *together* —
splitting them would have burned the call budget assembling one answer from four
calls, and near-duplicate tools are exactly what a model mis-routes under a
tight budget.

---

## Built

| Tool | Dataset | Returns | Unlocks |
| --- | --- | --- | --- |
| `dataset_coverage` | all | Row count and date span per dataset | Any question reaching past 2021. Cheapest high-value tool in the layer: without it the coverage boundary is guesswork, and a refusal for the wrong reason scores the same as a fabrication |
| `rba_rate_at` | RBA | Target in force on a date, plus the decision that set it | Every "use the cash-rate target in force on \<date\>" question. The join key for the cross-dataset set |
| `rba_decision_stats` | RBA | Counts of changes / increases / decreases / holds, cumulative change, target before the first change and at the end, optional per-year breakdown | The brief's §10 example 1. `target_before_first_change_pct` is what "took the target from X to Y" needs and is easy to omit |
| `rba_decisions` | RBA | Bounded chronological listing, filterable by direction, with each decision's date, change and resulting target | Questions wanting the decisions themselves rather than a count — "the three 2019 cuts and the targets they set". Reports the number matched separately, so truncation is visible |
| `rba_hold_runs` | RBA | Longest runs of an unchanged target, with start, end, duration and decision count | The brief's §10 example 2. Gaps-and-islands, computed here so no model does date arithmetic |
| `asx_return` | ASX | Per-company close-to-close returns, equal-weighted basket average, best, worst and full ranking, over a date window, a calendar year, or several forward windows at once | The whole price-movement class. Two traps handled: naming `tickers` does not shrink the basket, and substituted trading days are reported rather than silently resolved |
| `asx_summary_stats` | ASX | Average daily volume, maximum drawdown with peak and trough dates, or annualised volatility, ranked | Volume, drawdown and volatility questions |
| `asx_resolve_company` | ASX | The dataset's ticker for a company name or fragment, with the full list on a miss | Every ASX question. Files are named by company, records carry `.AX` tickers |
| `afr_count_matches` | AFR | Exact record counts for one or more terms, optionally grouped by year or month | All AFR counting questions. Two-stage: index or `LIKE` narrows, the `\b`-anchored regex decides |
| `afr_article_lookup` | AFR | One article by exact headline and date, with a bounded excerpt, duplicate count, and near matches on a miss | Every question quoting a headline |
| `afr_sentiment_evidence` | AFR | The article plus its positive, negative and hedging language, and the cue balance | The sentiment questions — as evidence. It returns no label |
| `afr_search` | AFR | Top-k semantically ranked excerpts, with an optional date window | Hidden questions that describe coverage by topic rather than naming an article |

Twelve tools is near the practical limit for reliable routing under a tight call
budget. Anything further should be justified by a public calibration question
that fails without it.

---

## Deliberately not built

**A generic `query_data` / SQL-passthrough tool.** Tempting, because it covers
everything at once. Rejected for three reasons: an unbounded model-authored
query is the exact "calls `list` on a large dataset" case the brief flags as a
guaranteed 60s breach (§7); correctness moves from tested code into untested
model output, which is what FR-3.4 exists to prevent; and read-only enforcement
becomes a matter of parsing model-generated SQL rather than a property of the
tool.

**A model-authored regex argument on `afr_count_matches`.** Same category. The
tool takes a list of terms and compiles the pattern itself, with a trailing `*`
as the only piece of matching semantics the model can express. An arbitrary
pattern would move the graded convention into model output and expose 219,538
documents to unbounded backtracking.

**A calculator tool.** If a model needs a calculator, the wrong tool was called.
The statistic belongs in the tool that already has the rows.

**A separate `asx_price_on`.** Subsumed by `asx_return`, which reports both
endpoint closes and the dates they came from.

**Anything that pages or lists without a limit.** Every query is filtered and
bounded (FR-3.5). `afr_count_matches` refuses outright above 80,000 candidate
records rather than truncating, because a truncated count reads as a real one.

---

## Settled design rules

From `architecture.md` §5, and unchanged by contact with the data:

- `async def` + `@tool` with `parse_docstring=True`, and a model-facing
  docstring: what it is for, when to reach for it, what each argument means.
- Return compact evidence, not data dumps.
- Numeric arguments carry declared min/max, so out-of-range values are rejected
  at the schema layer before any data access (FR-3.1).
- All counting, summing, ranking, percentage change, date arithmetic and
  run-length logic happens in pandas or SQL (FR-3.4).
- A failure returns a structured error into the trace rather than raising out of
  the graph (FR-3.6).
- Read-only access; the source datasets are never modified (FR-3.7, CON-3).
- Every tool docstring states what the tool **cannot** do and names the tool
  that can. `afr_search` disclaims counting and names `afr_count_matches`;
  `afr_count_matches` disclaims tone. This is asserted in
  `training/test_orchestrator.py`, because it is the §10 example 1 defence and a
  docstring edit could quietly remove it.
