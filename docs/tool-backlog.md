# Tool backlog — proposed capability surface

Candidate tools for `src/tools/`, ranked. **Nothing here is built.** All of it is
blocked on the real dataset files (BLK-2); signatures will shift on first contact
with the data. Requirement ids refer to `requirements.md`.

Ranking combines two things: how much of the scored question surface a tool
unlocks, and how cheap it is to build correctly. Ties break toward the tool the
brief explicitly names in its §10 failure examples.

---

## Ranked list

| # | Tool | Dataset | Returns | Why this rank | Effort |
| --- | --- | --- | --- | --- | --- |
| 1 | `rba_decision_stats` | RBA | Computed counts over a date range: total decisions, changes, increases, decreases, holds | Answers the brief's §10 example 1 outright. One `GROUP BY`. The highest score-per-line tool in the whole layer | S |
| 2 | `rba_decisions` | RBA | Bounded chronological list of decisions in a range, with first/last framing | The simplest possible query, and the substrate every other RBA question leans on. Bounded `LIMIT`, never a full scan (FR-3.5) | S |
| 3 | `rba_hold_runs` | RBA | Longest (or top-N) runs of unchanged rates, each with start date, end date and duration | Answers §10 example 2. Gaps-and-islands SQL — the one genuinely fiddly query, but the brief names it, so it is worth the care (FR-4.4) | M |
| 4 | `asx_price_change` | ASX | Absolute change, percentage change and both endpoint prices between two dates for one company | The whole "price movement" question class in one call. Keeps arithmetic out of both models (FR-3.4) | S |
| 5 | `rba_rate_at` | RBA | The cash rate in force on a given date, plus the decision that set it | Cheap, and it is the join key for most cross-dataset questions ("after the rate rise in X…") | S |
| 6 | `asx_resolve_company` | ASX | Dataset's own identifier for a company name or partial ticker, plus near matches | Not glamorous, but without it a near-miss name silently returns nothing and the agent reports "no data" for data that exists. Build before any ASX question is trusted | S |
| 7 | `afr_search` | AFR | Top-k semantically relevant article excerpts for a query, bounded k | Unlocks the entire sentiment / market-direction question class (FR-4.3). Ranked below the SQL tools only because it depends on the embedding stack landing first (DEP-4) | M |
| 8 | `afr_search_window` | AFR | The same, scoped to a date range | What makes cross-dataset questions work: "coverage in the week after \<event\>". Small delta over #7 once #7 exists | S |
| 9 | `asx_top_movers` | ASX | Ranked best/worst movers over a window, bounded result limit | Covers the ranking question class (FR-4.4). Needs an explicit, documented tie-break so results are stable, not just plausible | M |
| 10 | `asx_price_on` | ASX | Price on a date, or the nearest prior trading day | Mostly subsumed by #4. Worth having for questions that ask for a level rather than a move | S |
| 11 | `asx_price_series` | ASX | Bounded price series over a short window | For "did it trend up or down across the period" questions that a two-point change would misrepresent | M |
| 12 | `afr_article_context` | AFR | A longer bounded excerpt for one already-retrieved article | Only needed when a headline plus snippet is too thin to judge tone. Cheap, but speculative until retrieval quality is measured | S |

---

## Recommendation on how many to actually ship

Stop somewhere around **#8**, and treat #9–#12 as demand-driven.

The agent targets ≤3 tool calls per question (NFR-2.1) and the reasoning brain
routes by reading tool descriptions. A twelve-tool list makes routing *harder*,
not easier — near-duplicate tools such as `asx_price_change` and `asx_price_on`
are exactly the kind of choice a model gets wrong under a tight budget. Each
addition past the first six or so should be justified by a public calibration
question that fails without it (BLK-1).

---

## Deliberately not building

**A generic `query_data` / SQL-passthrough tool.** Tempting, because it covers
everything at once. Rejected for three reasons: an unbounded model-authored
query is the exact "calls `list` on a large dataset" case the brief flags as a
guaranteed 60s breach (§7); correctness moves from tested SQL into untested
model output, which is what FR-3.4 exists to prevent; and read-only enforcement
becomes a matter of parsing model-generated SQL rather than a property of the
tool.

**A calculator tool.** If a model needs a calculator, the wrong tool was called.
The statistic belongs in the tool that already has the rows.

**Anything that pages or lists without a limit.** Every query is filtered and
bounded (FR-3.5).

---

## Settled design rules for all of the above

From `architecture.md` §5, fixed before the data arrives:

- `async def` + `@tool`, with a model-facing docstring: what it is for, when to
  reach for it, what each argument means.
- Return compact evidence, not data dumps.
- Numeric arguments carry declared min/max, so out-of-range values are rejected
  at the schema layer before any data access (FR-3.1).
- All counting, summing, ranking, percentage change, date arithmetic and
  run-length logic happens in SQL or Python (FR-3.4).
- Per-request context (request id, deadline, remaining budget) arrives by
  injection, invisible to the model and impossible to forge.
- A failure returns a structured error into the trace rather than raising out of
  the graph (FR-3.6).
- Read-only connections; the source datasets are never modified (FR-3.7, CON-3).
- AFR tool docstrings state explicitly that the tool cannot answer counts, rate
  levels or computed price changes — steering the brain away from the §10
  example 1 failure at the tool layer, not only in the system prompt.
