#!/usr/bin/env python3
"""Build a Nemotron synthesis fine-tuning set from RBA/ASX/AFR data.

Each example teaches the domain model to read structured `query_data()`-style
tool results and write a concise, fully-grounded final answer -- the exact
responsibility split described in Challenge_Brief.md (Qwen plans + calls
tools, Nemotron synthesizes from verified tool results).

v2: full RBA date coverage, all ASX ticker/year combos, hiking/easing cycle
detection, cross-dataset RBA-vs-ASX examples, cross-ticker ranking, expanded
AFR pattern/month coverage, and phrasing variants per question type.
"""
import argparse
import json
import random
import re
import statistics
from datetime import datetime
from pathlib import Path

random.seed(7)

SYSTEM_PROMPT = (
    "You are the domain synthesis model for an Australian financial-market Q&A agent. "
    "You receive a question and verified structured tool results. Write a concise final "
    "answer that includes every requested fact, preserves exact numbers/dates/units from "
    "the tool results, and never invents information."
)


def variant(templates, *args):
    return random.choice(templates).format(*args)


# ---------------------------------------------------------------------------
# RBA
# ---------------------------------------------------------------------------

def parse_rba(rba_file):
    rows = []
    with open(rba_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            date = datetime.strptime(rec["Effective Date"], "%d %b %Y")
            rows.append(
                {
                    "date": date,
                    "date_str": date.strftime("%Y-%m-%d"),
                    "change": float(rec["Change % points"]),
                    "rate": float(rec["Cash rate target%"]),
                }
            )
    rows.sort(key=lambda r: r["date"])
    return rows


LOOKUP_TEMPLATES = [
    "What was the RBA cash rate target on {}?",
    "On {}, what was the RBA's official cash rate target?",
    "What cash rate target did the RBA set effective {}?",
]


def rba_lookup_examples(rows, variants_per_row=2):
    ex = []
    for r in rows:
        tool_result = {
            "dataset": "rba",
            "metric": "lookup_rate",
            "args": {"date": r["date_str"]},
            "result": {"effective_date": r["date_str"], "cash_rate_target_pct": r["rate"]},
        }
        answer = f"On {r['date_str']}, the RBA cash rate target was {r['rate']:.2f}%."
        templates = random.sample(LOOKUP_TEMPLATES, min(variants_per_row, len(LOOKUP_TEMPLATES)))
        for t in templates:
            ex.append((t.format(r["date_str"]), tool_result, answer))
    return ex


def rba_aggregate_examples(rows):
    ex = []
    changed = [r for r in rows if r["change"] != 0]
    increases = [r for r in changed if r["change"] > 0]
    decreases = [r for r in changed if r["change"] < 0]

    tool_result = {
        "dataset": "rba", "metric": "count_changes",
        "result": {"total_records": len(rows), "changes": len(changed),
                   "increases": len(increases), "decreases": len(decreases)},
    }
    ex.append((
        "From the first RBA record to the last, how many cash-rate decisions changed the rate, and how many were increases versus decreases?",
        tool_result,
        f"{len(changed)} of the {len(rows)} decision records changed the rate: {len(increases)} increases and {len(decreases)} decreases.",
    ))

    tool_result = {"dataset": "rba", "metric": "count_increases", "result": {"increases": len(increases)}}
    ex.append((
        "How many RBA cash-rate increases have there been across the full dataset?",
        tool_result,
        f"There have been {len(increases)} RBA cash-rate increases across the full dataset.",
    ))

    tool_result = {"dataset": "rba", "metric": "count_decreases", "result": {"decreases": len(decreases)}}
    ex.append((
        "How many RBA cash-rate decreases have there been across the full dataset?",
        tool_result,
        f"There have been {len(decreases)} RBA cash-rate decreases across the full dataset.",
    ))

    max_r = max(rows, key=lambda r: r["rate"])
    min_r = min(rows, key=lambda r: r["rate"])
    max_count = sum(1 for r in rows if r["rate"] == max_r["rate"])
    min_count = sum(1 for r in rows if r["rate"] == min_r["rate"])
    max_first = min(r["date_str"] for r in rows if r["rate"] == max_r["rate"])
    min_first = min(r["date_str"] for r in rows if r["rate"] == min_r["rate"])

    tool_result = {"dataset": "rba", "metric": "extremes",
                   "result": {"max_rate": max_r["rate"], "max_first_date": max_first, "max_record_count": max_count,
                              "min_rate": min_r["rate"], "min_first_date": min_first, "min_record_count": min_count}}
    ex.append((
        "What is the highest cash-rate target ever in the RBA dataset, and how many records show it?",
        tool_result,
        f"The highest cash-rate target in the RBA dataset is {max_r['rate']:.2f}, which first took effect on {max_first}. This maximum rate appears across {max_count} decision records.",
    ))
    ex.append((
        "What is the lowest cash-rate target in the RBA dataset, when did it first take effect, and how many decision records show that rate?",
        tool_result,
        f"The lowest cash-rate target in the RBA dataset was {min_r['rate']:.2f}, which first took effect on {min_first}, and {min_count} decision records show that rate.",
    ))

    # max hold streak
    streaks = []
    last_change_idx = 0
    for i in range(1, len(rows)):
        if rows[i]["change"] != 0:
            streaks.append((rows[last_change_idx]["date"], rows[i]["date"], rows[last_change_idx]["rate"], rows[i]["rate"]))
            last_change_idx = i
    if streaks:
        best = max(streaks, key=lambda s: (s[1] - s[0]).days)
        days = (best[1] - best[0]).days
        tool_result = {"dataset": "rba", "metric": "max_hold_streak",
                       "result": {"days": days, "start_date": best[0].strftime("%Y-%m-%d"),
                                  "end_date": best[1].strftime("%Y-%m-%d"), "rate_before": best[2], "rate_after": best[3]}}
        ex.append((
            "What was the longest stretch between two non-zero RBA rate changes?",
            tool_result,
            f"The longest stretch between two non-zero RBA rate changes was {days} days, lasting from {best[0].strftime('%Y-%m-%d')} to {best[1].strftime('%Y-%m-%d')}, during which the rate held at {best[2]:.2f} before changing to {best[3]:.2f}.",
        ))
    return ex


def rba_cycle_examples(rows):
    """Detect maximal hiking / easing cycles (consecutive non-zero changes of one sign)."""
    ex = []
    changes = [r for r in rows if r["change"] != 0]
    cycles = []
    cur_type, cur_changes = None, []
    for r in changes:
        sign = "hike" if r["change"] > 0 else "ease"
        if cur_type is None or sign == cur_type:
            cur_type = sign
            cur_changes.append(r)
        else:
            cycles.append((cur_type, cur_changes))
            cur_type, cur_changes = sign, [r]
    if cur_changes:
        cycles.append((cur_type, cur_changes))

    for kind, members in cycles:
        if len(members) < 2:
            continue
        start, end = members[0], members[-1]
        cum = sum(m["change"] for m in members)
        idx_before = rows.index(start) - 1
        rate_before = rows[idx_before]["rate"] if idx_before >= 0 else start["rate"] - start["change"]
        label = "hikes" if kind == "hike" else "cuts"
        cycle_word = "tightening" if kind == "hike" else "easing"
        tool_result = {
            "dataset": "rba", "metric": "cycle_summary",
            "args": {"start_date": start["date_str"], "end_date": end["date_str"]},
            "result": {"kind": kind, "num_changes": len(members), "cumulative_change_pct_points": round(cum, 2),
                       "start_date": start["date_str"], "end_date": end["date_str"],
                       "rate_before": rate_before, "rate_after": end["rate"]},
        }
        answer = (
            f"There were {len(members)} {label} during the {start['date_str']} to {end['date_str']} {cycle_word} cycle, "
            f"resulting in a cumulative change of {cum:+.2f} percentage points. The target rate immediately before the "
            f"first change was {rate_before:.2f} percent, and the final target reached on {end['date_str']} was {end['rate']:.2f} percent."
        )
        ex.append((
            f"How many {label} occurred during the {start['date_str']} to {end['date_str']} {cycle_word} cycle, and what was the cumulative change?",
            tool_result, answer,
        ))
    return ex


# ---------------------------------------------------------------------------
# ASX
# ---------------------------------------------------------------------------

def parse_asx(asx_dir):
    by_ticker = {}
    for path in sorted(Path(asx_dir).glob("*.jsonl")):
        ticker = path.stem.split("-ASX-")[0]
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        rows.sort(key=lambda r: r["date"])
        by_ticker[ticker] = rows
    return by_ticker


ANNUAL_RETURN_TEMPLATES = [
    "What was {}'s annual return in {}?",
    "How did {} perform on the ASX in {}, in percentage return terms?",
]


def asx_examples(by_ticker):
    ex = []
    rank_by_year = {}
    for ticker, rows in by_ticker.items():
        if len(rows) < 30:
            continue
        years = sorted({r["date"][:4] for r in rows})
        for year in years:
            yr_rows = [r for r in rows if r["date"][:4] == year]
            if len(yr_rows) < 2:
                continue
            start, end = yr_rows[0], yr_rows[-1]
            ret = (end["close"] - start["open"]) / start["open"] * 100
            tool_result = {
                "dataset": "asx", "metric": "annual_return", "args": {"ticker": ticker, "year": year},
                "result": {"ticker": ticker, "year": year, "start_date": start["date"], "end_date": end["date"],
                           "open": round(start["open"], 4), "close": round(end["close"], 4), "return_pct": round(ret, 2)},
            }
            direction = "gained" if ret >= 0 else "lost"
            answer = (
                f"{ticker} {direction} {abs(ret):.2f}% in {year}, moving from an opening price of "
                f"{start['open']:.2f} on {start['date']} to a closing price of {end['close']:.2f} on {end['date']}."
            )
            for t in random.sample(ANNUAL_RETURN_TEMPLATES, len(ANNUAL_RETURN_TEMPLATES)):
                ex.append((t.format(ticker, year), tool_result, answer))
            rank_by_year.setdefault(year, []).append((ticker, ret))

        # max drawdown
        peak = rows[0]["close"]
        peak_date = rows[0]["date"]
        max_dd, dd_peak_date, dd_trough_date = 0.0, rows[0]["date"], rows[0]["date"]
        for r in rows:
            if r["close"] > peak:
                peak, peak_date = r["close"], r["date"]
            dd = (r["close"] - peak) / peak
            if dd < max_dd:
                max_dd, dd_peak_date, dd_trough_date = dd, peak_date, r["date"]
        tool_result = {"dataset": "asx", "metric": "max_drawdown", "args": {"ticker": ticker},
                       "result": {"ticker": ticker, "max_drawdown_pct": round(max_dd * 100, 2),
                                  "peak_date": dd_peak_date, "trough_date": dd_trough_date}}
        ex.append((
            f"What was {ticker}'s maximum drawdown over the full available sample?",
            tool_result,
            f"{ticker}'s maximum drawdown over the full sample was {abs(max_dd)*100:.2f}%, from a peak on {dd_peak_date} to a trough on {dd_trough_date}.",
        ))

        # full sample return
        full_ret = (rows[-1]["close"] - rows[0]["open"]) / rows[0]["open"] * 100
        tool_result = {"dataset": "asx", "metric": "full_sample_return", "args": {"ticker": ticker},
                       "result": {"ticker": ticker, "start_date": rows[0]["date"], "end_date": rows[-1]["date"],
                                  "return_pct": round(full_ret, 2)}}
        direction = "gained" if full_ret >= 0 else "lost"
        ex.append((
            f"What was {ticker}'s total return across the full available sample, from {rows[0]['date']} to {rows[-1]['date']}?",
            tool_result,
            f"{ticker} {direction} {abs(full_ret):.2f}% across the full sample, from {rows[0]['date']} to {rows[-1]['date']}.",
        ))

        # volatility (annualized stdev of daily returns)
        daily_rets = []
        for i in range(1, len(rows)):
            prev, cur = rows[i - 1]["close"], rows[i]["close"]
            if prev:
                daily_rets.append((cur - prev) / prev)
        if len(daily_rets) > 2:
            vol = statistics.stdev(daily_rets) * (252 ** 0.5) * 100
            tool_result = {"dataset": "asx", "metric": "volatility", "args": {"ticker": ticker},
                           "result": {"ticker": ticker, "annualized_volatility_pct": round(vol, 2)}}
            ex.append((
                f"What was {ticker}'s annualized volatility over the full available sample?",
                tool_result,
                f"{ticker}'s annualized volatility (based on daily close-to-close returns) over the full sample was {vol:.2f}%.",
            ))

    # cross-ticker ranking per year
    for year, entries in sorted(rank_by_year.items()):
        best = max(entries, key=lambda e: e[1])
        worst = min(entries, key=lambda e: e[1])
        tool_result = {"dataset": "asx", "metric": "rank_annual_returns", "args": {"year": year},
                       "result": {"year": year, "ranked": sorted([{"ticker": t, "return_pct": round(r, 2)} for t, r in entries], key=lambda x: -x["return_pct"])}}
        ex.append((
            f"Which ASX company in this dataset had the highest annual return in {year}, and what was it?",
            tool_result,
            f"{best[0]} had the highest annual return in {year} in this dataset, at {best[1]:.2f}%.",
        ))
        ex.append((
            f"Which ASX company in this dataset had the lowest annual return in {year}, and what was it?",
            tool_result,
            f"{worst[0]} had the lowest annual return in {year} in this dataset, at {worst[1]:.2f}%.",
        ))
    return ex


def cross_dataset_examples(rba_rows, by_ticker, tickers=("CBA", "ANZ", "BHP")):
    ex = []
    for ticker in tickers:
        rows = by_ticker.get(ticker)
        if not rows:
            continue
        years = sorted({r["date"][:4] for r in rows})
        for year in years:
            yr_rows = [r for r in rows if r["date"][:4] == year]
            if len(yr_rows) < 2:
                continue
            start, end = yr_rows[0], yr_rows[-1]
            ret = (end["close"] - start["open"]) / start["open"] * 100
            rba_year = [r for r in rba_rows if r["date_str"][:4] == year]
            changed = [r for r in rba_year if r["change"] != 0]
            increases = [r for r in changed if r["change"] > 0]
            decreases = [r for r in changed if r["change"] < 0]
            tool_result = {
                "rba": {"year": year, "changes": len(changed), "increases": len(increases), "decreases": len(decreases)},
                "asx": {"ticker": ticker, "year": year, "return_pct": round(ret, 2),
                        "start_date": start["date"], "end_date": end["date"]},
            }
            direction = "gained" if ret >= 0 else "lost"
            answer = (
                f"In {year}, the RBA made {len(changed)} cash-rate changes ({len(increases)} increases, "
                f"{len(decreases)} decreases), while {ticker} {direction} {abs(ret):.2f}% on the ASX "
                f"(from {start['date']} to {end['date']})."
            )
            cross_templates = [
                f"In {year}, how many RBA cash-rate changes were there, and how did {ticker} perform on the ASX that year?",
                f"For {year}, summarize both the RBA cash-rate activity and {ticker}'s ASX return.",
            ]
            for q in cross_templates:
                ex.append((q, tool_result, answer))
    return ex


# ---------------------------------------------------------------------------
# AFR
# ---------------------------------------------------------------------------

PATTERNS = ["NAB", "CBA", "ANZ", "BHP", "interest rate", "recession", "unemployment", "inflation",
            "ASX", "Reserve Bank", "housing", "coronavirus", "COVID", "China", "budget deficit",
            "wage growth", "iron ore", "trade war", "royal commission", "drought", "bushfire",
            "election", "GDP", "credit rating", "dividend", "IPO", "takeover", "merger", "lithium",
            "gold price", "oil price", "Qantas", "Woolworths", "Wesfarmers", "Telstra", "iron ore price",
            "supply chain", "cybersecurity", "climate change", "renewable energy", "coal", "gas price",
            "mortgage", "APRA", "ASIC", "superannuation", "bond yield", "US Federal Reserve", "tariff",
            "productivity", "labour market", "consumer confidence"]


def afr_count_examples(sampled_records, patterns, total_scanned):
    ex = []
    for pat in patterns:
        rx = re.compile(r"\b" + re.escape(pat) + r"\b", re.IGNORECASE)
        count = 0
        for rec in sampled_records:
            blob = " ".join(str(rec.get(k, "")) for k in ("HEADLINE", "SUBHEAD", "INTRO", "TEXT"))
            if rx.search(blob):
                count += 1
        tool_result = {"dataset": "afr", "metric": "count", "args": {"pattern": pat},
                       "result": {"pattern": pat, "matching_articles": count, "articles_scanned": total_scanned}}
        answer = (
            f"{count} of the {total_scanned} scanned AFR articles mention \"{pat}\" "
            f"(matched case-insensitively as a whole word across HEADLINE, SUBHEAD, INTRO, and TEXT, "
            f"counted once per article)."
        )
        ex.append((f"How many AFR articles in this sample mention \"{pat}\"?", tool_result, answer))

        share_pct = (count / total_scanned * 100) if total_scanned else 0.0
        tool_result2 = {"dataset": "afr", "metric": "share", "args": {"pattern": pat},
                        "result": {"pattern": pat, "matching_articles": count, "articles_scanned": total_scanned,
                                   "share_pct": round(share_pct, 2)}}
        ex.append((
            f"What share of the sampled AFR articles mention \"{pat}\"?",
            tool_result2,
            f"\"{pat}\" appears in {share_pct:.2f}% of the sampled AFR articles ({count} of {total_scanned}), "
            f"matched case-insensitively as a whole word across HEADLINE, SUBHEAD, INTRO, and TEXT, once per article.",
        ))
    return ex


def afr_month_examples(files_by_month, n_patterns):
    """files_by_month: list of (month_label, records) for individually-scanned files."""
    ex = []
    for pat in random.sample(PATTERNS, min(n_patterns, len(PATTERNS))):
        rx = re.compile(r"\b" + re.escape(pat) + r"\b", re.IGNORECASE)
        counts = []
        for month_label, records in files_by_month:
            c = 0
            for rec in records:
                blob = " ".join(str(rec.get(k, "")) for k in ("HEADLINE", "SUBHEAD", "INTRO", "TEXT"))
                if rx.search(blob):
                    c += 1
            counts.append((month_label, c))
        if len(counts) < 2:
            continue
        top_month, top_count = max(counts, key=lambda x: x[1])
        breakdown = ", ".join(f"{c} in {m}" for m, c in counts)
        tool_result = {"dataset": "afr", "metric": "count_by_month", "args": {"pattern": pat},
                       "result": {"pattern": pat, "by_month": [{"month": m, "count": c} for m, c in counts]}}
        answer = (
            f"AFR mentions of \"{pat}\" across the sampled months were: {breakdown} "
            f"(whole-word, case-insensitive, once per article across HEADLINE/SUBHEAD/INTRO/TEXT); "
            f"{top_month} had the most mentions with {top_count}."
        )
        ex.append((
            f"How did AFR mentions of \"{pat}\" vary across the sampled months, and which month had the most?",
            tool_result, answer,
        ))
    return ex


def load_afr_files(afr_dir, n_files):
    files = sorted(Path(afr_dir).glob("AFR_*.jsonl"))
    files = [f for f in files if f.stat().st_size > 100]
    sample_files = random.sample(files, min(n_files, len(files)))
    per_file = []
    for fp in sample_files:
        month_label = fp.stem.replace("AFR_", "")
        recs = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        per_file.append((month_label, recs))
    return per_file


# ---------------------------------------------------------------------------

def to_chat_example(question, tool_result, answer):
    user_msg = (
        f"Question: {question}\n\n"
        f"Verified tool result (from agent runtime, already executed against raw data):\n"
        f"{json.dumps(tool_result, ensure_ascii=False)}\n\n"
        f"Write the final answer."
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--afr_dir", required=True)
    ap.add_argument("--asx_dir", required=True)
    ap.add_argument("--rba_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--afr_n_files", type=int, default=14)
    ap.add_argument("--afr_n_patterns", type=int, default=30)
    ap.add_argument("--afr_month_n_patterns", type=int, default=12)
    ap.add_argument("--cross_tickers", nargs="+", default=["CBA", "ANZ", "BHP", "Qantas"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rba_rows = parse_rba(args.rba_file)
    rba_ex = rba_lookup_examples(rba_rows) + rba_aggregate_examples(rba_rows) + rba_cycle_examples(rba_rows)
    print(f"RBA examples: {len(rba_ex)}")

    by_ticker = parse_asx(args.asx_dir)
    asx_ex = asx_examples(by_ticker)
    print(f"ASX examples: {len(asx_ex)}")

    cross_ex = cross_dataset_examples(rba_rows, by_ticker, tuple(args.cross_tickers))
    print(f"Cross-dataset examples: {len(cross_ex)}")

    afr_files = load_afr_files(args.afr_dir, args.afr_n_files)
    all_afr_records = [rec for _, recs in afr_files for rec in recs]
    count_patterns = PATTERNS[: args.afr_n_patterns] if args.afr_n_patterns < len(PATTERNS) else PATTERNS
    afr_ex = afr_count_examples(all_afr_records, count_patterns, len(all_afr_records))
    afr_ex += afr_month_examples(afr_files, args.afr_month_n_patterns)
    print(f"AFR examples: {len(afr_ex)} (from {len(afr_files)} files, {len(all_afr_records)} articles)")

    all_ex = [to_chat_example(*e) for e in (rba_ex + asx_ex + cross_ex + afr_ex)]
    random.shuffle(all_ex)

    n = len(all_ex)
    n_val = max(1, int(n * 0.08))
    n_test = max(1, int(n * 0.07))
    val = all_ex[:n_val]
    test = all_ex[n_val:n_val + n_test]
    train = all_ex[n_val + n_test:]

    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for e in train:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(out_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for e in val:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(out_dir / "test.jsonl", "w", encoding="utf-8") as f:
        for e in test:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"TOTAL: {n} examples -> {len(train)} train / {len(val)} val / {len(test)} test")


if __name__ == "__main__":
    main()
