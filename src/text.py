"""AFR text conventions — the single definition of how records are matched.

Imported by ``src/ingest.py`` (indexing) and ``src/tools/afr.py`` (querying),
so the two cannot drift. Every rule below was established by running candidate
conventions against the four published reference counts in
``Participant_Package/public_questions.jsonl`` and keeping the one that
reproduced all of them exactly.

Verified counts, reproduced by :func:`build_confirm_regex` over
:func:`raw_text`, counted once per record:

===========================================  ============  ========
Terms                                        Scope         Expected
===========================================  ============  ========
``["unemployment"]``                         2020            1,452
``["unemployment"]``                         May 2020          218
``["QBE"]``                                  2021              369
``["interest rate*", "cash rate*",
"rate cut*", "rate hike*", "RBA"]``          2019            3,181
===========================================  ============  ========

Four findings that are easy to undo by accident, each of which silently
changes those numbers:

**Match the unescaped field text, never a serialised record.** ``json.dumps``
renders a newline as a literal backslash-n, so ``...\\n\\nQBE Insurance``
presents as ``nQBE`` and a leading ``\\b`` fails to fire. Counting ``QBE`` that
way returns 95 instead of 369 — a silent fourfold undercount with no error.

**Never deduplicate.** The corpus holds 219,538 records against 182,490 unique
``(HEADLINE, PUBLICATIONDATE)`` pairs. The 37,048 apparent duplicates are
counted, and dropping them breaks every reference value.

**A trailing ``*`` means plain substring, with no word boundary on either
side.** The graded pattern behind the 3,181 figure is
``interest rates?|cash rate|rate cut|rate hike|\\bRBA\\b`` — only the single
token closes on the right. Adding a leading ``\\b`` to the phrases loses two
records; closing them on the right loses sixty-two. So a bounded phrase is
wrong in both directions, and ``*`` exists to express the open form.

**Normalised text is a superset generator, not a count.**
:func:`norm_text` is exact for single tokens but not for phrases: the 2019
pattern yields 3,119 when phrases are anchored on both sides and 3,200 when
anchored only on the left, against 3,181 exact. Normalised matching therefore
narrows candidates; :func:`build_confirm_regex` decides the count.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

# ---------------------------------------------------------------------------
# Record surface
# ---------------------------------------------------------------------------
#: Fields concatenated to form the matchable surface of one AFR record, in
#: file order. ``INTRO`` is empty on many records; ``TEXT`` carries the body.
AFR_FIELDS: tuple[str, ...] = ("HEADLINE", "SUBHEAD", "INTRO", "TEXT")

_NONWORD = re.compile(r"[^a-z0-9]+")


def raw_text(record: Mapping[str, Any]) -> str:
    """Return the matchable surface of one AFR record.

    The four text fields joined by newlines, unescaped. This is the string
    every count is confirmed against, and the string indexed for retrieval.
    """
    parts = []
    for field in AFR_FIELDS:
        value = record.get(field)
        parts.append("" if value is None else str(value))
    return "\n".join(parts)


def norm_text(s: str) -> str:
    """Lowercase, strip punctuation, and pad with single spaces.

    Padding lets a word-boundary probe be written as a plain substring: a
    single token bounded on both sides is ``' token '``. Used for candidate
    narrowing and for normalising query text before embedding — never as the
    final arbiter of a count (see the module docstring).
    """
    return " " + _NONWORD.sub(" ", s.lower()).strip() + " "


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TermSpec:
    """One search term and the boundary semantics it carries.

    Attributes:
        term: The literal text to look for, with any ``*`` marker removed.
        substring: ``True`` when the term was written with a trailing ``*``,
            meaning match anywhere including inside a longer word. ``False``
            means bounded by ``\\b`` on both sides.
    """

    term: str
    substring: bool

    @property
    def is_single_token(self) -> bool:
        return " " not in self.term


def parse_term(raw: str) -> TermSpec:
    """Parse one model-supplied term into a :class:`TermSpec`.

    A trailing ``*`` selects substring semantics, which is what multi-word
    phrases need in order to pick up inflected forms ("cash rate" also
    matching "cash rates"). Without it the term is bounded on both sides.

    Raises:
        ValueError: if the term is empty once the marker is stripped.
    """
    stripped = raw.strip()
    substring = stripped.endswith("*")
    if substring:
        stripped = stripped[:-1].strip()
    if not stripped:
        raise ValueError(f"empty search term: {raw!r}")
    return TermSpec(term=stripped, substring=substring)


def parse_terms(raws: Iterable[str]) -> tuple[TermSpec, ...]:
    return tuple(parse_term(r) for r in raws)


def build_confirm_regex(specs: Iterable[TermSpec]) -> re.Pattern[str]:
    """Compile the pattern that decides the count.

    Bounded terms become ``\\bterm\\b``; substring terms are emitted bare.
    Alternatives are joined in the order given and matched case-insensitively
    against :func:`raw_text`.

    Raises:
        ValueError: if no terms were supplied.
    """
    alternatives = []
    for spec in specs:
        literal = re.escape(spec.term)
        alternatives.append(literal if spec.substring else rf"\b{literal}\b")
    if not alternatives:
        raise ValueError("at least one search term is required")
    return re.compile("|".join(alternatives), re.IGNORECASE)


def describe_convention(specs: Iterable[TermSpec]) -> str:
    """Render the applied convention for the tool trace.

    The count is only interpretable alongside the rule that produced it, so
    every result carries this string back to the organizers' diagnostics.
    """
    rendered = ", ".join(
        f"{s.term!r} (substring)" if s.substring else f"{s.term!r} (whole word)"
        for s in specs
    )
    return (
        f"case-insensitive match on {rendered}; once per record; over "
        f"{' + '.join(AFR_FIELDS)}; no deduplication"
    )


# ---------------------------------------------------------------------------
# Candidate narrowing
# ---------------------------------------------------------------------------
# Two mechanisms, because neither covers both term shapes on its own.
#
# Bounded terms narrow through the FTS5 index: its unicode61 tokenizer
# lowercases and splits on punctuation, which is exactly ``norm_text``, so
# every bounded match is guaranteed to be a candidate.
#
# Substring terms cannot use the index at all. A substring may fall inside a
# token ("corporate cuts" contains "rate cut"), and no tokenised query can
# reach it. Those terms scan with SQL ``LIKE``, which runs in C over the stored
# column — seconds, against the sixty a Python scan of the corpus costs.

_LIKE_SPECIALS = re.compile(r"([%_\\])")


def fts_match_query(specs: Iterable[TermSpec]) -> str | None:
    """Build the FTS5 ``MATCH`` expression for the bounded terms.

    Returns ``None`` when no term is bounded, in which case the caller relies
    on :func:`like_predicates` alone.
    """
    clauses = []
    for spec in specs:
        if spec.substring:
            continue
        tokens = _NONWORD.sub(" ", spec.term.lower()).split()
        if not tokens:
            continue
        # Quoted so multi-word terms match as a phrase rather than as an
        # implicit AND of tokens anywhere in the document.
        clauses.append('"' + " ".join(tokens) + '"')
    return " OR ".join(clauses) if clauses else None


def like_predicates(specs: Iterable[TermSpec]) -> tuple[str, list[str]]:
    """Build the ``LIKE`` predicate and parameters for the substring terms.

    ``LIKE`` is ASCII case-insensitive in SQLite by default, which matches the
    ``IGNORECASE`` confirmation regex for the ASCII terms these questions use.

    Returns:
        A ``(sql, params)`` pair. ``sql`` is empty when no term is a substring
        term.
    """
    clauses: list[str] = []
    params: list[str] = []
    for spec in specs:
        if not spec.substring:
            continue
        clauses.append("body LIKE ? ESCAPE '\\'")
        params.append("%" + _LIKE_SPECIALS.sub(r"\\\1", spec.term) + "%")
    return " OR ".join(clauses), params


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
def parse_pub_date(value: Any) -> date:
    """Parse an AFR ``PUBLICATIONDATE`` (``YYYYMMDD``) into a date.

    Accepts the int and str forms both present across the corpus.

    Raises:
        ValueError: if the value is not eight digits.
    """
    raw = str(value).strip()
    if len(raw) != 8 or not raw.isdigit():
        raise ValueError(f"unparseable PUBLICATIONDATE: {value!r}")
    return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))


#: Sentinel stored for a record whose publication date is absent or malformed.
#: Chosen so that any lower bound excludes it: an undated record cannot honestly
#: be placed in a year or a month, but it is still part of the corpus and still
#: counts toward an unscoped total — which is exactly how the published
#: reference counts treat it, since all of them are date-scoped.
UNDATED = 0


def pub_date_stamp(value: Any) -> int:
    """Return ``YYYYMMDD`` as an int, or :data:`UNDATED` if unparseable.

    A handful of records across the corpus carry an empty ``PUBLICATIONDATE``.
    Dropping them would change unscoped counts; guessing a date for them would
    invent evidence. They are kept, marked, and reported at ingest.
    """
    try:
        parsed = parse_pub_date(value)
    except ValueError:
        return UNDATED
    return parsed.year * 10_000 + parsed.month * 100 + parsed.day


def period_key(d: date, group_by: str) -> str:
    """Render the grouping key for a date: ``'2020'`` or ``'2020-05'``."""
    if group_by == "year":
        return f"{d.year:04d}"
    if group_by == "month":
        return f"{d.year:04d}-{d.month:02d}"
    raise ValueError(f"unsupported grouping: {group_by!r}")


# ---------------------------------------------------------------------------
# Sentiment cues
# ---------------------------------------------------------------------------
# Evidence, not a verdict. These lists let a tool report *what* directional
# language an article contains and in what balance; the fine-tuned synthesis
# model assigns the label and the market direction from that evidence
# (FR-5.2, CON-7). Keeping them here as plain data makes them inspectable and
# directly testable, which a prompt-embedded list would not be.
#
# Terms are matched with bounded semantics over ``norm_text`` and chosen for
# the register the AFR markets desk actually uses.

POSITIVE_CUES: tuple[str, ...] = (
    "rally", "rallied", "surge", "surged", "soar", "soared", "jump", "jumped",
    "gain", "gains", "gained", "climb", "climbed", "rise", "rose", "risen",
    "advance", "advanced", "rebound", "rebounded", "recover", "recovery",
    "recovered", "upgrade", "upgraded", "outperform", "outperformed", "beat",
    "boost", "boosted", "optimism", "optimistic", "confidence", "buoyant",
    "bullish", "strength", "strengthened", "record high", "stimulus", "upbeat",
)

NEGATIVE_CUES: tuple[str, ...] = (
    "fall", "fell", "fallen", "slump", "slumped", "plunge", "plunged", "sink",
    "sank", "tumble", "tumbled", "slide", "slid", "drop", "dropped", "decline",
    "declined", "loss", "losses", "weak", "weaker", "weakness", "downgrade",
    "downgraded", "underperform", "underperformed", "miss", "missed", "warn",
    "warned", "warning", "pressure", "pressured", "concern", "concerns",
    "fear", "fears", "risk", "risks", "uncertainty", "bearish", "selloff",
    "sell-off", "recession", "downturn", "slowdown", "cut", "cuts",
)

HEDGE_CUES: tuple[str, ...] = (
    "but", "however", "although", "though", "despite", "nevertheless",
    "meanwhile", "on the other hand", "mixed", "caution", "cautious",
    "cautiously", "sceptical", "skeptical", "doubt", "doubts", "unconvinced",
    "questioned", "if", "unless", "may", "might", "could",
)


def find_cues(text: str, cues: Iterable[str]) -> list[str]:
    """Return the cue terms present in ``text``, in cue-list order, deduped.

    Matching is bounded and case-insensitive, via :func:`norm_text`, so
    "cuts" does not fire on "cutscene" and punctuation is irrelevant.
    """
    haystack = norm_text(text)
    return [cue for cue in cues if f" {norm_text(cue).strip()} " in haystack]
