"""AFR news tools — exact counting, article lookup, sentiment evidence, search.

Thin, bounded wrappers over ``src/retrieval.py``. The engine is shared with the
ingest self-check, so a tool cannot disagree with the counts ingest verified.

Scope discipline matters more here than anywhere else in the tool layer. Two of
these tools answer questions about *tone and narrative*; the other two answer
questions about *how many records*. Sending a counting question to semantic
search is the single most common way this agent scores zero, so every docstring
below states what its tool cannot do and names the tool that can — routing is
defended at the tool descriptions, not only in the system prompt.

None of these tools labels sentiment. ``afr_sentiment_evidence`` returns the
article plus the directional language it contains; assigning
positive / negative / mixed and the likely market direction is the fine-tuned
synthesis model's job (FR-5.2, CON-7).
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from langchain.tools import tool
from pydantic import Field

from src import retrieval
from src.db import IndexMissingError
from src.frames import ArtifactMissingError
from src.text import parse_terms

logger = logging.getLogger(__name__)

DataError = (IndexMissingError, ArtifactMissingError)


def _failure(tool_name: str, message: str) -> dict[str, Any]:
    """Return a structured error rather than raising (FR-3.6).

    The reasoning brain reads this, so it says what went wrong and what to do
    differently — not a stack trace.
    """
    logger.warning("%s failed: %s", tool_name, message)
    return {"error": message, "tool": tool_name}


def _resolve_window(
    year: int | None, date_from: str | None, date_to: str | None
) -> tuple[str | None, str | None]:
    """Turn the convenience ``year`` argument into an explicit date window."""
    if year is not None:
        return f"{year:04d}0101", f"{year:04d}1231"
    return date_from, date_to


@tool(parse_docstring=True)
async def afr_count_matches(
    terms: Annotated[list[str], Field(min_length=1, max_length=8)],
    group_by: Literal["none", "year", "month"] = "none",
    year: Annotated[int | None, Field(ge=2015, le=2021)] = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top_n: Annotated[int, Field(ge=1, le=20)] = 5,
) -> dict[str, Any]:
    """Count how many AFR articles mention given terms, once per article.

    Use this for every "how many articles mention X", "which year had the most
    coverage of X" or "how many records match X" question. It returns finished
    counts computed over the whole corpus — never a list of articles to count
    yourself.

    A term is matched case-insensitively and bounded by word edges, so
    "unemployment" does not fire on "unemployments". Add a trailing asterisk to
    match the term anywhere, including inside a longer word and across
    inflections: "cash rate*" also counts "cash rates", and "rate cut*" also
    counts "rate cuts". Use the asterisk for multi-word phrases and plain form
    for single words. Multiple terms are combined with OR, and an article
    matching several of them still counts once.

    This tool cannot judge tone or sentiment, quote an article, or return
    article text — use afr_sentiment_evidence or afr_search for those.

    Args:
        terms: Search terms, one to eight. Trailing asterisk means match
            anywhere including inside longer words.
        group_by: Return counts per calendar year, per calendar month, or a
            single total.
        year: Restrict to one calendar year, 2015 to 2021. A shortcut for
            setting both date bounds.
        date_from: Inclusive start date, YYYY-MM-DD.
        date_to: Inclusive end date, YYYY-MM-DD.
        top_n: How many groups to return when grouping, ranked by count
            descending.
    """
    try:
        specs = parse_terms(terms)
        window_from, window_to = _resolve_window(year, date_from, date_to)
        result = await retrieval.count_matches(
            specs,
            group_by=group_by,
            date_from=window_from,
            date_to=window_to,
            top_n=top_n,
        )
    except DataError as exc:
        return _failure("afr_count_matches", str(exc))
    except ValueError as exc:
        return _failure("afr_count_matches", str(exc))

    payload: dict[str, Any] = {
        "total_records": result.total,
        "terms": list(terms),
        "convention": result.convention,
    }
    if window_from or window_to:
        payload["window"] = {"from": window_from, "to": window_to}
    if result.groups:
        payload["groups"] = result.groups
        payload["group_by"] = group_by
    return payload


@tool(parse_docstring=True)
async def afr_article_lookup(
    headline: str, publication_date: str
) -> dict[str, Any]:
    """Fetch one specific AFR article by its exact headline and publication date.

    Use this whenever the question names an article — a quoted headline plus a
    date. It is an exact lookup, not a search, so it is the right tool even
    when the headline is long. Comparison ignores case, punctuation and quote
    style.

    Returns the headline, date, a bounded excerpt of the opening, and how many
    records share that headline and date.

    This tool cannot count articles, and it cannot find an article you can only
    describe by topic — use afr_count_matches or afr_search for those.

    Args:
        headline: The article headline, as quoted in the question.
        publication_date: Publication date, YYYY-MM-DD.
    """
    try:
        article = await retrieval.lookup_article(headline, publication_date)
    except DataError as exc:
        return _failure("afr_article_lookup", str(exc))
    except ValueError as exc:
        return _failure("afr_article_lookup", str(exc))

    if article is None:
        # A miss must not read as absent data — surface near matches so a date
        # slip or a truncated title is visibly recoverable.
        return {
            "found": False,
            "message": (
                f"No article with that headline on {publication_date}. "
                "Nearest headlines by wording are listed; check the date."
            ),
            "near_matches": retrieval.suggest_headlines(headline),
        }
    return {"found": True, **article}


@tool(parse_docstring=True)
async def afr_sentiment_evidence(
    headline: str, publication_date: str
) -> dict[str, Any]:
    """Get one AFR article plus the directional language it uses, as evidence.

    Use this for questions asking about an article's sentiment, its tone, or the
    market direction it implies. Retrieve the evidence and stop — do not decide
    the label yourself. The final answer is written from this evidence
    downstream, and your own reading of the article is not what gets returned.

    Returns the article excerpt, the positive, negative and hedging language
    found in it, and the balance between positive and negative cues.

    This tool cannot count articles or return a price move — use
    afr_count_matches or the ASX tools for those.

    Args:
        headline: The article headline, as quoted in the question.
        publication_date: Publication date, YYYY-MM-DD.
    """
    try:
        evidence = await retrieval.sentiment_evidence(headline, publication_date)
    except DataError as exc:
        return _failure("afr_sentiment_evidence", str(exc))
    except ValueError as exc:
        return _failure("afr_sentiment_evidence", str(exc))

    if evidence is None:
        return {
            "found": False,
            "message": (
                f"No article with that headline on {publication_date}. "
                "Nearest headlines by wording are listed; check the date."
            ),
            "near_matches": retrieval.suggest_headlines(headline),
        }
    return {"found": True, **evidence}


@tool(parse_docstring=True)
async def afr_search(
    query: str,
    k: Annotated[int, Field(ge=1, le=10)] = 5,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Find AFR articles about a topic by meaning, when no headline is given.

    Use this only when the question describes coverage by subject rather than
    naming an article — "what did commentary say about X around <date>". Scope
    it with the date window whenever the question implies one.

    Returns up to k bounded excerpts ranked by relevance.

    This tool cannot count anything. Relevance ranking says nothing about how
    many articles mention a term, so a count taken from these results will be
    wrong — use afr_count_matches. It also cannot supply a cash-rate level or a
    computed price change; those come from the RBA and ASX tools.

    Args:
        query: What to look for, in plain words.
        k: How many articles to return, one to ten.
        date_from: Inclusive start date, YYYY-MM-DD.
        date_to: Inclusive end date, YYYY-MM-DD.
    """
    try:
        hits = await retrieval.search(
            query, k=k, date_from=date_from, date_to=date_to
        )
    except DataError as exc:
        return _failure("afr_search", str(exc))
    except ValueError as exc:
        return _failure("afr_search", str(exc))

    return {
        "query": query,
        "articles": [
            {
                "headline": hit.headline,
                "publication_date": hit.publication_date,
                "excerpt": hit.excerpt,
                "relevance": None if hit.score is None else round(hit.score, 4),
            }
            for hit in hits
        ],
        "note": (
            "Relevance-ranked sample, not a count. Use afr_count_matches for "
            "how many articles mention a term."
        ),
    }


AFR_TOOLS = (
    afr_count_matches,
    afr_article_lookup,
    afr_sentiment_evidence,
    afr_search,
)
