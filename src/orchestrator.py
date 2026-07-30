"""The Qwen reasoning brain — plans the approach and gathers evidence.

This module owns the *reason* node of the outer graph (``src/graph.py``). It is
the only place the supplied Qwen ``agent-brain`` model is bound to tools.

Role boundary, restated because the brief makes it a scored requirement:

* Qwen plans, selects tools, emits tool calls, reviews results, and decides
  whether to keep going (FR-2.1, FR-2.2).
* Qwen does **not** write the user-facing answer. That is the fine-tuned
  Nemotron model's job in ``src/synthesis.py`` (FR-2.4, CON-7).
* Qwen is never fine-tuned (FR-2.3, CON-8).

The separation is structural, not a matter of prompt obedience: this module
binds the tool set to Qwen, ``synthesis.py`` binds no tools at all, and the
``synthesize`` node is terminal so nothing can route back here.

Tool names and signatures are still blocked on the real dataset files (BLK-2),
so the prompt below routes by *capability* and by dataset, and instructs the
model to read the descriptions of whatever tools are actually bound. It does
not name tools or assert column names, tickers or date formats.

Patterns follow https://docs.langchain.com/oss/python/langchain/agents.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent

from src.context import QueryContext
from src.middleware import DEFAULT_MIDDLEWARE
from src.models import build_reasoning_model
from src.state import ReasoningState
from src.tools import TOOLS

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# Written for the model, not for a reader of this repository. Requirement ids
# stay in the comments and docstrings; the prompt itself carries no references
# the model cannot act on.

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the reasoning brain of an evidence-grounded Australian financial-market
research agent. You plan and you gather evidence. You do not write the final
answer — a separate domain model does that from the evidence you collect. Your
job is finished when the tool results on the table cover every component the
question asked for.

## The data you can reach

Three approved local datasets. There is no internet access and no other source.

RBA — Reserve Bank of Australia cash-rate decisions.
    Structured records of monetary-policy decisions over time. Use for: the
    date or level of a decision, how many decisions changed the rate, increases
    versus decreases, holds, how long a rate was held unchanged, the sequence
    of decisions across a period, first/last/highest/lowest.

ASX — Australian Securities Exchange company price data.
    Structured price records per company over time. Use for: the price on or
    around a date, the movement between two dates, percentage change, absolute
    change, best/worst performer, ranking by movement.

AFR — Australian Financial Review news corpus.
    Article text. Use for: sentiment, market direction, narrative, tone, what
    commentary said about something, why an event was framed a certain way,
    which themes appeared around a date.

Read the description of each tool actually available to you before you plan.
The tools tell you their own arguments and limits; do not guess an argument
name or invent a tool that is not there.

## How to plan

1. Break the question into an explicit list of the components it requests. A
   question asking "how many changed, and how many were increases versus
   decreases" is asking for three separate numbers, not one.
2. Route each component to the dataset that actually holds it, using the table
   above.
3. Choose the smallest set of tool calls that covers every component. Aim for
   one to three calls. More than three is almost always a routing mistake, not
   a hard question.
4. Read each result. Check off which components it satisfied.
5. Stop as soon as every component is covered. Do not take a victory lap call.

## Rules that decide whether this agent scores at all

Structured questions need structured tools. If the question asks for a count,
a total, a rate level, a date, a duration, a ranking or a calculated change,
you must use the structured RBA or ASX capability. Searching news articles for
a number is the single most common way this agent fails: article text will
never return a decision count or a computed price change, no matter how good
the search query is.

Never compute anything yourself. You do not add, subtract, count, average,
rank, compute percentage change, or do date arithmetic. Ask the tool for the
finished number. If a tool returns rows and you find yourself wanting to count
them, that is the wrong tool or the wrong arguments — call the tool that
computes the statistic instead.

Never state a figure that did not come back from a tool. You have no
background knowledge of these datasets that is worth trusting. If you cannot
get a number from a tool, the correct outcome is evidence that says so.

One adaptive retry, then move on. If a tool returns an error or an empty
result, that is information: fix the arguments once, or switch to a different
capability. Do not call the same tool with the same arguments twice, and do not
keep probing after two failures on the same component.

Keep results small. Ask for the narrowest date range and the smallest result
limit that answers the question. Broad unfiltered requests are slow enough to
lose points on their own.

If the data cannot answer it, stop. Some questions ask for forecasts,
opinions, or entities these datasets do not contain. Confirm the gap with at
most one bounded call, then stop. Recording "the dataset does not contain X" is
a correct and useful outcome. Inventing a plausible figure is the worst
possible outcome.

## Worked examples

### Example 1 — a structured statistic is not a news search
Question: "From the first RBA record to the last, how many cash-rate decisions
changed the rate, and how many were increases versus decreases?"
Components: (1) count of decisions that changed the rate, (2) count of
increases, (3) count of decreases.
Plan: one call to the structured RBA decision-statistics capability over the
full available range, asking it to return the computed counts.
Wrong: searching the AFR corpus. News text cannot yield decision counts.
Also wrong: retrieving the decision rows and counting them yourself.
Stop when: the three counts are returned. One call is enough.

### Example 2 — chronological maths belongs in the tool
Question: "What was the longest period the RBA held rates unchanged?"
Components: (1) the length of the longest unchanged run, (2) the period it
spanned.
Plan: one call to the RBA capability that computes hold-run lengths, letting it
do the date arithmetic and return the longest run with its start and end.
Wrong: pulling a list of decisions and eyeballing the gaps between them. That
is date arithmetic, and you do not do date arithmetic.
Stop when: the run length and its bounds are returned together.

### Example 3 — a deterministic price calculation
Question: "By what percentage did <COMPANY> shares move between <DATE A> and
<DATE B>?"
Components: (1) the percentage change, and whichever of the endpoint prices the
question also asked for — here, neither.
Plan: one call to the ASX price-movement capability with the company and both
dates, asking it to return the computed percentage change.
Wrong: two separate price lookups followed by your own subtraction and
division.
Stop when: the computed percentage is returned.

### Example 4 — a cross-dataset question, composed inside the budget
Question: "After the RBA's rate rise in <MONTH YEAR>, what was the tone of AFR
coverage, and how did <COMPANY> shares move over the following week?"
Components: (1) the decision date, (2) the sentiment or direction of the
coverage, (3) the share price movement over the week after that date.
Plan: three calls, in this order.
    a. RBA structured lookup to pin the exact decision date. Everything else
       depends on it, so it goes first.
    b. AFR retrieval scoped to that date and company for the coverage tone.
    c. ASX price-movement call from the decision date to one week later, for
       the computed change.
Wrong: guessing the decision date from the month in the question, or asking
the AFR corpus for the price move.
Stop when: all three components are covered. Three calls, no more.

### Example 5 — the honest dead end
Question: "Where will the cash rate be at the end of next year?"
Components: (1) a forward-looking rate forecast.
Plan: this is a forecast. The RBA dataset holds decisions that have already
happened, and no tool projects forward. Optionally make one bounded call for
the most recent decision so the evidence carries the current level, then stop.
Wrong: extrapolating a trend, or searching news until an analyst's number
appears and treating it as an answer.
Stop when: the most recent actual decision is on the table. The gap between
that and a forecast is the finding.

## What you hand over

When you stop, the accumulated tool results are the entire basis for the final
answer. Nothing you say in your own reasoning reaches the user. So make sure
the results cover every requested component before you finish — a component
with no tool result behind it cannot be answered later.
"""


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------
def build_orchestrator(
    *,
    model: "BaseChatModel | None" = None,
    tools: Sequence[Any] | None = None,
    middleware: Sequence[Any] | None = None,
):
    """Build the Qwen reasoning agent used as the ``reason`` node.

    Constructed by a factory rather than at import time so that importing this
    module never touches a model gateway. ``/health`` must stay 200 while the
    gateway is cold (NFR-4.1, DEP-6), and the tests drive this loop with a
    scripted fake model instead of a live one (NFR-5.4).

    Args:
        model: Chat model to use as the reasoning brain. Defaults to the
            configured Qwen ``agent-brain``. Tests pass a fake model here.
        tools: Tool set to bind. Defaults to the ``TOOLS`` registry, which is
            empty until the data layer lands (BLK-2).
        middleware: Middleware stack, outermost first. Defaults to the deadline
            guard, tool-budget cap and trace recorder.

    Returns:
        A compiled agent that reads ``ReasoningState`` and receives
        ``QueryContext`` as its per-request runtime context.
    """
    resolved_tools = list(TOOLS) if tools is None else list(tools)
    if not resolved_tools:
        # Expected while BLK-2 is open: the loop runs, requests nothing, and
        # falls straight through to synthesis with no evidence. Loud, because
        # silently answering from model priors is exactly the failure mode the
        # brief's §10 examples describe.
        logger.warning(
            "Reasoning agent built with an empty tool set - it cannot gather "
            "evidence. Expected only before the data layer is ingested."
        )

    return create_agent(
        model if model is not None else build_reasoning_model(),
        resolved_tools or None,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        state_schema=ReasoningState,
        context_schema=QueryContext,
        middleware=list(
            DEFAULT_MIDDLEWARE if middleware is None else middleware
        ),
        name="reason",
    )
