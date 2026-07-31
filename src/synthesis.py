"""The fine-tuned Nemotron synthesis node — writes the final answer.

Receives the original question plus the accumulated verified tool results, and
nothing else (FR-5.1). Specifically *not* the reasoning transcript: excluding
Qwen's deliberation is what stops intermediate speculation leaking into the
answer.

Bound with no tools and unreachable from the tool loop (FR-5.2, CON-7). If the
evidence is thin, that is a reasoning-loop outcome to state plainly, not
something synthesis may paper over (CON-5).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

from src.config import Settings, load_settings
from src.models import build_synthesis_model

if TYPE_CHECKING:
    from src.state import GraphState

logger = logging.getLogger(__name__)

# The served Qwen build emits a chain-of-thought preamble terminated by a
# literal ``</think>`` even with ``enable_thinking: false`` in the server's
# generation config, and the opening tag is frequently absent. Anything up to
# and including the last closing tag is deliberation, not answer — shipping it
# would put internal scaffolding in the user-facing text (FR-5.6).
_THINK_CLOSE = re.compile(r"(?s)^.*</think\s*>")

# When the model is cut off mid-deliberation the closing tag never arrives, so
# tag-stripping alone leaves the whole preamble in place. These match the way
# this build opens a reasoning block; text from the first such marker on is
# discarded, keeping whatever preceded it.
_THINK_OPEN = re.compile(
    r"(?im)^\s*(?:<think\s*>|here'?s?\s+(?:a|my)\s+thinking\s+process\b"
    r"|okay,?\s+(?:so\s+)?(?:let'?s|the\s+user)\b"
    r"|(?:first|step)\s*,?\s*(?:let'?s|I)\s+(?:need\s+to|should|will)\b)"
)

# Bounded so synthesis holds its ~15s slice of the latency budget (NFR-1.3).
# Generous enough that a cut-off answer is rare; the guard against a runaway
# preamble is the cleaner below, not a tight cap.
_MAX_TOKENS = 700

SYNTHESIS_SYSTEM_PROMPT = """\
You are a financial-market analyst writing the final answer for a research
agent. Another system has already gathered the evidence. Your only job is to
turn that evidence into the answer.

Rules, each of which decides whether the answer scores:

1. Answer every component the question asked for. A question asking "how many
   changed, and how many were increases versus decreases" wants three numbers.
   Address each one explicitly.
2. Use only the figures and facts in the evidence below. Every number in your
   answer must appear in the evidence. Do not calculate new numbers, do not
   adjust the ones you are given, and do not add context from your own
   knowledge of these markets.
3. Where the evidence does not cover something the question asked, say so
   plainly in one clause and move on. A stated gap is a correct answer. An
   invented figure is the worst possible outcome.
4. Never mention tools, evidence records, retrieval, prompts, or any internal
   machinery. Do not write "the tool returned" or "according to the evidence".
   Write the finding directly, as an analyst would.
5. Be direct and concise. Lead with the answer, then the supporting figures.
   No preamble, no restating the question, no closing summary.

Write only the answer text.
"""


def _clean(text: str) -> str:
    """Strip reasoning scaffolding from a model completion (FR-5.6).

    Two passes, because this build leaks deliberation in two different shapes.
    A completed block is bounded by ``</think>`` and everything before it goes.
    A block cut off by the token cap has no closing tag at all, so the opening
    marker is used instead and everything from it onwards goes.

    Returns "" when the response was nothing but deliberation. That is the
    correct outcome: the caller then falls back to a deterministic answer
    rather than publishing a chain of thought.
    """
    cleaned = _THINK_CLOSE.sub("", text or "").strip()
    if match := _THINK_OPEN.search(cleaned):
        cleaned = cleaned[: match.start()].strip()
    return cleaned


def _evidence_lines(tool_trace: list[Any]) -> list[str]:
    """One readable line per recorded tool result.

    Reads defensively: entries arrive from the trace channel as ``TypedDict``
    instances, but a middleware change or a test double could supply objects,
    and synthesis silently dropping evidence would be near-impossible to spot
    in a scored answer.
    """
    lines: list[str] = []
    for entry in tool_trace or []:
        if isinstance(entry, dict):
            result = entry.get("result", "")
        else:
            result = getattr(entry, "result", "")
        result = str(result).strip()
        if result:
            lines.append(f"- {result}")
    return lines


def _fallback_answer(question: str, evidence: list[str]) -> str:
    """Deterministic answer assembled without the model.

    Used when synthesis is unavailable, times out, or returns nothing. Returns
    the evidence as-is rather than an apology: a plain restatement of verified
    findings is worth partial credit, an error string is worth none. Never
    empty, never an invented figure (FR-1.4, CON-5).
    """
    if not evidence:
        return (
            "The available data does not support an answer to this question. "
            "No supporting records were retrieved from the RBA, ASX or AFR "
            "datasets."
        )
    return "Based on the available records:\n" + "\n".join(evidence)


def _mock_answer(question: str, evidence: list[str]) -> str:
    """Deterministic stand-in used before the adapter is served (FR-5.5)."""
    return "[mock synthesis] " + _fallback_answer(question, evidence)


async def synthesize(
    state: "GraphState", runtime: Any = None, *, settings: Settings | None = None
) -> dict[str, str]:
    """Write the final answer from the question and the verified evidence.

    Returns a state update carrying ``answer`` only. Never raises: every
    failure path degrades to a deterministic answer, because a 5xx or an empty
    answer scores zero (FR-1.4, FR-1.6).
    """
    cfg = settings or load_settings()

    question = (state.get("question") or "").strip()
    evidence = _evidence_lines(state.get("tool_trace") or [])

    if cfg.is_mock_synthesis:
        # Loud, because shipping in mock forfeits the fine-tuned-model
        # evidence entirely (AC-12).
        logger.warning(
            "DOMAIN_PREDICT_MODE=mock - answer is a deterministic stand-in, "
            "not the fine-tuned model. Set DOMAIN_PREDICT_MODE=llm before "
            "evaluation."
        )
        return {"answer": _mock_answer(question, evidence)}

    evidence_block = (
        "\n".join(evidence)
        if evidence
        else "(no supporting records were retrieved)"
    )
    prompt = (
        f"Question:\n{question}\n\n"
        f"Evidence gathered:\n{evidence_block}\n\n"
        "Write the final answer."
    )

    # The node's own slice of the request budget. The outer deadline still
    # governs the request as a whole; this stops a hung gateway from consuming
    # the remainder of it and forcing a hard timeout upstream (NFR-1.3).
    remaining = getattr(getattr(runtime, "context", None), "seconds_remaining", None)
    timeout = max(5.0, min(float(remaining), 20.0)) if remaining else 20.0

    try:
        model = build_synthesis_model(cfg).bind(max_tokens=_MAX_TOKENS)
        response = await asyncio.wait_for(
            model.ainvoke(
                [
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            ),
            timeout=timeout,
        )
        answer = _clean(str(response.content))
        if answer:
            return {"answer": answer}
        logger.warning("Synthesis returned an empty answer; using fallback.")
    except asyncio.TimeoutError:
        logger.warning("Synthesis exceeded %.1fs; using fallback.", timeout)
    except Exception:
        logger.exception("Synthesis failed; using fallback.")

    return {"answer": _fallback_answer(question, evidence)}
