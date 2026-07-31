"""Ask the agent a question from the terminal.

    python ask.py "What was the RBA cash rate on 18 June 2024?"
    python ask.py                     # interactive, one question per line

Runs the real graph in-process, so what you see is what ``POST /query`` would
return — the same nodes, middleware, deadline and tool budget.

The ``--tools`` flag exists because the supplied server runs with
``--max-model-len 2048``, and the orchestrator prompt plus all twelve tool
schemas is roughly 5,750 tokens. Until the server is restarted with a larger
window the full agent cannot be invoked at all, so this defaults to a reduced
tool set and a short prompt that fit. Pass ``--tools all`` once the window is
raised; that is the real configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from src.config import load_settings  # noqa: E402
from src.context import QueryContext  # noqa: E402
from src.graph import _package, graph as full_graph  # noqa: E402
from src.middleware import DEFAULT_MIDDLEWARE  # noqa: E402
from src.models import build_reasoning_model  # noqa: E402
from src.state import GraphState, ReasoningState  # noqa: E402
from src.synthesis import synthesize  # noqa: E402
from src.tools import TOOLS  # noqa: E402

# Enough capability to answer real RBA and ASX questions inside 2048 tokens.
# AFR tools carry the largest schemas and are dropped first.
_SMALL_TOOLS = ("rba_rate_at", "rba_hold_runs", "rba_decision_stats", "asx_return")

_SMALL_PROMPT = """\
You answer Australian financial-market questions using tools.

Never compute anything yourself - no counting, no arithmetic, no date maths.
Ask the tool for the finished number. Never state a figure that did not come
back from a tool. Use the smallest number of calls that covers the question,
then stop. If a call fails, fix the arguments once or try another tool; do not
repeat the same call.
"""


def _reduced_graph():
    """The real topology, with a tool set and prompt that fit in 2048 tokens."""
    tools = [t for t in TOOLS if t.name in _SMALL_TOOLS]
    agent = create_agent(
        build_reasoning_model(),
        tools,
        system_prompt=_SMALL_PROMPT,
        state_schema=ReasoningState,
        context_schema=QueryContext,
        middleware=list(DEFAULT_MIDDLEWARE),
        name="reason",
    )

    async def reason(state, runtime=None):
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=state["question"])]},
            context=getattr(runtime, "context", None),
        )
        messages = result.get("messages") or []
        return {
            "messages": messages,
            "steps": sum(
                1 for m in messages if m.__class__.__name__ == "AIMessage"
            ),
            "tool_trace": list(result.get("tool_trace") or []),
        }

    builder = StateGraph(GraphState, context_schema=QueryContext)
    builder.add_node("reason", reason)
    builder.add_node("synthesize", synthesize)
    builder.add_node("package", _package)
    builder.add_edge(START, "reason")
    builder.add_edge("reason", "synthesize")
    builder.add_edge("synthesize", "package")
    builder.add_edge("package", END)
    return builder.compile()


async def ask(graph, question: str, settings, *, show_trace: bool) -> None:
    ctx = QueryContext(
        request_id="cli",
        deadline=time.monotonic() + settings.request_deadline_seconds,
        tool_budget=settings.max_tool_calls,
    )
    started = time.monotonic()
    try:
        result = await graph.ainvoke({"question": question}, context=ctx)
    except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not crash
        print(f"\n  FAILED: {type(exc).__name__}: {exc}\n", file=sys.stderr)
        return
    elapsed = time.monotonic() - started

    trace = result.get("tool_trace") or []
    print(f"\n\033[1mANSWER\033[0m  ({elapsed:.1f}s, "
          f"{result.get('steps', 0)} steps, {len(trace)} tool calls)")
    print(result.get("answer", ""))

    if show_trace and trace:
        print("\n\033[1mTOOL TRACE\033[0m")
        for entry in trace:
            print(f"  {entry['tool']}({json.dumps(entry['args'])})")
            print(f"    -> {str(entry['result'])[:300]}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument(
        "--tools",
        choices=("small", "all"),
        default="small",
        help="'small' fits the 2048-token server; 'all' needs a larger window",
    )
    parser.add_argument("--trace", action="store_true", help="show the tool trace")
    parser.add_argument("--debug", action="store_true", help="show library logs")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.debug else logging.ERROR,
        format="%(levelname)s %(name)s %(message)s",
    )

    settings = load_settings()
    print(f"  brain: {settings.agent_brain_model}  @ {settings.agent_brain_base_url}")
    print(f"synth'd: {settings.domain_ft_model}  @ {settings.domain_ft_base_url}")
    print(f"   mode: {settings.domain_predict_mode}  tools: {args.tools}")
    if settings.is_mock_synthesis:
        print("  !! DOMAIN_PREDICT_MODE=mock - answers are a stand-in, not the model")
    if (
        settings.domain_ft_model == settings.agent_brain_model
        and settings.domain_ft_base_url == settings.agent_brain_base_url
    ):
        # Easy to miss otherwise: the roles stay separate and the graph is
        # unchanged, but both ends resolve to the same base model, so the
        # submission carries no fine-tuned-model evidence (AC-12).
        print("  !! synthesis is using the SAME model as the brain - not fine-tuned")

    graph = full_graph if args.tools == "all" else _reduced_graph()

    if args.question:
        asyncio.run(
            ask(graph, " ".join(args.question), settings, show_trace=args.trace)
        )
        return 0

    print("\nType a question, or Ctrl-D to quit.")
    while True:
        try:
            question = input("\n\033[1m>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question:
            asyncio.run(ask(graph, question, settings, show_trace=args.trace))


if __name__ == "__main__":
    raise SystemExit(main())
