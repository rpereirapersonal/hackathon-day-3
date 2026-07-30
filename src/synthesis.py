"""The fine-tuned Nemotron synthesis node — writes the final answer.

Receives the original question plus the accumulated verified tool results, and
nothing else (FR-5.1). Specifically *not* the reasoning transcript: excluding
Qwen's deliberation is what stops intermediate speculation leaking into the
answer.

Bound with no tools and unreachable from the tool loop (FR-5.2, CON-7). If the
evidence is thin, that is a reasoning-loop outcome to state plainly, not
something synthesis may paper over (CON-5).

TODO(build step 3): implement the node, the prompt, and the deterministic
fallback.
"""

from __future__ import annotations

# TODO(build step 3): async def synthesize(state, runtime) -> dict
#
# Prompt requirements, each of them scored:
#   - Answer every component the question explicitly requested; multi-part
#     questions earn partial credit per component (FR-5.3).
#   - Make no claim absent from the supplied evidence (FR-5.4).
#   - Where the evidence does not support an answer, state the limitation
#     rather than filling it in (CON-5).
#   - No tool names, prompt fragments or internal scaffolding in the answer
#     text (FR-5.6).
#   - Direct and concise; bounded output length to hold the ~15s synthesis
#     slice of the latency budget (NFR-1.3).
#
# TODO(build step 3): mock mode. When Settings.is_mock_synthesis, return a
# deterministic stand-in assembled from tool_trace so the full pipeline is
# testable before the adapter is served (FR-5.5). Warn loudly at startup
# whenever mock is active and echo the mode in /health — shipping in mock
# forfeits the fine-tuned-model evidence entirely (AC-12).
#
# TODO(build step 3): fallback. If the synthesis model is unavailable or slow,
# assemble a deterministic template answer from tool_trace and return HTTP 200
# with the limitation stated. Never an empty answer, never an invented figure
# (FR-1.4, CON-5).
