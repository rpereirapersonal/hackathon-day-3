"""The outer graph: reason -> synthesize -> package.

Three nodes, one linear path. The linearity is the point — ``synthesize`` is
terminal apart from its edge to ``package``, so no synthesis output can route
back into the reasoning loop, and ``package`` reads ``answer`` only from the
synthesis node's output (FR-5.2, CON-7, AC-4).

The reasoning agent is a compiled subgraph used as a node rather than being the
whole graph, which makes the Qwen -> Nemotron hand-off a visible edge and keeps
deadline and packaging concerns out of the agent's control flow
(architecture.md §4).

TODO(build step 2): wire the nodes and export the compiled ``graph``.
"""

from __future__ import annotations

# TODO(build step 2): build with StateGraph(GraphState, context_schema=QueryContext):
#   - node "reason":     orchestrator.build_orchestrator(), invoked as a subgraph
#   - node "synthesize": synthesis.synthesize
#   - node "package":    assemble {answer, steps, tool_trace} via schemas.py
#   - edges: START -> reason -> synthesize -> package -> END
#   - compile with a recursion limit that cannot exceed MAX_TOOL_CALLS (NFR-2.2)
#
# All three reasoning-loop termination paths (no more tools requested, budget
# exhausted, deadline reached) fall through to synthesize. None is an error path
# (FR-2.5, NFR-2.3).
