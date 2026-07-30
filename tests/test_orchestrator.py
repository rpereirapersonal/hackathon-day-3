"""Orchestrator construction and planning-behaviour tests.

The prompt in ``src/orchestrator.py`` is doing scored work — it is what steers
the reasoning brain away from the brief's §10 failure modes — so the routing
rules it encodes are asserted here rather than assumed.

TODO(build step 2): implement.
"""

from __future__ import annotations

# TODO(build step 2): build_orchestrator(model=fake_model) compiles, binds the
# TOOLS registry, and exposes QueryContext as its context schema.
#
# TODO(build step 4): routing — a question asking for an RBA decision count
# drives a structured RBA tool call, not an AFR retrieval call. This is the
# single most valuable behavioural test in the suite (brief §10 example 1).
#
# TODO(build step 4): a question answerable in one call does not consume the
# whole budget.
#
# TODO(build step 4): a cross-dataset question composes RBA, AFR and ASX calls
# within the 3-call target (FR-4.5, NFR-2.1).
#
# TODO(build step 3): after a tool returns a structured error, the loop adapts
# or stops — it does not retry the identical call (FR-3.6).
