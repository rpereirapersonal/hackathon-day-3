"""Shared test fixtures.

Tests never contact a live model (NFR-5.4), so they are fast and work while the
gateway credentials are outstanding (BLK-3, BLK-4).

TODO(build step 2): implement the two fixtures everything else depends on.
"""

from __future__ import annotations

# TODO(build step 2): fake_model — a scripted fake chat model that returns a
# predetermined sequence of tool calls and then a final message. Makes graph
# structure, tool sequencing and loop termination testable with no gateway.
# Pass it to orchestrator.build_orchestrator(model=...).
#
# TODO(build step 2): fake_runtime — a lightweight stand-in exposing
# .context as a QueryContext, so tools and middleware can be invoked directly.
# Note that direct invocation bypasses argument validation, so argument bounds
# are asserted separately at the schema layer.
#
# TODO(build step 4): tmp_db — a small fixture database with known values, for
# the determinism tests (AC-5).
