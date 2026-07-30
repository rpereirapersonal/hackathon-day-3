"""FastAPI application — the agent endpoint declared in ``submission.json``.

Two routes, with very different obligations.

``GET /health`` is a hard gate: a non-200 during the organizers' pre-evaluation
check skips the team for zero points. It therefore makes no model, gateway or
database call, so a cold or broken upstream cannot convert a recoverable
degradation into total failure (FR-1.1, NFR-4, AC-1).

``POST /query`` owns the per-request deadline and the outermost fallback. No
path returns a 5xx, an empty ``answer``, or a malformed body — each of those
scores zero (FR-1.5, FR-1.6, CON-5).

TODO(build step 1): implement. This is the first thing built, because it makes
the hard gate and the scored contract provable before any agent logic exists.
"""

from __future__ import annotations

# TODO(build step 1): app = FastAPI(...)
#
# GET /health   -> 200 with a small static body. No upstream calls, ever.
#
# POST /query:
#   1. Validate against schemas.QueryRequest; assign a request id.
#   2. Start the hard deadline; build QueryContext (NFR-1.2).
#   3. await graph.ainvoke({"question": ...}, context=ctx) under a timeout.
#   4. Shape through schemas.QueryResponse and return 200.
#   Async throughout, so three concurrent requests are served without blocking
#   the event loop (NFR-3.1, AC-7).
#
# TODO(build step 1): wrap the handler so any unexpected exception still yields
# 200 with a valid contract-conformant body explaining the limitation (FR-1.6).
#
# TODO(build step 3): per-request diagnostic logging to LOG_DIR with the
# correlation id and per-stage timings — planning, each tool call, synthesis,
# total. Never log credentials (FR-6.3, NFR-6.4).
#
# Serving note (DEP-5): bind 0.0.0.0, not 127.0.0.1. A loopback-bound service is
# invisible to the tunnel and fails the health gate.
