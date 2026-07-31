"""Concurrency test — AC-7.

The harness sends up to three hidden questions concurrently, so this covers a
scored failure mode, not a theoretical one (NFR-3).

TODO(build step 3): implement.
"""

from __future__ import annotations

# TODO(build step 3): issue three simultaneous POST /query requests with
# distinguishable questions; assert three correctly matched responses and no
# cross-contamination of `tool_trace` between them.
#
# TODO(build step 3): assert no mutable module-level state participates — the
# clearest version of this is running the same three requests twice and getting
# identical results (NFR-3.2).
