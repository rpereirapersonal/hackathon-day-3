"""Response-contract tests — AC-1, AC-2, AC-3.

Covers FR-1.x and NFR-4: the shape of every response, on every path.

TODO(build step 1): implement.
"""

from __future__ import annotations

# TODO(build step 1): GET /health returns 200 within milliseconds with the model
# gateway deliberately unreachable (AC-1).
#
# TODO(build step 1): POST /query with a well-formed question returns JSON with a
# non-empty `answer`, an integer `steps`, and an ordered `tool_trace` (AC-2).
#
# TODO(build step 3): malformed input, an unreachable gateway, and a
# deliberately failing tool each still yield 200 with a valid, non-empty
# `answer` stating the limitation (AC-3).
