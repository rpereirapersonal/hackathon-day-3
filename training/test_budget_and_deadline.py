"""Budget and deadline tests — AC-6, AC-8.

Covers NFR-1.2 and NFR-2. Both are enforced in code rather than requested in a
prompt, so both are testable without a live model.

TODO(build step 3): implement.
"""

from __future__ import annotations

# TODO(build step 3): a fake Qwen that requests tools forever terminates at the
# cap and still returns a valid answer (AC-6). Assert the refused call appears in
# `tool_trace` (FR-6.2) and that exhaustion is not an error path (NFR-2.3).
#
# TODO(build step 3): an artificially slowed pipeline hits the deadline and
# returns a degraded but valid answer rather than exceeding the latency band
# (AC-8).
