"""Deterministic-calculation tests — AC-5.

No model in the loop. The calculation helpers are called directly against known
inputs, because this is precisely where the brief's §10 examples fail: a
statistic that was never actually computed (FR-3.4, FR-4.4).

TODO(build step 4): implement — blocked on the tool layer (BLK-2).
"""

from __future__ import annotations

# TODO(build step 4): RBA decision counts over a fixture range — total, changed,
# increases, decreases, holds (brief §10 example 1).
#
# TODO(build step 4): longest unchanged run, including its start and end dates,
# over a fixture with a deliberately ambiguous tie (brief §10 example 2).
#
# TODO(build step 4): ASX percentage and absolute change, including the
# non-trading-day case where the nearest prior price must be used.
#
# TODO(build step 4): ranking, with an explicit tie-break rule so the result is
# stable rather than merely correct-on-average.
