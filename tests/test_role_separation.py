"""Role-separation tests — AC-4.

The one test group that directly defends CON-7 and FR-2.4. Worth writing
carefully: "evidence that the final agent uses Qwen for planning/tool-calls and
routes verified results to fine-tuned Nemotron for final synthesis" is an
explicitly scored item (brief §6A).

TODO(build step 3): implement.
"""

from __future__ import annotations

# TODO(build step 3): assert the synthesis model is constructed with no tools
# bound (FR-5.2).
#
# TODO(build step 3): assert the returned `answer` originates from the synthesis
# node, not from the reasoning brain — drive the loop with a fake Qwen whose
# final message is a recognisable sentinel, and assert that sentinel never
# appears in the response `answer` (FR-2.4).
#
# TODO(build step 2): assert the graph has no edge out of `synthesize` other
# than to `package`, so synthesis cannot re-enter the tool loop.
