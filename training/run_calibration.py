"""Calibration harness over the 15 public questions.

Unblocked — the Participant Package and all three datasets are present, and the
tool layer is verified. What remains is the serving layer: this harness posts to
``/query``, which does not exist yet (build step 1).

Converts the latency budget in ``architecture.md`` §7 from a design allocation
into a measurement, and the routing rules in ``orchestrator.py`` from an
intention into a score.

Run by path, not as a module — ``training/`` is deliberately not a package::

    python training/run_calibration.py

Two rules that are easy to violate and expensive to violate:
  - Pass **only** the ``prompt`` field to the agent. Nothing else from
    ``public_questions.jsonl`` may reach the service.
  - Never map a question id to an answer, anywhere (CON-9, AC-13).

TODO(build step 5): implement.
"""

from __future__ import annotations

# TODO(build step 5): read public_questions.jsonl; POST each `prompt` to
# /query; record per-question latency and the response.
#
# TODO(build step 5): score each response against grading.components[].
# expected_fact, per component rather than per question — the official grader
# awards partial credit, so a per-question pass/fail here would hide most of the
# signal (brief §6C).
#
# TODO(build step 5): report per-stage timings (planning, tools, synthesis,
# total) and the tool-call count distribution, to confirm the <=3 call target and
# the 50s internal budget are realistic on the target hardware (AC-9).
