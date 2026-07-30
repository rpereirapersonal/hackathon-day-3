"""The single definition of the scored JSON contract.

Every response — success, degraded, or internal error — is serialised through
these models, so no path can drift out of shape (FR-1.3, FR-1.5, CON-4).

TODO(build step 1): implement. Validate the shape against the organizers'
``validate.json`` and ``answer_template.json`` once the Participant Package
arrives (BLK-1, AC-9).
"""

from __future__ import annotations

# TODO(build step 1):
#
# class QueryRequest(BaseModel):
#     question: str        # the evaluator sends exactly one field (FR-1.2)
#
# class TraceEntry(BaseModel):
#     tool: str
#     args: dict[str, Any]
#     result: str
#
# class QueryResponse(BaseModel):
#     answer: str          # required, always non-empty (FR-1.4)
#     steps: int           # optional in the brief; treated as required here
#     tool_trace: list[TraceEntry]
#
# class HealthResponse(BaseModel):
#     status: str
#     domain_predict_mode: str   # echoed so `mock` is visible at a glance
