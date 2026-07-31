"""FastAPI application — the agent endpoint declared in ``submission.json``.

Two routes, with very different obligations.

``GET /health`` is a hard gate: a non-200 during the organizers' pre-evaluation
check skips the team for zero points. It therefore makes no model, gateway or
database call, so a cold or broken upstream cannot convert a recoverable
degradation into total failure (FR-1.1, NFR-4, AC-1).

``POST /query`` owns the per-request deadline and the outermost fallback. No
path returns a 5xx, an empty ``answer``, or a malformed body — each of those
scores zero (FR-1.5, FR-1.6, CON-5).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Before any module reads the environment. ``uvicorn src.api:app`` imports this
# module directly, with no shell wrapper to source ``.env`` first, so without
# this the process dies at import on a missing AGENT_BRAIN_MODEL. Real exported
# variables still win — ``load_dotenv`` does not override what is already set.
load_dotenv()

from src import embeddings, frames  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.context import QueryContext  # noqa: E402
from src.graph import graph  # noqa: E402
from src.schemas import (  # noqa: E402
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TraceEntry,
)

logger = logging.getLogger(__name__)

#: Read once at startup. A missing variable should fail loudly here, at boot,
#: rather than on the first scored request.
_SETTINGS = load_settings()


def _configure_logging() -> None:
    """Per-request diagnostics to LOG_DIR, with the correlation id (FR-6.3).

    Never logs credentials: only ids, tool names, timings and answer lengths
    reach the handlers (NFR-6.4).
    """
    log_dir = Path(_SETTINGS.log_dir)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "agent.log"))
    except OSError:
        # An unwritable log directory is not a reason to fail the health gate.
        logging.getLogger(__name__).warning(
            "Log directory %s is not writable; logging to stderr only.", log_dir
        )
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the data layer before the app reports healthy.

    Both warms are idempotent and log rather than raise on a missing artifact,
    so a partial ingest degrades ``/query`` answers instead of taking down the
    health gate (NFR-1.4, NFR-4.2). Paying the parquet load and the encoder
    construction here is what keeps them out of the request path.
    """
    _configure_logging()
    if _SETTINGS.is_mock_synthesis:
        logger.warning(
            "DOMAIN_PREDICT_MODE=mock - the fine-tuned model is NOT in use. "
            "Set DOMAIN_PREDICT_MODE=llm before evaluation (AC-12)."
        )
    for name, warm in (("frames", frames.warm), ("embeddings", embeddings.warm)):
        try:
            await asyncio.to_thread(warm)
        except Exception:
            logger.exception("Warming %s failed; /query will degrade.", name)
    yield


app = FastAPI(title="Market Signal Agent", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """The hard gate. No upstream calls, ever (AC-1)."""
    return HealthResponse(
        status="ok", domain_predict_mode=_SETTINGS.domain_predict_mode
    )


def _degraded(answer: str, steps: int = 0) -> JSONResponse:
    """A contract-conformant 200 carrying a stated limitation (FR-1.6)."""
    return JSONResponse(
        status_code=200,
        content=QueryResponse(answer=answer, steps=steps, tool_trace=[]).model_dump(),
    )


@app.post("/query")
async def query(request: Request) -> JSONResponse:
    """Answer one question. Always 200, always a non-empty answer.

    The request body is read and validated by hand rather than through a
    FastAPI parameter, because FastAPI's own validation failure is a 422 — and
    a 422 is a malformed-contract response on a scored request (FR-1.5, AC-3).
    """
    request_id = str(uuid.uuid4())[:8]
    started = time.monotonic()

    try:
        payload = await request.json()
        question = QueryRequest.model_validate(payload).question
    except Exception:
        logger.warning("[%s] Malformed request body.", request_id)
        return _degraded(
            "No question was received in a form this service could read, so no "
            "answer can be given."
        )

    logger.info("[%s] question=%r", request_id, question[:200])

    deadline_s = _SETTINGS.request_deadline_seconds
    ctx = QueryContext(
        request_id=request_id,
        deadline=time.monotonic() + deadline_s,
        tool_budget=_SETTINGS.max_tool_calls,
    )

    try:
        # The outer timeout is the backstop, not the mechanism: the middleware
        # deadline should close the loop first and leave time for synthesis.
        # Sized above the internal deadline so the graceful path wins.
        result = await asyncio.wait_for(
            graph.ainvoke({"question": question}, context=ctx),
            timeout=deadline_s + 8,
        )
    except asyncio.TimeoutError:
        logger.warning("[%s] Hard timeout after %.1fs.", request_id, deadline_s)
        return _degraded(
            "This question could not be answered within the available time "
            "budget, so no verified figures can be reported."
        )
    except Exception:
        logger.exception("[%s] Unhandled failure.", request_id)
        return _degraded(
            "An internal error prevented this question from being answered."
        )

    trace = [
        TraceEntry(
            tool=e.get("tool", "unknown"),
            args=e.get("args", {}) or {},
            result=str(e.get("result", "")),
        )
        for e in (result.get("tool_trace") or [])
        if isinstance(e, dict)
    ]
    response = QueryResponse(
        answer=result.get("answer") or "",
        steps=int(result.get("steps") or 0),
        tool_trace=trace,
    )
    if not response.answer.strip():
        # Should be unreachable: both synthesis and package guard this.
        logger.error("[%s] Empty answer survived to the API layer.", request_id)
        response.answer = (
            "The available data does not support an answer to this question."
        )

    logger.info(
        "[%s] done in %.2fs steps=%d tools=%d",
        request_id,
        time.monotonic() - started,
        response.steps,
        len(response.tool_trace),
    )
    return JSONResponse(status_code=200, content=response.model_dump())
