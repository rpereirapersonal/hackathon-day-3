# Market Signal Agent

An evidence-grounded agent that answers Australian financial-market questions over three
approved local datasets — RBA cash-rate decisions, ASX company prices, and the AFR news
corpus — and returns them over HTTP in a fixed JSON contract.

Built for the Cognitivo hackathon (July 2026). Two supporting documents sit alongside this
one: [`requirements.md`](requirements.md) traces every requirement to the challenge brief,
and [`architecture.md`](architecture.md) is the full design rationale. Identifiers of the
form `FR-`, `NFR-`, `DEP-`, `CON-` and `BLK-` throughout this README refer to
`requirements.md`.

---

## Build status

Stated plainly, because a README that describes unbuilt code is worse than no README.

| Step | Deliverable | Status |
| --- | --- | --- |
| 0 | Host preflight — pinned deps on `aarch64`, `onnxruntime` wheel confirmed | Not started |
| 1 | Contract and gate — `schemas.py`, `api.py`, `/health`, `/query` stub, tunnel verified off-host | Not started |
| 2 | Graph skeleton — `state.py`, `context.py`, `graph.py`, orchestrator wired to `synthesize` in mock mode | **In progress** — `state.py`, `context.py`, `config.py`, `models.py` done |
| 3 | Models and synthesis — live Qwen planning, budget cap, trace recorder, fallback ladder | Not started |
| 4 | Data layer and tools — `db.py`, `retrieval.py`, `embeddings.py`, `scripts/ingest.py`, `src/tools/` | **Blocked: BLK-2** |
| 5 | Calibration — public-question harness, per-stage latency | **Blocked: BLK-1** |

Built and verified today: the Qwen reasoning agent in [`src/orchestrator.py`](src/orchestrator.py),
its supporting state and context schemas, configuration, and the model factories. Everything
else in `src/` is a documented placeholder carrying `TODO(build step N)` markers that map to
the table above.

---

## Architecture

### The three roles

The brief requires two distinct model roles and prohibits two specific inversions: Nemotron
must not replace the Qwen brain, and Nemotron must not be the primary tool-calling model
(CON-7). This system makes both structurally impossible rather than relying on prompt
instructions.

| Role | Component | Responsibility | Explicitly does not |
| --- | --- | --- | --- |
| **Reasoning brain** | Qwen `agent-brain`, via `src/orchestrator.py` | Plan the approach, select tools, emit tool calls and arguments, review results, decide whether to continue | Write the user-facing answer (FR-2.4); get fine-tuned (CON-8) |
| **Agent runtime** | `src/middleware.py`, `src/tools/`, `src/db.py`, `src/retrieval.py` | Validate tool calls, execute them against approved local data, compute deterministic facts, record the trace | Make reasoning decisions; generate prose |
| **Domain synthesis** | Fine-tuned Nemotron, via `src/synthesis.py` | Turn the question plus accumulated verified evidence into a concise, grounded answer | Select or call tools; re-enter the reasoning loop |

**How the separation is enforced.** The Qwen instance is bound to the tool set; the Nemotron
instance is constructed with no tools at all, so it has no mechanism to emit a tool call. The
`synthesize` node is terminal — its only outgoing edge leads to `package` — so no synthesis
output can route back into the reasoning loop. The reasoning brain's messages are never
returned as the `answer` field; `package` reads the answer exclusively from the synthesis
node's output. `tests/test_role_separation.py` asserts all three.

### Request flow

```
POST /query {"question": "..."}
        │
        ▼  src/api.py — request id assigned, hard deadline started
   graph.ainvoke({"question": ...}, context=QueryContext(...))
        │
        ├── node: reason      Qwen plans → emits tool calls → reviews results → loops
        │                     middleware enforces the tool budget and records the trace
        │                     tools execute against local SQLite + vector index
        │
        ├── node: synthesize  fine-tuned Nemotron receives question + verified tool results
        │                     → writes the final answer (no tools bound)
        │
        └── node: package     assembles {answer, steps, tool_trace}
        │
        ▼
   200 {"answer": "...", "steps": 3, "tool_trace": [...]}
```

The reasoning agent is a **subgraph** used as a node rather than being the whole service. That
buys three things: the Qwen→Nemotron hand-off becomes an explicit graph edge a reader can point
at; the synthesis step is unreachable from the tool loop by construction; and the deadline and
packaging concerns live in the outer graph, out of the agent's own control flow.

### Data flow

Structured data — RBA decisions and ASX prices — is normalised into read-only SQLite, so
counts, rankings, percentage changes and date arithmetic happen in SQL. The AFR corpus is
embedded into a vector index for semantic retrieval, so sentiment and market-direction
questions find relevant articles without exact keyword overlap.

`scripts/ingest.py` is the single adapter boundary: all knowledge of raw file layout lives
there, so when the real schemas arrive that script changes and the tool layer above it does
not. Ingestion runs ahead of serving, never inside a request (NFR-1.4), and never modifies the
sources (CON-3).

**Determinism is the highest-leverage rule in the system.** Counting, summing, ranking,
percentage change, date arithmetic and longest-run calculations are all computed in SQL or
Python and returned as finished values. Neither model performs arithmetic (FR-3.4). Both of
the brief's §10 worked failures are exactly this kind of error — a retrieval tool asked for a
structured statistic, and a chronological calculation never actually performed — so the
orchestrator prompt, the tool docstrings and the test suite each independently defend against
it.

### Serving

Two endpoints, deliberately asymmetric in what they are allowed to do.

**`GET /health`** returns 200 with a small static body whenever the process is alive. It makes
no model, gateway or database call. This is load-bearing: a non-200 during the pre-evaluation
check is a hard gate that skips the team for zero points, and a health check depending on a
slow upstream would convert a recoverable degradation into total failure (NFR-4).

**`POST /query`** validates the request, assigns a request id, starts the hard deadline,
invokes the graph asynchronously, and shapes the result through `schemas.py` — the single
definition of the contract, so no response path can drift out of shape.

---

## Repository layout

```
src/
  api.py            FastAPI app. GET /health, POST /query. Owns the deadline and outermost fallback
  schemas.py        Pydantic request/response models — the single definition of the scored contract
  graph.py          Outer StateGraph wiring reason → synthesize → package
  orchestrator.py   The Qwen reasoning agent, built with create_agent
  synthesis.py      The Nemotron synthesis node: evidence assembly, prompt, deterministic fallback
  models.py         Model factories — one per role. Central gateway configuration
  state.py          Graph state: messages plus the tool_trace and steps channels
  context.py        QueryContext: request id, deadline, remaining tool budget. Injected per request
  middleware.py     Tool-budget cap, trace recorder, deadline guard
  config.py         Environment loading and fail-fast validation
  db.py             Async SQLite connections, opened read-only, scoped per operation
  retrieval.py      Semantic search over the AFR corpus
  embeddings.py     Embedding generation, sync encoder kept off the event loop
  tools/            One module per dataset plus the TOOLS registry          (blocked: BLK-2)
scripts/ingest.py   Builds SQLite tables and the vector index from source   (blocked: BLK-2)
docs/               Tool backlog and design notes
tests/              Contract, role-separation, determinism, budget, concurrency, fallback
evals/              Calibration harness over the public questions           (blocked: BLK-1)
logs/               Per-request diagnostic logs with correlation ids
training/           Fine-tuning evidence (model workstream)
Participant_Package/ Sample answer_template.json
```

---

## Running the agent

Requires Python 3.12+. The target host is a Gigabyte Atom (`aarch64`, Linux) reached over SSH;
the install has not yet been exercised there (build step 0, DEP-4).

### 1. Install

```bash
git clone <repository-url> && cd <repository>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On the `aarch64` host, do this **first and on its own** (DEP-4). `fastembed` pulls in
`onnxruntime`, which is the dependency most likely to lack a wheel for that architecture. If
it cannot be installed, AFR retrieval falls back to SQLite FTS5 keyword search — weaker
recall, no native dependency, same tool signatures, critical path unblocked.

### 2. Configure

```bash
cp .env.example .env
```

`.env` holds non-secret values only. Export credentials in the shell — never commit them
(NFR-6.1, CON-6):

```bash
export AGENT_BRAIN_API_KEY=...
export DOMAIN_FT_API_KEY=...
```

| Variable | Purpose |
| --- | --- |
| `AGENT_BRAIN_MODEL` / `_BASE_URL` / `_API_KEY` | The supplied Qwen reasoning brain, via the LiteLLM gateway |
| `DOMAIN_FT_MODEL` / `_BASE_URL` / `_API_KEY` | The team's fine-tuned Nemotron synthesis model |
| `DOMAIN_PREDICT_MODE` | `mock` for pre-adapter integration testing, `llm` for real inference |
| `REQUEST_DEADLINE_SECONDS` | Hard per-request wall-clock budget (default 50) |
| `MAX_TOOL_CALLS` | Hard tool-call cap (default 3) |
| `DATA_DIR` / `DB_PATH` / `INDEX_PATH` | Locations of ingested artifacts |
| `EMBEDDING_MODEL_NAME` / `EMBEDDING_CACHE_DIR` | Retrieval model selection and local cache |
| `LOG_DIR` | Diagnostic log destination |

`config.py` fails fast on a missing model alias or base URL rather than falling back to a
default that would silently point at the wrong model.

> **`DOMAIN_PREDICT_MODE=llm` before official evaluation.** The cluster bootstrap starts in
> `mock`. Shipping in `mock` returns plausible answers while forfeiting the fine-tuned-model
> evidence entirely — 30% of the score. The service warns loudly at startup and echoes the mode
> in `/health` so it is visible at a glance. It is also on the pre-submission checklist below.

### 3. Ingest the datasets

```bash
python -m scripts.ingest        # blocked: BLK-2
```

Run once, before serving. Never inside a request.

### 4. Serve

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Bind `0.0.0.0`, not `127.0.0.1` (DEP-5) — a loopback-bound service is invisible to the tunnel
and fails the health gate.

Run it under a session-independent supervisor so a dropped SSH connection cannot take down the
endpoint mid-evaluation (DEP-2):

```bash
tmux new -s agent -d 'uvicorn src.api:app --host 0.0.0.0 --port 8000'
```

### 5. Expose and verify

```bash
cloudflared tunnel --url http://localhost:8000     # or: ngrok http 8000
```

Then verify **from outside the host** — a local `curl` proves nothing about the gate (DEP-1):

```bash
curl https://<public-url>/health
curl -X POST https://<public-url>/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What was the longest period the RBA held rates unchanged?"}'
```

Tunnel URLs are typically ephemeral across restarts, so re-verify the URL in
`submission.json` immediately before submitting and after any restart (DEP-3).

### Local inspection

```bash
langgraph dev
```

Registers the graph via `langgraph.json` for stepping through during development. A debugging
aid only — the production path is always the FastAPI application.

---

## API contract

**`GET /health`** → `200`, always, whenever the process is alive.

**`POST /query`**

```json
{"question": "From the first RBA record to the last, how many cash-rate decisions changed the rate?"}
```

→ `200`

```json
{
  "answer": "Direct answer containing every requested component.",
  "steps": 3,
  "tool_trace": [
    {"tool": "tool_name", "args": {"param": "value"}, "result": "tool output summary"}
  ]
}
```

`answer` is the only field scored on the hidden set, and it is always present and non-empty.
`steps` and `tool_trace` are optional in the brief but treated as required here, because they
are what let organizers diagnose a failed request.

**No path returns a 5xx, an empty `answer`, or a malformed body** — each of those scores zero.
The fallback ladder:

| Condition | Behaviour |
| --- | --- |
| A tool raises or times out | Error captured into `tool_trace`; the reasoning brain sees it and may adapt or retry once |
| Tool budget exhausted | Proceed to synthesis with the evidence gathered so far |
| Insufficient evidence for the question | Synthesis states the limitation in the answer; no figure is fabricated |
| Synthesis model unavailable or slow | Deterministic template answer assembled from `tool_trace`, returned 200 with the limitation stated |
| Request deadline reached | Same degraded path — a late valid answer still earns 80%, a timeout earns nothing |
| Unexpected internal exception | Caught at the API boundary; 200 with a valid body explaining the limitation |

### Latency and concurrency

Against the brief's bands — ≤60s full points, 61–300s a 20% penalty, >300s zero:

| Stage | Target | Mechanism |
| --- | --- | --- |
| Warm-up | before healthy | ingest at build time; embedding model loaded at import |
| Planning + tools | ≤ 30s | ≤3 tool calls targeted; hard cap and recursion limit enforced in middleware |
| Synthesis | ≤ 15s | bounded output length; no re-entry into the tool loop |
| Wall clock | ≤ 50s soft, hard deadline above it | per-request timeout, then the degraded path |

The 50s soft target leaves headroom inside the 60s band for network and tunnel overhead.

The harness sends up to three questions concurrently. No mutable module-level state
participates in request handling; connections are scoped per operation rather than shared;
`tool_trace` accumulates in graph state so traces cannot bleed between requests; and the
synchronous embedding encoder runs off the event loop.

---

## Testing

```bash
pytest
```

Tests never contact a live model, so they are fast and work while the gateway credentials are
outstanding. A scripted fake chat model drives the reasoning loop deterministically, which
makes graph structure, tool sequencing and termination testable without a gateway.

**The test modules below are currently stubs** — each carries the specific assertions it owns
as `TODO(build step N)` comments, landing with the code it covers. `pytest` collects them and
passes trivially today.

| File | Covers |
| --- | --- |
| `tests/test_contract.py` | `/health` under a broken gateway; `/query` under malformed input, tool failure, budget exhaustion, deadline expiry — every case asserting 200 with a valid non-empty `answer` |
| `tests/test_role_separation.py` | The synthesis model is bound with no tools; the answer originates from the synthesis node, not the reasoning brain |
| `tests/test_determinism.py` | Calculation helpers directly against known inputs, no model in the loop — including chronological and longest-run logic |
| `tests/test_budget_and_deadline.py` | A question exceeding the tool budget terminates at the cap and still answers; a slowed pipeline degrades rather than overruns |
| `tests/test_concurrency.py` | Three simultaneous requests, correctly matched, with no `tool_trace` cross-contamination |
| `tests/test_orchestrator.py` | Routing — a question asking for an RBA count drives a structured tool call, not an AFR retrieval call |

### Calibration

```bash
python -m evals.run_calibration        # blocked: BLK-1
```

Runs the 15 public questions, passing **only** the `prompt` field, and records per-component
correctness and latency. Scoring is per component rather than per question, because the
official grader awards partial credit and a pass/fail summary would hide most of the signal.
No file maps a question id to an answer (CON-9).

---

## Training summary

Fine-tuning Nemotron is owned by the model workstream and consumed here as an interface. Its
evidence lives in [`training/`](training/): preparation notes, training scripts and
configuration, hyperparameters, checkpoint selection rationale, logs, and the quantitative and
qualitative comparison against the supplied base model on held-out examples.

*To be completed by the model workstream before submission. This section should end up
summarising the training data and its provenance, the method and configuration, the
checkpoint chosen and why, and the base-vs-fine-tuned comparison — with the detail in
`training/`.*

What this workstream guarantees on that model's behalf: it receives the original question plus
the accumulated verified tool results and nothing else — not the reasoning transcript, so
intermediate speculation cannot leak into the answer — and it is bound with no tools, so it
cannot become the primary tool-calling model.

---

## Known limitations

- **Dataset schemas are unverified.** The tool layer and ingestion script are specified by
  capability, not against real files (BLK-2). No column name, instrument identifier or date
  format is asserted anywhere in this repository. Tool signatures will shift on first contact
  with the data. Ranked candidates are in [`docs/tool-backlog.md`](docs/tool-backlog.md).
- **No calibration run yet.** The latency budget above is a design allocation, not a
  measurement. Per-stage timings on the target hardware are needed to confirm the 50s target
  is realistic (BLK-1).
- **Retrieval recall is unmeasured.** Semantic search quality over the AFR corpus is unknown
  until the corpus exists. Sentiment and market-direction questions depend on it, and the FTS5
  fallback would be materially weaker.
- **`mock` mode remains a live footgun.** The mitigations are procedural — a startup warning,
  the mode echoed in `/health`, a checklist item. A submission left in `mock` would return
  plausible answers while forfeiting the fine-tuned-model evidence.
- **Single-process deployment.** One process, one host, one tunnel. Adequate for three
  concurrent requests, but a single point of failure for a hard gate — which is why
  session-independent supervision and off-host verification are mandatory steps above.
- **Model gateway endpoints not yet available.** Live behaviour, real latency and error modes
  of both models are untested (BLK-3, BLK-4). The fallback ladder is designed but only
  partially exercisable.
- **Exposure mechanism unconfirmed.** The tunnel assumes the host is not directly reachable by
  the harness. If it is, the tunnel hop is dropped with no code change (BLK-7).

---

## Pre-submission checklist

- [ ] `DOMAIN_PREDICT_MODE=llm`, confirmed via the `/health` payload
- [ ] `GET /health` returns 200 **from off-host**
- [ ] `POST /query` returns a valid contract-shaped response **from off-host**
- [ ] `submission.json` complete, with the endpoint re-verified at the declared commit SHA
- [ ] Secret scan over the tracked tree finds no keys, tokens or machine-specific paths
- [ ] No hidden evaluation material committed; no question id mapped to an answer
- [ ] `Participant_Package/answer_template.json` present and valid
- [ ] `training/` contains reproducible fine-tuning evidence
- [ ] Setup instructions succeed from a clean checkout on the `aarch64` host
