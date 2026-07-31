# Architecture — Agentic Layer

System architecture for the agent application built for the Cognitivo hackathon (July 2026).
Requirement identifiers (`FR-`, `NFR-`, `DEP-`, `CON-`, `BLK-`) refer to `requirements.md`.

---

## 1. Overview

The service answers financial-market questions over approved local datasets. A supplied Qwen
reasoning brain plans and requests tools; application code executes those tools against local
data and records the evidence; the team's fine-tuned Nemotron model then writes the final answer
from that evidence. This three-way split is the brief's core architectural requirement (§3) and is
enforced structurally rather than by prompt instruction.

The stack is LangChain / LangGraph — `create_agent` for the reasoning loop, `@tool` async
functions with per-request runtime context, and middleware for cross-cutting concerns — served by
an in-process FastAPI application that owns the required HTTP contract.

```
POST /query {"question": "..."}
        │
        ▼  src/api.py — request id assigned, hard deadline started
   graph.ainvoke({"question": ...}, context=QueryContext(...))
        │
        ├── node: reason      Qwen (agent-brain) plans → emits tool calls → reviews results → loops
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

---

## 2. Role separation

The brief prohibits two specific inversions: Nemotron must not replace the Qwen brain, and
Nemotron must not be the primary tool-calling model (§3, CON-7). The architecture makes both
structurally impossible rather than relying on instructions.

| Role | Component | Responsibility | Explicitly does not |
| --- | --- | --- | --- |
| **Reasoning brain** | Qwen `agent-brain`, via `src/models.py` | Plan the approach, select tools, emit tool calls and arguments, review results, decide whether to continue | Write the user-facing answer (FR-2.4); get fine-tuned (CON-8) |
| **Agent runtime** | `src/middleware.py`, `src/tools/`, `src/db.py`, `src/retrieval.py` | Validate tool calls, execute them against approved local data, compute deterministic facts, record the trace | Make reasoning decisions; generate prose |
| **Domain synthesis** | Fine-tuned Nemotron (`DOMAIN_FT_MODEL`), via `src/synthesis.py` | Turn the question plus accumulated verified evidence into a concise, grounded answer | Select or call tools; re-enter the reasoning loop |

**How the separation is enforced.** The Qwen model instance is bound to the tool set; the Nemotron
instance is constructed with no tools at all, so it has no mechanism to emit a tool call. The
`synthesize` node is terminal — the graph's only edge from it leads to `package`, so no synthesis
output can route back into the reasoning loop. The reasoning brain's messages are never returned
as the `answer` field; `package` reads the answer exclusively from the synthesis node's output.

This boundary is also why the reasoning agent is a **subgraph** rather than one flat loop (§4): it
makes the Qwen→Nemotron hand-off a visible architectural seam that a reader can point at, instead
of an implementation detail buried inside a single agent's control flow.

---

## 3. Module map

Layout follows the brief's required folder structure (§8), with a flat, single-responsibility
module set under `src/` in the style of the reference LangChain project.

| Module | Responsibility |
| --- | --- |
| `src/api.py` | FastAPI application. `GET /health`, `POST /query`. Owns the per-request deadline and the outermost fallback. |
| `src/schemas.py` | Pydantic request and response models. The single definition of the scored JSON contract. |
| `src/graph.py` | Outer `StateGraph` wiring `reason` → `synthesize` → `package`. Exports the compiled graph. |
| `src/orchestrator.py` | The Qwen reasoning agent, built with `create_agent`. |
| `src/synthesis.py` | The Nemotron synthesis node: evidence assembly, prompt, and the deterministic fallback answer. |
| `src/models.py` | Model factories — one for the reasoning brain, one for the domain model. Central place for gateway configuration and timeouts. |
| `src/state.py` | Graph state: messages plus the `tool_trace` and `steps` channels. |
| `src/context.py` | `QueryContext` dataclass: request id, deadline, remaining tool budget. Injected per request. |
| `src/middleware.py` | Tool-budget cap, trace recorder, deadline guard. Cross-cutting concerns kept out of the tools. |
| `src/config.py` | Environment loading and fail-fast validation. No defaults that would silently point at the wrong model. |
| `src/text.py` | AFR matching conventions — the single definition of how a record counts, shared by ingest and the query path so the two cannot drift. |
| `src/ingest.py` | Builds the AFR index and the structured frames from the source files. The single adapter boundary between raw data and the tool layer. |
| `src/db.py` | Async SQLite connections over the AFR index, opened read-only, scoped per operation. |
| `src/frames.py` | Pandas frames over the ingested artifacts, loaded once per process and never mutated. |
| `src/retrieval.py` | AFR index reads: exact counting, article lookup, sentiment evidence, semantic search. |
| `src/embeddings.py` | Embedding generation, with the synchronous encoder kept off the event loop. |
| `src/tools/` | `afr.py`, `asx.py`, `rba.py`, `meta.py` and the `TOOLS` registry — twelve tools. |
| `training/` | Fine-tuning evidence, the test suite, and the calibration harness. |
| `data/` | Ingested artifacts, ~1.9 GB. Gitignored and rebuilt from source. |
| `"data set"/` | The supplied sources. Read-only (CON-3). |
| `logs/`, `training/`, `Participant_Package/` | Brief-mandated deliverable folders (§8). |
| `langgraph.json` | Registers the graph for local inspection with `langgraph dev`. Not the production serving path. |

Modules mirroring proven patterns from the reference project: `db.py` (per-call async connections),
`embeddings.py` (sync encoder wrapped for async use), `context.py` (dataclass injected as runtime
context), `middleware.py` (tool-call wrapping to gate and observe execution).

---

## 4. Graph design

An outer `StateGraph` with three nodes.

| Node | Input | Output |
| --- | --- | --- |
| `reason` | question, `QueryContext` | messages including tool results; `tool_trace` and `steps` populated |
| `synthesize` | question, accumulated verified tool results | final answer text |
| `package` | answer, `tool_trace`, `steps` | contract-shaped response object |

**State channels.** `messages` carries the reasoning conversation. `tool_trace` accumulates
append-only entries as tools execute. `steps` counts reasoning iterations. `question` retains the
original text so synthesis is not dependent on reconstructing it from message history. All of it
lives in graph state, never in module-level variables (NFR-3.2).

**Why a subgraph for reasoning.** `reason` is a `create_agent` agent — the Qwen model bound to the
tool set, with `QueryContext` as its context schema and the budget and trace middleware attached.
Wrapping it as a node inside an outer graph, rather than making the whole service one ReAct loop,
buys three things: the Qwen→Nemotron hand-off becomes an explicit graph edge (§2); the synthesis
step is unreachable from the tool loop by construction; and the outer graph is where the deadline
and packaging concerns live, keeping them out of the agent's own control flow.

**Termination.** The reasoning loop ends when Qwen requests no further tools, when the tool budget
is exhausted, or when the deadline guard fires. All three paths flow into `synthesize` — none is
an error path (FR-2.5, NFR-2.3).

**The synthesis step.** `synthesize` builds its prompt from the original question plus the
accumulated verified tool results, and nothing else (FR-5.1). Specifically:

- **Evidence, not history.** The prompt carries the computed tool results, not the raw reasoning
  transcript and not raw dataset extracts. The reasoning brain's deliberation is deliberately
  excluded, so intermediate speculation cannot leak into the answer.
- **Grounded and complete.** The instruction requires every component the question explicitly asked
  for — multi-part questions earn partial credit per component (FR-5.3) — and forbids any claim not
  present in the supplied evidence (FR-5.4). Where the evidence does not support an answer, the
  limitation is stated rather than filled in (CON-5).
- **Clean surface.** Tool names, prompt fragments and internal scaffolding are kept out of the
  answer text (FR-5.6).
- **No tools, no return path.** The model instance has no tools bound (FR-5.2) and the node is
  terminal, so synthesis cannot request more data. If the evidence is thin, that is a reasoning-loop
  outcome to be surfaced, not something synthesis is permitted to paper over.
- **Mode-aware.** In `mock` mode the node returns a deterministic stand-in so the full pipeline is
  testable before the adapter is served (FR-5.5); see §9 for why this mode is a submission risk.

---

## 5. Tool design principles

The principles below survived contact with the real data unchanged. The built surface, and what
was deliberately left out, is catalogued in `tool-backlog.md`.

- **Async tools returning compact evidence.** Each tool is an `async def` decorated as a tool, with
  a docstring written for the model: what the tool is for, when to reach for it, and what each
  argument means. Return values are short, factual strings or small structures — evidence, not
  data dumps (FR-3.5).
- **Per-request context by injection.** Tools receive the request-scoped runtime (request id,
  deadline, remaining budget) as an injected parameter, not as a model-visible argument. The model
  cannot see or forge it.
- **Bounded arguments at the schema layer.** Numeric arguments such as result limits carry
  declared minimums and maximums, so an out-of-range value is rejected by validation before any
  data access happens (FR-3.1).
- **Determinism in code, never in the model.** Counting, summing, ranking, percentage change, date
  arithmetic and longest-run calculations are performed in SQL or Python and returned as computed
  values. Neither model is asked to do arithmetic (FR-3.4). The brief's §10 worked failures are
  both of this kind — a retrieval tool asked for a structured statistic, and a chronological
  calculation not actually performed — so this is the highest-leverage rule in this document.
- **No unbounded scans.** Every query is filtered and limited. The brief states that calling a
  listing operation on a large dataset will likely breach the latency band (§7).
- **Errors as results.** A failing tool returns a structured error into the trace so the reasoning
  brain can adapt, rather than propagating an exception out of the graph (FR-3.6).
- **Read-only.** Connections are opened read-only; no tool writes to the datasets (CON-3, FR-3.7).

### Capability grouping

| Group | Capability | Backing store |
| --- | --- | --- |
| Coverage | Row counts and date spans per dataset, so an out-of-range question is refused on evidence | Parquet frame |
| RBA | Cash-rate decision queries, change/hold statistics, hold-run lengths, rate in force on a date | Parquet frame |
| ASX | Per-company and basket returns, rankings, volume, drawdown, volatility, name resolution | Parquet frame |
| AFR | Exact term counts, article lookup by headline and date, sentiment evidence, semantic search | SQLite + FTS5, plus a memmapped vector file |

### Answer precision

Grading is component-based: a marker checks each required fact independently, against a published
tolerance. That makes numeric precision an architectural constraint on the tool layer rather than a
presentation detail, because a value outside tolerance scores zero for its component rather than
partial credit.

| Quantity | Tolerance |
| --- | --- |
| Dates, tickers, counts, rankings, RBA rates, AFR counts | Exact |
| Returns, drawdowns, volatility, percentage shares | ±0.02 percentage points |
| Correlations | ±0.001 |
| Quoted closing prices | ±0.0001 |
| Calculated average volume | ±1 share |

Three consequences are designed in rather than left to chance:

- **Tools return more precision than the answer needs.** Percentages carry four decimals and prices
  six, so the rounding decision belongs to the synthesis step and no tolerance is spent on
  intermediate rounding.
- **Where a convention could plausibly go two ways, the one that reproduces the published value
  wins.** A basket return is the arithmetic mean of its constituents' returns, not the return of an
  averaged index; an annual return runs from the first close *within* the year, not from the prior
  year's close; a drawdown peak is the running maximum at the trough, not the global maximum. Each
  is asserted in `training/test_determinism.py` against a published figure.
- **Equivalent date formats are accepted, so dates are emitted in ISO** and the grader's tolerance
  note covers the difference in wording.

---

## 6. Data layer

Storage is split by size rather than by dataset, which is the one design decision the real files
changed.

- **Small structured data → resident pandas frames.** RBA is 175 rows and ASX is 31,932. Both fit
  comfortably in memory as parquet-backed frames, so aggregation, ranking, percentage change and
  date arithmetic all happen in pandas — exactly what the §10 failure examples demand, at
  sub-millisecond cost. SQLite would have added a query layer for no benefit at this size.
- **The AFR corpus → SQLite with an FTS5 index.** 780 MB across 219,538 records is too large to
  hold resident, so the article text stays on disk behind a tokenised index and only the metadata
  — headline, normalised headline, date, identifiers — is loaded as a frame.
- **AFR counting is two-stage, and the second stage is the one that counts.** The graded convention
  is case-insensitive, word-bounded, once per record, over the four text fields concatenated, with
  no deduplication. FTS5 narrows word-bounded terms, because its `unicode61` tokenizer lowercases
  and splits on punctuation and therefore cannot miss a bounded match. Substring terms cannot use
  the index at all — a substring may sit inside a token, as "corporate cuts" contains "rate cut" —
  so they narrow with a SQL `LIKE` scan, which runs in C over the stored column in about a second.
  Both stages only narrow; a `\b`-anchored regex over the unescaped text decides the count.
  Narrowing first is the difference between a 61-second full scan and a sub-second query.
- **The convention was settled by experiment, not by inspection.** Four counts published in
  `public_questions.jsonl` are the only evidence that this repository's matching rule matches the
  grader's, and three plausible alternatives fail them: matching a `json.dumps` string undercounts
  by fourfold, deduplicating breaks every value, and punctuation-stripped phrase matching lands
  62 low or 19 high. `src/text.py` records all three with their numbers, and `src/ingest.py`
  re-asserts the four counts at the end of every run and exits non-zero on a mismatch.
- **One vector per article → a memmapped `.npy`.** Vectors are L2-normalised at ingest, so cosine
  similarity is a single matmul and there is no FAISS dependency and no second native library to
  satisfy on `aarch64`.
- **`src/ingest.py` is the single adapter boundary.** All knowledge of raw file layout lives in one
  module: the empty AFR file, the trailing blank lines, the 92 records with no publication date,
  the company-named ASX files whose records carry `.AX` tickers, and the BOM in the RBA JSONL that
  makes the CSV the correct source. Ingestion runs ahead of serving, never inside a request
  (NFR-1.4), and reads the sources without modifying them (CON-3).
- **Local data only.** Every tool reads from these ingested local artifacts. Nothing in the request
  path reaches the public internet — the only outbound calls are to the two configured model
  gateways (CON-1, CON-2).
- **The native dependency is confined to one capability.** `fastembed` pulls in `onnxruntime`,
  whose `aarch64` wheel availability is verified before anything else is installed (DEP-4, §10
  build order step 0). If it cannot be satisfied, the only capability lost is `afr_search`:
  exact counting, article lookup and sentiment evidence all run on FTS5 and pandas. No published
  question requires semantic search, so the critical path does not depend on that wheel.

---

## 7. Serving layer

The brief requires **two distinct reachable endpoints**, and they are easily confused, so the
boundary is restated here:

| Endpoint | Contract | Owner |
| --- | --- | --- |
| **Agent endpoint** — declared in `submission.json` (§8) | `GET /health`, `POST /query` | This service |
| **Fine-tuned model endpoint** (§8, §6A) | OpenAI-compatible chat completions | Model workstream |

`LiteLLM` appears once in the brief (§1), as the alias through which the supplied Qwen brain is
reached. It is an OpenAI-compatible model proxy that this service *calls*; it is not the host of
this service, and it has no mechanism to expose a `/query` endpoint that runs a tool loop and
returns `{answer, steps, tool_trace}`. This service sits above the gateway and calls through it
twice — Qwen for planning, Nemotron for synthesis.

### `GET /health`

Returns 200 with the body `{"status": "ok"}` whenever the process is alive — the exact shape the
organizers' submission guide documents. It makes **no** model, gateway or database call. This is
deliberate and load-bearing: the brief makes a non-200 during the pre-evaluation check a hard gate
that skips the team for zero points (§7, NFR-4). A health check that depended on a slow upstream
would convert a recoverable degradation into total failure.

`DOMAIN_PREDICT_MODE` is echoed alongside `status` so a submission still running in `mock` is
visible at a glance (§9), but `status` remains the field the harness reads.

### `POST /query`

1. Validate the request against `schemas.py`; assign a request id.
2. Start the hard deadline and build `QueryContext`.
3. Invoke the compiled graph asynchronously.
4. Shape the result into the response contract and return 200.

`schemas.py` is the single definition of the contract, so every response — success, degraded or
error — is serialised through the same validated model and cannot drift out of shape (FR-1.3,
CON-4).

The endpoint is `async` throughout and the graph is invoked with its async API, so the event loop
serves concurrent requests without blocking (NFR-3).

### Latency budget

Against the brief's bands (§7) — ≤60s full points, 61–300s a 20% penalty, >300s zero:

| Stage | Target | Mechanism |
| --- | --- | --- |
| Warm-up | before healthy | ingest at build time; embedding model loaded at import |
| Planning + tools | ≤ 30s | ≤3 tool calls targeted, hard cap and recursion limit enforced in middleware |
| Synthesis | ≤ 15s | bounded output length; no re-entry into the tool loop |
| Wall clock | ≤ 50s soft, hard deadline above it | per-request timeout, then the degraded path below |

The 50s soft target leaves headroom inside the 60s band for network and exposure-layer overhead.

### Failure and fallback ladder

"Error handling, timeouts, and safe fallbacks" is a named scoring criterion (§6B), and CON-5
forbids both empty answers and invented figures. The ladder is therefore explicit:

| Condition | Behaviour |
| --- | --- |
| A tool raises or times out | Error captured into `tool_trace`; the reasoning brain sees it and may adapt or retry once |
| Tool budget exhausted | Proceed to synthesis with the evidence gathered so far |
| Insufficient evidence for the question | Synthesis states the limitation in the answer; no figure is fabricated |
| Synthesis model unavailable or slow | Deterministic template answer assembled from `tool_trace`, returned 200 with the limitation stated |
| Request deadline reached | Same degraded path — a late valid answer still earns 80%, a timeout earns nothing |
| Unexpected internal exception | Caught at the API boundary; 200 with a valid, contract-conformant body explaining the limitation |

No path returns a 5xx, an empty `answer`, or a malformed body — each of those scores zero (§6C).

### Concurrency

The harness sends up to three questions concurrently (§7). Measures: no mutable module-level state
in the request path; one database connection per operation rather than a shared handle;
`tool_trace` accumulated in graph state so traces cannot bleed between requests; the synchronous
embedding encoder executed off the event loop; and HTTP clients to the model gateways shared but
used concurrently, which is safe for an async client.

### Local inspection

`langgraph.json` registers the graph so it can be run and stepped through with `langgraph dev`
during development. This is a debugging aid only — it does not serve the required contract, and
the production path is always the FastAPI application.

---

## 8. Deployment

The service runs on a Gigabyte Atom accessed over SSH. The harness is remote, so the port must be
exposed.

```
judges' harness ──HTTPS──> tunnel (ngrok / cloudflared) ──> uvicorn 0.0.0.0:5000 (Atom, via SSH)
                                                                 │
                                                                 ├──> gateway: agent-brain (Qwen)
                                                                 ├──> gateway: DOMAIN_FT_MODEL
                                                                 └──> local SQLite + vector index
```

Operational requirements, each mapped to a requirement id:

- **Bind all interfaces on port 5000** (DEP-5). A service bound to `127.0.0.1` is invisible to the
  tunnel and fails the health gate. Port 5000 is the organizers' documented convention and what
  their `submission_template.json` shows; the harness in fact calls whatever `agent.endpoint`
  declares, so the two must agree.
- **Declare the endpoint in the organizers' own schema** (DEP-7). `submission.json` must use their
  field names — `schema_version`, `team_id`, `team_name`, `github_url`, `commit_sha`, and the
  nested `agent{endpoint, health_path, query_path, timeout_seconds}` and `model{endpoint,
  model_name}` — because those fields are read directly by the harness. A file that is valid JSON
  but differently shaped is not a partial pass.
- **Outlive the SSH session** (DEP-2). The process runs under a session-independent supervisor —
  `tmux`, `systemd --user`, or equivalent — so a dropped connection cannot take down the endpoint
  mid-evaluation.
- **Re-verify the URL before submission** (DEP-3). Tunnel URLs are typically ephemeral across
  restarts. The URL in `submission.json` must be confirmed against the live endpoint at the
  declared commit SHA, and re-confirmed after any restart.
- **Verify off-host reachability, not just local** (DEP-1). `GET /health` and `POST /query` are both
  exercised from outside the host before submission; a local `curl` proves nothing about the gate.
- **Architecture preflight** (DEP-4). The pinned dependency set is installed on the host early, to
  confirm `aarch64` wheel availability before the data layer depends on it (§10 step 0).
- **Tolerate a cold gateway** (DEP-6). Startup order is documented, and the service starts healthy
  even when the model gateways are not yet ready — `/health` makes no upstream call.

If the host turns out to be directly reachable by the harness (BLK-7), the tunnel hop is simply
removed and its URL replaced in `submission.json`. No code changes.

---

## 9. Configuration

All configuration is environment-supplied. `.env.example` carries placeholders only; real
credentials are exported in the shell and never committed (NFR-6, CON-6).

| Variable | Purpose |
| --- | --- |
| `AGENT_BRAIN_MODEL` | Model alias for the supplied Qwen reasoning brain |
| `AGENT_BRAIN_BASE_URL` | Gateway base URL for the reasoning brain |
| `AGENT_BRAIN_API_KEY` | Credential for the reasoning brain gateway — shell only |
| `DOMAIN_FT_MODEL` | Model alias for the fine-tuned Nemotron synthesis model |
| `DOMAIN_FT_BASE_URL` | Gateway base URL for the synthesis model |
| `DOMAIN_FT_API_KEY` | Credential for the synthesis model gateway — shell only |
| `DOMAIN_PREDICT_MODE` | `mock` for pre-adapter integration testing, `llm` for real inference |
| `REQUEST_DEADLINE_SECONDS` | Hard per-request wall-clock budget |
| `MAX_TOOL_CALLS` | Hard tool-call cap. Set to 5, above the ≤3 design target, so the adaptive retry after a failed call still fits on a three-dataset question |
| `SOURCE_DATA_DIR` | The supplied datasets, read-only |
| `DATA_DIR` | Where ingested artifacts are written |
| `EMBEDDING_MODEL_NAME` / `EMBEDDING_CACHE_DIR` | Retrieval model selection and local cache |
| `LOG_DIR` | Diagnostic log destination |

**The organizers' environment uses different names for several of these**, and their setup
instructions export them before the agent starts. `config.py` therefore accepts each organizer name
as an alias for the corresponding local one, so the service runs unmodified on a correctly
provisioned host instead of failing fast on a variable that *is* set, under another name:

| Organizer variable | Local equivalent |
| --- | --- |
| `LITELLM_BASE_URL` / `LITELLM_URL` | `AGENT_BRAIN_BASE_URL`, and `DOMAIN_FT_BASE_URL` when both models sit behind the one gateway |
| `LITELLM_KEY` | `AGENT_BRAIN_API_KEY` / `DOMAIN_FT_API_KEY` |
| `BRAIN_MODEL` | `AGENT_BRAIN_MODEL` |
| `MAX_AGENT_STEPS` | `MAX_TOOL_CALLS` |

`DOMAIN_FT_MODEL` and `DOMAIN_PREDICT_MODE` are already spelled the same in both. Where a local
name and its alias are both set, the local name wins, so an explicit override is never silently
ignored.

`config.py` fails fast on a missing model alias or base URL, rather than falling back to a default
that would silently point at the wrong model. The data-layer paths are resolved separately, by
`load_data_paths()`, and do default — partly because a wrong path surfaces at once as a
missing-file error, and partly because `python -m src.ingest` runs long before any gateway exists
and must not be blocked by an unset `AGENT_BRAIN_API_KEY`.

**`DOMAIN_PREDICT_MODE` is a submission risk, not just a setting.** The brief states that the
cluster bootstrap begins in `mock` and that teams **must** switch to `llm` before official
evaluation, so that the submission actually uses the fine-tuned model (§3). Shipping in `mock`
would forfeit the fine-tuned-model evidence entirely. Mitigations: a prominent startup warning
whenever `mock` is active, the mode echoed in the health payload for at-a-glance confirmation, and
a pre-submission checklist item. Whether this variable is read by this service or by a supplied
bootstrap service is ambiguous in the brief (BLK-6); honouring it here satisfies either reading.

**LiteLLM ownership** (BLK-5) has no code impact — the gateway is a base URL plus credential
either way. If organizer-hosted, the supplied values go into the environment. If team-hosted on
the Atom, a gateway configuration file is added under `config/` and its startup ordering
documented; the service must still start healthy before the gateway is ready (DEP-6).

---

## 10. Build order

Originally sequenced so the scored hard gate was proven first and the blocked data layer landed
last. The data arrived early, so step 4 was brought forward: it was the only step whose
correctness could be *verified* rather than merely designed, since the public questions publish
reference values for it. Steps 1 to 3 remain.

| # | Step | Deliverable | Status |
| --- | --- | --- | --- |
| 0 | **Host preflight** | Pinned dependencies installed on the Atom; `aarch64` wheel availability confirmed, `onnxruntime` in particular; FTS5 compiled in | Verified on `win32`/3.13. **`aarch64` outstanding** (DEP-4). |
| 1 | **Contract and gate** | `schemas.py`, `api.py`; `/health` returning 200 with no model call; `/query` returning a valid stub response; tunnel up; both verified from off-host | Not started. Now the critical path. |
| 2 | **Graph skeleton** | `graph.py`; orchestrator wired to a `synthesize` node in `mock` mode | `state.py`, `context.py`, `config.py`, `models.py` done. |
| 3 | **Models and synthesis** | `synthesis.py`, `middleware.py` bodies; live Qwen planning, tool budget cap, trace recorder, fallback ladder; concurrency verified | Not started. Middleware is a pass-through stub. |
| 4 | **Data layer and tools** | `text.py`, `ingest.py`, `db.py`, `frames.py`, `retrieval.py`, `embeddings.py`, `src/tools/` | **Done.** Twelve tools; 53 tests passing against published reference values. |
| 5 | **Calibration** | Public-question harness run; per-stage latency measured; tool cap and prompts tuned | Not started; unblocked. Needs step 1. |

---

## 11. Testing and evaluation

Tests run without contacting a live model (NFR-5.4), so they are fast and work while BLK-3 and
BLK-4 are outstanding.

- **Framework.** `pytest` with automatic async test support.
- **Fake models.** A scripted fake chat model drives the reasoning loop deterministically, which
  makes graph structure, tool sequencing and termination testable without a gateway.
- **Fake runtime.** Tools are invoked directly with a lightweight stand-in for the request runtime,
  so context-dependent behaviour is exercised without building a full graph. Argument bounds are
  asserted separately at the schema layer, since direct invocation bypasses validation.
- **Contract tests.** `/health` under a deliberately broken gateway; `/query` under malformed
  input, tool failure, budget exhaustion and deadline expiry — every case asserting HTTP 200 with a
  valid non-empty `answer` (AC-1 to AC-3).
- **Role-separation tests.** Assert the synthesis model is bound with no tools, and that the
  returned answer originates from the synthesis node (AC-4).
- **Determinism tests.** Every tool is called directly, with no model in the loop, and its output
  compared against a value published in `public_questions.jsonl` — AFR counts, RBA statistics and
  rate-in-force, ASX returns and rankings, volume, drawdowns with their peak and trough dates, and
  coverage spans (AC-5, AC-14). The direction matters: expectations come from the published set,
  not from this implementation's own output, because a test written against its own output proves
  consistency rather than correctness. Expectations are keyed by tool arguments and no question id
  appears in the file (CON-9, AC-13).
- **Routing tests.** Each tool's description is asserted to disclaim what it cannot do and to name
  the tool that can — `afr_search` disclaiming counting and naming `afr_count_matches`, in
  particular. That description *is* the §10 example 1 defence, and a docstring edit could otherwise
  quietly remove it.
- **Concurrency test.** Three simultaneous requests with distinguishable questions, asserting
  correctly matched responses and no `tool_trace` cross-contamination (AC-7).
- **Calibration harness.** `training/run_calibration.py` runs the public questions, passing only
  the `prompt` field, and records per-component correctness and latency. No question id is ever
  mapped to an answer (CON-9).

---

## 12. Observability

Each request is assigned a correlation id, logged with per-stage timings — planning, each tool
call, synthesis, total — to `logs/` (FR-6.3). `tool_trace` doubles as the primary diagnostic
artifact: it is returned in the response for organizer diagnostics (§5) and retained in logs for
failure analysis (§8). Logs record questions, tool names, arguments and timings, and never
credentials (NFR-6.4).

---

## 13. Known limitations

Stated plainly, as the brief requires (§6B).

- **The serving layer is not built.** Nothing is reachable over HTTP yet. Every tool is callable
  and verified directly; none has been exercised through a request, so the deadline, the budget cap
  and the trace recorder are all still designs.
- **No end-to-end calibration run yet.** Tool latency is measured — the four AFR reference counts
  return in 0.16s, 0.05s, 0.18s and 1.1s, structured calls are sub-millisecond — but per-stage
  request timings on the target hardware are not, so the 50s target in §7 remains an allocation.
- **Retrieval recall is unmeasured.** Exact counting is verified against published values; semantic
  search has no reference answer to check against, because no public question needs it. That also
  bounds the exposure: `afr_search` is insurance for hidden topical questions, and every graded
  AFR capability works without it.
- **One vector per article, encoded from its opening 320 characters.** A question turning on an
  argument made only late in a long column may not retrieve it. Chunking the full text would fix
  that at roughly 1.5–2M chunks and several hours of encoding, which is hard to justify for a
  capability no published question requires.
- **The published RBA span and the supplied file disagree.** `RBA-rates.csv` holds 175 decisions
  running to June 2026, while the published reference answer for the coverage question describes
  the RBA data as ending in November 2023. The row count matches, so this appears to be the same
  file described against an earlier snapshot. `dataset_coverage` reports what is actually present,
  and the load-bearing claim — AFR and ASX both stopping in 2021 — reproduces exactly.
- **92 AFR records carry no usable publication date.** They are retained and counted in unscoped
  totals but cannot appear in a date-bounded or grouped result. Every published reference count is
  date-scoped, so none is affected.
- **`mock` mode remains a live footgun.** The mitigations in §9 are procedural. A submission left
  in `mock` would return plausible answers while forfeiting the fine-tuned-model evidence.
- **Single-process deployment.** One process on one host, fronted by one tunnel. Adequate for three
  concurrent requests, but it is a single point of failure for a hard gate, which is why §8
  requires session-independent supervision and off-host verification.
- **Model gateway endpoints not yet available.** Live behaviour, real latency and error modes of
  both models are untested (BLK-3, BLK-4). The fallback ladder is designed but only partially
  exercisable.
- **Exposure mechanism unconfirmed.** The tunnel approach assumes the host is not directly
  reachable by the harness. If it is, §8's topology simplifies (BLK-7).
