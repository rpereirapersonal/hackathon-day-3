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
| 0 | Host preflight — pinned deps on `aarch64`, `onnxruntime` wheel confirmed | Verified on `win32`/Python 3.13; **`aarch64` still outstanding** (DEP-4) |
| 1 | Contract and gate — `schemas.py`, `api.py`, `/health`, `/query` stub, tunnel verified off-host | Not started |
| 2 | Graph skeleton — `state.py`, `context.py`, `graph.py`, orchestrator wired to `synthesize` in mock mode | **In progress** — `state.py`, `context.py`, `config.py`, `models.py` done |
| 3 | Models and synthesis — live Qwen planning, budget cap, trace recorder, fallback ladder | Not started |
| 4 | Data layer and tools — `text.py`, `ingest.py`, `db.py`, `frames.py`, `retrieval.py`, `embeddings.py`, `src/tools/` | **Done** — 12 tools, 53 tests passing against published reference values |
| 5 | Calibration — public-question harness, per-stage latency | Not started — unblocked (BLK-1 closed) |

Built and verified: the Qwen reasoning agent in [`src/orchestrator.py`](src/orchestrator.py), and
the whole data layer — ingest, the AFR index, the structured frames, and all twelve tools. The
remaining placeholders are the serving layer (`api.py`, `schemas.py`, `graph.py`), the synthesis
node, and the middleware bodies, each carrying `TODO(build step N)` markers that map to the table
above.

### What the data layer is verified against

Every number below is published in `Participant_Package/public_questions.jsonl` and reproduced
exactly by `python -m src.ingest` and `pytest`. They are the evidence that the matching and
calculation conventions match the grader's, rather than merely being self-consistent.

| Capability | Verified against |
| --- | --- |
| AFR term counting | 1,452 records / 2020 and 218 / May 2020 for `unemployment`; 369 / 2021 for `QBE`; 3,181 / 2019 for the five-term rate pattern |
| RBA decision statistics | 175 records, 41 changes, 20 increases, 21 decreases over the full record; 8 cuts across 2011–2013 totalling −2.25 pp, 4.75% → 2.50% |
| RBA rate in force | 0.10% on 23 Feb 2021, 25 Nov 2021 and 28 Nov 2020 — each set by a decision in the preceding weeks |
| ASX basket returns | +2.88%, +0.24%, −2.17% over the three 2019 post-decision weeks; +2.37% for 30 Nov – 7 Dec 2020; +20.11% for 2019 |
| ASX rankings | BHP.AX +22.17% best and AMP.AX −50.04% worst in 2018; QBE.AX +35.57% best in 2021 |
| ASX statistics | AMP.AX average daily volume 11,635,671.71; the three worst drawdowns with all six peak and trough dates |
| Dataset dimensions | 18 tickers × 1,774 rows, 2 Jan 2015 – 30 Dec 2021; 219,538 AFR records; 175 RBA decisions |

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
node's output. `training/test_role_separation.py` asserts all three.

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

Storage is split by size, not by dataset. RBA decisions (175 rows) and ASX prices (31,932)
are small enough to hold resident as pandas frames, so counts, rankings, percentage changes
and date arithmetic all happen in pandas. The AFR corpus is 780 MB across 219,538 records, so
its text stays in SQLite behind an FTS5 index and only its metadata — headline, date,
identifiers — is held in a frame.

`src/ingest.py` is the single adapter boundary: every fact about raw file layout lives there,
so the tools above it never parse a source file. It runs ahead of serving, never inside a
request (NFR-1.4), and never modifies the sources (CON-3). Its stages are separable because
they cost wildly different amounts — the structured frames build in a second, the AFR index in
about fifty, and encoding 219,538 articles takes roughly an hour and is resumable.

**AFR counting is a two-stage query, and the reason is worth knowing before editing it.**
Counting is graded against an exact convention: case-insensitive, word-bounded, once per
record, over the four text fields concatenated, with no deduplication. Two things narrow the
candidate set — the FTS5 index for word-bounded terms, and a SQL `LIKE` scan for substring
terms, which cannot use an index because a substring can sit inside a token ("corporate cuts"
contains "rate cut"). Neither decides the count: a `\b`-anchored regex confirms every
candidate. Narrowing first is what turns a 61-second full scan into a sub-second query.

Three findings behind that design, each of which silently changes the answer if undone, are
documented in [`src/text.py`](src/text.py): matching a serialised record instead of the field
text undercounts `QBE` by fourfold, because `json.dumps` renders newlines as literal `\n` and
breaks the leading word boundary; the corpus's 37,048 repeated headline-and-date pairs are
counted, so deduplicating breaks every reference value; and punctuation-stripped matching is
exact for single tokens but off by −62 or +19 for phrases depending on how they are anchored.

**Determinism is the highest-leverage rule in the system.** Counting, summing, ranking,
percentage change, date arithmetic and longest-run calculations are all computed in pandas or
SQL and returned as finished values. Neither model performs arithmetic (FR-3.4). Both of the
brief's §10 worked failures are exactly this kind of error — a retrieval tool asked for a
structured statistic, and a chronological calculation never actually performed — so the
orchestrator prompt, the tool docstrings and the test suite each independently defend against
it.

`python -m src.ingest` ends by asserting the four published AFR reference counts against the
index it just built and exits non-zero on any mismatch. A wrong convention is the most
expensive failure available here — every downstream number inherits it, with nothing to signal
the error — so it cannot be a silent outcome.

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
  config.py         Environment loading, fail-fast validation, and the data-layer paths
  ingest.py         Builds the AFR index and the structured frames. The single adapter boundary
  text.py           AFR matching conventions — the one definition of how a record counts
  db.py             Async read-only SQLite over the AFR index, scoped per operation
  frames.py         Pandas frames over the ingested artifacts, loaded once per process
  retrieval.py      AFR index reads: exact counting, article lookup, semantic search
  embeddings.py     Embedding generation, sync encoder kept off the event loop
  tools/            afr.py, asx.py, rba.py, meta.py, plus the TOOLS registry
data/               Ingested artifacts, ~1.9 GB. Gitignored, rebuilt by src/ingest.py
"data set"/         The supplied RBA, ASX and AFR sources. Never written to (CON-3)
logs/               Per-request diagnostic logs with correlation ids
training/           Fine-tuning evidence, the test suite, and the calibration harness
Participant_Package/ Challenge brief, public questions, and the answer template
tool-backlog.md     What was built, and what was deliberately left out
```

`training/` holds the tests and `run_calibration.py` alongside the fine-tuning evidence, so the
tree carries only the folders the brief names (§8). It is deliberately not a package — pytest
collects it by path via `testpaths` in `pyproject.toml`.

---

## Running the agent

Requires Python 3.12 or 3.13. The target host is a Gigabyte Atom (`aarch64`, Linux) reached over
SSH; the install has been exercised on `win32`/3.13 but not yet there (build step 0, DEP-4).

**Do not build the venv on Python 3.14.** `pandas` and `fastembed` wheels lag the newest
interpreter, and a source build of either on the Atom is not a fight worth having.

### 1. Install

Linux and macOS:

```bash
git clone <repository-url> && cd <repository>
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install "fastembed>=0.7,<1.0"     # DEP-4 probe: do this first, on its own
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

Installing `fastembed` **first and on its own** on the `aarch64` host is deliberate (DEP-4). It
pulls in `onnxruntime`, the one dependency in the set likely to lack a wheel for that
architecture, and finding that out in isolation is much clearer than watching a combined install
fail. If it cannot be installed, `afr_search` is the only capability lost: the ten other tools —
including all exact AFR counting — use SQLite FTS5 and pandas, neither of which has a native
dependency. No public question requires semantic search, so the critical path stays intact.

`sqlite3` ships with Python, but **FTS5 is a compile-time option rather than a guarantee**.
`src/ingest.py` probes for it before doing any work and fails with an actionable message if it is
missing. Run the probe on the host during step 0, not on ingest day:

```bash
python -c "import sqlite3; sqlite3.connect(':memory:').execute('create virtual table t using fts5(b)'); print('FTS5 OK')"
```

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
| `MAX_TOOL_CALLS` | Hard tool-call cap (default 5) |
| `SOURCE_DATA_DIR` | The supplied datasets. Read-only (default `./data set`) |
| `DATA_DIR` | Where ingested artifacts are written (default `./data`) |
| `EMBEDDING_MODEL_NAME` / `EMBEDDING_CACHE_DIR` | Retrieval model selection and local cache |
| `LOG_DIR` | Diagnostic log destination |

`config.py` fails fast on a missing model alias or base URL rather than falling back to a
default that would silently point at the wrong model. The data-layer paths default instead,
because a wrong path surfaces immediately as a missing-file error while a wrong model alias
would quietly produce a scored answer from the wrong model.

**On the tool cap being 5, not 3.** The prompt still targets ≤3 calls and most questions need
one or two. But the hardest cross-dataset questions genuinely need three *different* datasets —
an article, the cash-rate target in force, and a basket return — and a cap of 3 leaves no room
for the single adaptive retry the fallback ladder promises after a failed call. Latency is not
the constraint: all twelve tools are indexed reads, not model calls. The brief's warning is
about exceeding five.

> **`DOMAIN_PREDICT_MODE=llm` before official evaluation.** The cluster bootstrap starts in
> `mock`. Shipping in `mock` returns plausible answers while forfeiting the fine-tuned-model
> evidence entirely — 30% of the score. The service warns loudly at startup and echoes the mode
> in `/health` so it is visible at a glance. It is also on the pre-submission checklist below.

### 3. Ingest the datasets

```bash
python -m src.ingest
```

Run once, before serving. Never inside a request. The stages are separable, which matters
because they cost very different amounts:

```bash
python -m src.ingest --stage structured   # RBA + ASX + coverage frames — about a second
python -m src.ingest --stage afr          # AFR body, FTS5 index, metadata — about 50 seconds
python -m src.ingest --stage embeddings   # 219,538 article vectors — about an hour, resumable
python -m src.ingest --verify             # re-check the reference counts against an existing index
```

Run the AFR stage before `--stage structured`, since the coverage frame reads the AFR metadata.
The embedding pass writes into a memmapped `.npy` and records progress in a sidecar file, so an
interrupted run resumes where it stopped rather than starting over.

Artifacts land in `DATA_DIR` and total about 1.9 GB:

| Artifact | Contents |
| --- | --- |
| `afr.sqlite` | Article text plus the FTS5 index over it, `id` aligned to the metadata frame |
| `afr_meta.parquet` | 219,538 rows of headline, normalised headline, date, year, month |
| `afr_vectors.npy` | `float32[219538, 384]`, L2-normalised, row `i` describing `id == i + 1` |
| `asx.parquet` | 31,932 daily price rows across 18 tickers, with company names |
| `rba.parquet` | 175 decisions with signed changes and a derived direction |
| `coverage.parquet` | Row counts and date spans, measured rather than asserted |

Every run ends with the reference-count check and exits non-zero if it fails.

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
| Warm-up | before healthy | ingest at build time; frames and encoder warmed at startup |
| Planning + tools | ≤ 30s | ≤3 tool calls targeted; hard cap and recursion limit enforced in middleware |
| Synthesis | ≤ 15s | bounded output length; no re-entry into the tool loop |
| Wall clock | ≤ 50s soft, hard deadline above it | per-request timeout, then the degraded path |

The 50s soft target leaves headroom inside the 60s band for network and tunnel overhead.

Measured tool latency, on the development host: the four AFR reference counts return in 0.16s,
0.05s, 0.18s and 1.1s. The last is the five-term rate pattern, where four substring terms force
a `LIKE` scan; a Python pass over the same corpus takes 61s, which is the whole planning budget
for one call. Structured RBA and ASX calls are pandas operations over resident frames and are
sub-millisecond. Semantic search is a 219,538 × 384 matmul, tens of milliseconds, plus one query
encode. The remaining unknown is model latency at both ends (BLK-3, BLK-4).

The harness sends up to three questions concurrently. No mutable module-level state
participates in request handling; connections are scoped per operation rather than shared;
`tool_trace` accumulates in graph state so traces cannot bleed between requests; and the
synchronous embedding encoder runs off the event loop.

---

## Testing

```bash
pytest
```

Tests live in `training/` and never contact a live model, so they are fast and work while the
gateway credentials are outstanding.

53 tests pass. The determinism and routing modules are implemented; the rest still carry their
assertions as `TODO(build step N)` comments and land with the code they cover.

| File | Covers | Status |
| --- | --- | --- |
| `training/test_determinism.py` | Every tool called directly against a published reference value — AFR counts, RBA statistics and rate-in-force, ASX returns, rankings, volume, drawdowns, coverage. No model in the loop | **42 tests** |
| `training/test_orchestrator.py` | The registry is populated, names are unique, and each tool's description disclaims what it cannot do and names the tool that can | **11 tests** |
| `training/test_contract.py` | `/health` under a broken gateway; `/query` under malformed input, tool failure, budget exhaustion, deadline expiry — every case asserting 200 with a valid non-empty `answer` | stub (step 1) |
| `training/test_role_separation.py` | The synthesis model is bound with no tools; the answer originates from the synthesis node, not the reasoning brain | stub (step 3) |
| `training/test_budget_and_deadline.py` | A question exceeding the tool budget terminates at the cap and still answers; a slowed pipeline degrades rather than overruns | stub (step 3) |
| `training/test_concurrency.py` | Three simultaneous requests, correctly matched, with no `tool_trace` cross-contamination | stub (step 1) |

Every expectation in `test_determinism.py` is a value published in `public_questions.jsonl`, not
a figure this implementation produced and then enshrined. That direction is the point: a test
written against its own output proves consistency, not correctness. Expectations are keyed by
tool arguments, and no question id appears in the file (CON-9, AC-13). The module skips itself
when the ingested artifacts are absent, so a clean checkout still collects and passes.

### Calibration

```bash
python training/run_calibration.py
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

- **The serving layer is not built.** `api.py`, `schemas.py`, `graph.py`, `synthesis.py` and the
  middleware bodies are still stubs, so nothing is reachable over HTTP yet. Every tool is
  callable and tested directly; none of them has been exercised through a request.
- **No end-to-end calibration run yet.** Tool latency is measured, but per-stage request timings
  on the target hardware are not, so the 50s target remains an allocation rather than a
  measurement. `run_calibration.py` is unblocked and unimplemented.
- **Retrieval recall is unmeasured.** Exact counting is verified against published values, but
  semantic search quality has no reference answer to check against, since no public question
  needs it. That also bounds the risk: `afr_search` is insurance for hidden topical questions,
  and every graded AFR capability works without it.
- **One vector per article, from its opening 320 characters.** A question turning on an argument
  made only in the closing paragraphs of a long column may not retrieve it. Chunking the full
  text would fix that at roughly 1.5–2M chunks and multiple hours of encoding, which is hard to
  justify for a capability no published question requires.
- **The published RBA span and the supplied file disagree.** The supplied `RBA-rates.csv` holds
  175 decisions running to June 2026, while the public reference answer for the coverage
  question describes the RBA data as ending in November 2023. The row count matches, so this is
  the same file described against a different snapshot. `dataset_coverage` reports what is
  actually present, which is the honest answer, and the load-bearing claim — that AFR and ASX
  both stop in 2021 — is reproduced exactly. Worth raising with the organizers.
- **92 AFR records carry no publication date.** They are kept and count toward unscoped totals,
  but no date-bounded or grouped query can include them. All four published reference counts are
  date-scoped, so none is affected; an unscoped count would include them.
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
- [ ] `python -m src.ingest --verify` reproduces all four published AFR reference counts on the host
- [ ] `pytest` passes on the host with the ingested artifacts present
