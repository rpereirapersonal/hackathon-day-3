# Requirements — Agentic Layer

Requirements for the agent application that hosts the team's fine-tuned model for the Cognitivo
hackathon (July 2026). Scope is the **agentic layer only**: the HTTP service, the Qwen reasoning
loop, the tool runtime, and the hand-off to the fine-tuned Nemotron model for answer synthesis.

Every requirement below is traceable to the challenge brief. Section references in the form
"brief §7" point at the brief's numbered sections.

---

## 1. Purpose and scope

The system answers unseen financial-market questions over approved local datasets (RBA cash-rate
decisions, ASX company prices, the AFR news corpus) and returns them over HTTP in a fixed JSON
contract.

**In scope for this workstream**

- The HTTP service and its request/response contract.
- The Qwen (`agent-brain`) planning and tool-selection loop.
- The tool runtime: argument validation, execution against local data, trace recording.
- Data access: structured queries and news retrieval.
- The call into the fine-tuned Nemotron model for final answer synthesis.
- Latency, concurrency, error handling and fallback behaviour.
- Tests and evaluation harness for the above.

**Out of scope** (owned by the model workstream, consumed here as an interface)

- Fine-tuning Nemotron: data preparation, training, checkpoint selection, base-vs-tuned evidence.
- Serving the Nemotron adapter and its endpoint.
- The organizers' independent LLM judge.
- Any modification of the source datasets (prohibited outright — brief §9 rule 3).

## 2. System context

The brief requires **two distinct reachable endpoints**. They are frequently conflated, so they
are separated here explicitly.

| Endpoint | Contract | Owner | Contributes to |
| --- | --- | --- | --- |
| **Agent endpoint** — declared in `submission.json` (brief §8) | `GET /health`, `POST /query` | This workstream | 40% hidden-question score, 30% architecture score |
| **Fine-tuned model endpoint** (brief §8, §6A) | OpenAI-compatible chat completions | Model workstream | 30% fine-tuned model score |

`LiteLLM` is named once in the brief (§1), as the alias through which the supplied Qwen reasoning
brain is reached. It is an upstream dependency that this service *calls*; it is not where this
service is hosted. LiteLLM is an OpenAI-compatible model proxy and has no mechanism to expose a
`/query` endpoint that executes a tool loop and returns the required response shape. The agent
endpoint is therefore this workstream's responsibility, and it sits above LiteLLM, calling through
it twice — once for Qwen planning, once for Nemotron synthesis.

Note the asymmetry in the brief's wording: the fine-tuned model endpoint may be substituted by "a
documented organizer-approved method for technical model assessment". The agent endpoint has no
such alternative — `GET /health` is a hard gate (brief §7) and `POST /query` compliance is scored
directly (brief §6B).

---

## 3. Functional requirements

### FR-1 — HTTP contract

| ID | Requirement |
| --- | --- |
| FR-1.1 | `GET /health` returns HTTP 200 whenever the process is running. It MUST NOT call any language model, gateway or remote service, so that a slow or unavailable upstream cannot fail the gate. |
| FR-1.2 | `POST /query` accepts one JSON object containing a single `question` field, per the brief's input contract (§4). |
| FR-1.3 | `POST /query` returns one JSON object with `answer` (string, required), `steps` (integer, optional) and `tool_trace` (ordered array of `{tool, args, result}`, optional). `steps` and `tool_trace` are strongly recommended for organizer diagnostics (§5) and are therefore treated as required by this implementation. |
| FR-1.4 | The `answer` field is always present and non-empty. Where evidence is insufficient, the limitation is stated in `answer`; the response is never empty and a figure is never invented (§9 rule 5). |
| FR-1.5 | Every response is valid, parseable JSON, including on internal error. A malformed or missing `answer` scores zero (§6C). |
| FR-1.6 | Unexpected internal failures still produce HTTP 200 with a valid contract-conformant body explaining the limitation, rather than a 4xx/5xx with no `answer`. |

### FR-2 — Reasoning brain (Qwen / `agent-brain`)

| ID | Requirement |
| --- | --- |
| FR-2.1 | Qwen receives the question, plans the approach, selects tools, and emits tool calls with arguments (§3). |
| FR-2.2 | Qwen reviews returned tool results and decides whether a further tool call is required. |
| FR-2.3 | Qwen is NOT fine-tuned (§3, explicit prohibition). |
| FR-2.4 | Qwen does NOT write the final user-facing answer. Its output is a plan and a set of verified tool results. |
| FR-2.5 | The reasoning loop terminates on: no further tool calls requested, tool budget exhausted (NFR-2), or deadline reached (NFR-1). Every path proceeds to synthesis. |

### FR-3 — Tool runtime

| ID | Requirement |
| --- | --- |
| FR-3.1 | Tool arguments are validated against a declared schema before execution. Out-of-range or malformed arguments are rejected without touching the data layer. |
| FR-3.2 | Tools read only from the approved local datasets. No external network calls or unrestricted browsing during scoring (§9 rules 1–2). |
| FR-3.3 | Every tool invocation is appended to `tool_trace` in order, with its name, arguments and a summary of its result. |
| FR-3.4 | All dataset-derived facts — counts, sums, differences, percentage changes, rankings, date arithmetic, longest-run calculations — are computed deterministically in application code or SQL. The language models perform no arithmetic (§6B). |
| FR-3.5 | Tool results are bounded in size. No tool performs an unbounded listing or full scan of a large dataset; the brief identifies this as a guaranteed timeout (§7). |
| FR-3.6 | A tool that fails returns a structured error result into the trace rather than raising out of the graph, so the reasoning brain can adapt. |
| FR-3.7 | Tools are read-only with respect to the datasets (§9 rule 3). |

### FR-4 — Tool capability surface

Specified by **capability**, not by column name or schema. The concrete signatures depend on the
real dataset files, which are not yet available (see BLK-2).

| ID | Requirement |
| --- | --- |
| FR-4.1 | Structured querying of RBA cash-rate decisions: filter by date range, retrieve decisions in chronological order, and derive change/hold statistics deterministically. The brief's §10 worked examples both fail on this capability, making it a priority. |
| FR-4.2 | Structured querying of ASX company prices: look up by company/instrument and date or date range, and compute price movements deterministically. |
| FR-4.3 | Retrieval over the AFR news corpus: return the most relevant articles for a query, with enough context for sentiment and market-direction questions. |
| FR-4.4 | Deterministic aggregation and comparison across retrieved records: counting, ranking, chronological comparison, and financial calculation. |
| FR-4.5 | Cross-dataset questions are answerable by composing the above within the tool budget (NFR-2). |

### FR-5 — Domain answer synthesis (fine-tuned Nemotron)

| ID | Requirement |
| --- | --- |
| FR-5.1 | After the reasoning loop closes, the fine-tuned Nemotron model receives the original question plus the accumulated verified tool results, and synthesizes the final answer (§3). |
| FR-5.2 | Nemotron is bound with NO tools and cannot re-enter the reasoning loop. It is not the primary tool-calling model (§3, explicit prohibition). |
| FR-5.3 | The synthesized answer is direct and concise, and contains every component the question explicitly requested — multi-part questions earn partial credit per component (§6C). |
| FR-5.4 | The answer makes no claim unsupported by the supplied tool results (§6A: "avoidance of unsupported claims"). |
| FR-5.5 | The synthesis step honours `DOMAIN_PREDICT_MODE`: `mock` for pre-adapter integration testing, `llm` for real inference. The service warns loudly at startup when running in `mock`, because shipping in that mode would mean the submission does not use the fine-tuned model (§3). |
| FR-5.6 | Answer text is free of internal implementation detail — tool names, prompt fragments, stack traces or reasoning scaffolding. |

### FR-6 — Diagnostics

| ID | Requirement |
| --- | --- |
| FR-6.1 | `steps` reports the number of reasoning steps actually taken. |
| FR-6.2 | `tool_trace` is ordered and reflects real executions, including failed and rejected calls. |
| FR-6.3 | Per-request diagnostic logs are written to `logs/` with a correlation id and per-stage timings, sufficient for organizers to diagnose a failed request (§8). |

---

## 4. Non-functional requirements

### NFR-1 — Latency

The brief's bands (§7): ≤60s earns full points; 61–300s forfeits 20% of points earned for that
question; >300s scores zero.

| ID | Requirement |
| --- | --- |
| NFR-1.1 | Internal wall-clock target of ≤50s per request, leaving headroom inside the 60s band for network and tunnel overhead. |
| NFR-1.2 | A hard per-request deadline is enforced. On expiry the service returns a degraded but valid answer rather than continuing, because a late answer still scores 80% while a timeout scores zero. |
| NFR-1.3 | Per-stage budgets are allocated and measured: planning plus tool execution, then synthesis. |
| NFR-1.4 | Startup costs — data indexing, embedding-model loading — are paid before the service reports healthy, never inside a request. |

### NFR-2 — Tool budget

| ID | Requirement |
| --- | --- |
| NFR-2.1 | The design targets ≤3 tool calls per question (§7). |
| NFR-2.2 | A hard cap on tool calls and on graph recursion is enforced in code, not merely requested in a prompt. The brief warns that more than 5 loops will likely breach 60s (§7). |
| NFR-2.3 | Exhausting the budget proceeds to synthesis with the evidence gathered so far; it does not error. |

### NFR-3 — Concurrency

| ID | Requirement |
| --- | --- |
| NFR-3.1 | At least 3 simultaneous `POST /query` requests are handled without mixing state or responses (§7). |
| NFR-3.2 | No mutable module-level state participates in request handling. All per-request state lives in graph state or request-scoped context. |
| NFR-3.3 | Data access is safe under concurrency, with connections scoped per operation rather than shared globally. |
| NFR-3.4 | Blocking or CPU-bound work is executed off the event loop so it cannot stall concurrent requests. |

### NFR-4 — Availability

| ID | Requirement |
| --- | --- |
| NFR-4.1 | `/health` is a hard gate: a non-200 during the pre-evaluation check means the team is skipped and scores zero (§7). It must remain 200 while the process lives, independent of upstream gateway health. |
| NFR-4.2 | No single upstream failure — gateway, model server, missing index — may cause `/health` to fail. Such conditions are surfaced in logs and in degraded `/query` answers instead. |

### NFR-5 — Reproducibility and maintainability

| ID | Requirement |
| --- | --- |
| NFR-5.1 | Dependencies are pinned; the setup and run procedure is documented in `README.md` and works from a clean checkout (§6B). |
| NFR-5.2 | Module boundaries are single-responsibility, with explicit error handling and timeouts (§6B). |
| NFR-5.3 | The repository contains the folders the brief requires: `src/`, `training/`, `logs/`, `Participant_Package/` (§8). |
| NFR-5.4 | Automated tests cover the response contract, the role separation, the deterministic calculations, and the fallback paths, and run without contacting a live model. |

### NFR-6 — Security and hygiene

| ID | Requirement |
| --- | --- |
| NFR-6.1 | No credentials, API keys, tokens, machine-specific paths or secrets in any committed file or log (§9 rule 6). Secrets are supplied through the environment only. |
| NFR-6.2 | `.env.example` contains placeholder values exclusively. |
| NFR-6.3 | No hidden evaluation material is committed (§6B). |
| NFR-6.4 | Logs record questions, tool calls and timings, but never keys or credentials. |

---

## 5. Deployability

The service runs on a Gigabyte Atom accessed over SSH, and must be reachable by the organizers'
harness.

| ID | Requirement |
| --- | --- |
| DEP-1 | The agent endpoint is reachable from outside the host for the entire evaluation window. |
| DEP-2 | The service process survives an SSH session disconnect. A dropped terminal must not take down the health gate. |
| DEP-3 | The endpoint URL recorded in `submission.json` matches the live endpoint at the declared commit SHA. Where the exposure mechanism issues an ephemeral URL, this is re-verified immediately before submission. |
| DEP-4 | The pinned dependency set installs and runs on the host's `aarch64` architecture. Any dependency that cannot be satisfied there has a documented fallback that keeps the critical path working. |
| DEP-5 | The service binds all interfaces rather than loopback, so the exposure mechanism can reach it. |
| DEP-6 | Startup order is documented and the service tolerates upstream model gateways that are not yet ready (per NFR-4.2). |

---

## 6. Constraints

Non-negotiable rules, from brief §3, §4 and §9.

| ID | Constraint |
| --- | --- |
| CON-1 | Only approved local datasets (RBA, ASX, AFR) and approved services may be used during official scoring. |
| CON-2 | No unrestricted external browsing during scoring. |
| CON-3 | The source datasets must not be altered. |
| CON-4 | All official responses must be valid JSON conforming to the required contract. |
| CON-5 | A response must be returned for every question. Insufficient evidence is stated plainly; empty responses and invented figures are both prohibited. |
| CON-6 | No secrets, machine-specific paths or credentials in submitted files or logs. |
| CON-7 | Nemotron must not be trained to replace the Qwen reasoning brain, and must not be used as the primary tool-calling model. |
| CON-8 | Qwen must not be fine-tuned. |
| CON-9 | No question-ID-specific hard-coded answers (§4). Answers must be derived from the data for every question. |

---

## 7. Blocked and open items

Work that cannot be completed until external inputs arrive. Each names what it gates.

| ID | Blocked on | Gates |
| --- | --- | --- |
| BLK-1 | **Participant Package** — `public_questions.jsonl`, `questions_template.json`, `answer_template.json`, `validate.json` | Calibration harness; the sample `answer_template.json` required in `Participant_Package/` (§8); contract validation against `validate.json`; acceptance criterion AC-9. |
| BLK-2 | **RBA / ASX / AFR dataset files and their real schemas** | `scripts/ingest.py`, the whole `src/tools/` package, FR-4 signatures, the data-layer section of `architecture.md`. No column names, tickers or date formats are asserted anywhere until these are in hand. |
| BLK-3 | **`agent-brain` gateway base URL and credential** | Live Qwen planning. Until supplied, the reasoning loop is exercised against fake models in tests. |
| BLK-4 | **`DOMAIN_FT_MODEL` endpoint and credential** | Live synthesis; `DOMAIN_PREDICT_MODE=llm`. Until supplied, `mock` mode covers integration. |
| BLK-5 | **Ownership of the LiteLLM gateway** — organizer-hosted, or team-hosted on the Atom | Operational documentation and startup ordering only. No code impact: the gateway is a base URL plus credential behind environment variables in either case. |
| BLK-6 | **Semantics of `DOMAIN_PREDICT_MODE`** — whether it is read by this service or belongs to a supplied bootstrap service (brief §3 is ambiguous) | Nothing. The safe reading is to honour it in this service's configuration regardless, which satisfies both interpretations. Worth confirming with the organizers. |
| BLK-7 | **Confirmation of host network reachability** | Choice of exposure mechanism (DEP-1). If the host is already reachable by the harness, the tunnel hop is dropped with no code change. |

---

## 8. Acceptance criteria

| ID | Criterion | Verifies |
| --- | --- | --- |
| AC-1 | `GET /health` returns 200 within milliseconds, with the model gateway deliberately unreachable. | FR-1.1, NFR-4 |
| AC-2 | `POST /query` with a well-formed question returns JSON containing a non-empty `answer`, an integer `steps`, and an ordered `tool_trace`. | FR-1.2, FR-1.3, FR-6 |
| AC-3 | Malformed input, an unreachable gateway, and a deliberately failing tool each still yield HTTP 200 with a valid, non-empty `answer` stating the limitation. | FR-1.4, FR-1.5, FR-1.6, CON-5 |
| AC-4 | A test asserts the synthesis model is bound with no tools, and that the final answer originates from the synthesis step rather than the reasoning brain. | FR-2.4, FR-5.2, CON-7 |
| AC-5 | Tests cover the deterministic calculations directly, independent of any model, including chronological and longest-run logic. | FR-3.4, FR-4.4 |
| AC-6 | A question that would exceed the tool budget terminates at the cap and still returns a valid answer. | NFR-2 |
| AC-7 | Three concurrent `POST /query` requests with distinguishable questions return three correctly matched responses, with no cross-contamination of `tool_trace`. | NFR-3 |
| AC-8 | An artificially slowed pipeline hits the deadline and returns a degraded but valid answer rather than exceeding the latency band. | NFR-1.2 |
| AC-9 | The 15 public calibration questions run end to end; per-question latency and per-component correctness are recorded. *(Blocked: BLK-1, BLK-2.)* | FR-4, NFR-1 |
| AC-10 | A secret scan over the tracked tree finds no keys, tokens, credentials or machine-specific paths. | NFR-6, CON-6 |
| AC-11 | Setup and run instructions succeed from a clean checkout on the target `aarch64` host, and the endpoint answers from off-host. | NFR-5.1, DEP-1, DEP-4, DEP-5 |
| AC-12 | The service refuses to start silently in `mock` mode: startup emits a prominent warning, and the README states the pre-evaluation switch to `llm`. | FR-5.5 |
| AC-13 | No file maps a question id to an answer. | CON-9 |

---

## 9. Traceability

`architecture.md` addresses each requirement group: FR-1 and NFR-4 in *Serving layer*; FR-2 and
FR-5 in *Role separation* and *Graph design*; FR-3 and FR-4 in *Tool design principles* and
*Data layer*; NFR-1 to NFR-3 in *Latency and concurrency*; NFR-6 and CON-6 in *Configuration*;
DEP-1 to DEP-6 in *Deployment*; and the BLK items in *Known limitations*.
