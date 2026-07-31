# Training Summary — Nemotron Domain-Synthesis LoRA

## What is in this folder

| Path | Contents |
|---|---|
| `scripts/` | `01_prepare_data.py`, `02_finetune_lora.py`, `03_compare_base_vs_finetuned.py` |
| `data/` | `train.jsonl` (725), `val.jsonl` (68), `test.jsonl` (59) — run2's 852-example set |
| `checkpoints/run2/hf_adapter/` | Adapter config, tokenizer, chat template, model card |
| `checkpoints/run2/*.json` | `eval_metrics`, `train_metrics`, and the full `log_history` loss curve |
| `comparison_base_vs_finetuned*.json` | Per-prompt base vs fine-tuned outputs for both runs |

**The 321 MB `adapter_model.safetensors` is deliberately not committed** — it exceeds GitHub's
100 MB per-file hard limit and git-lfs is not available in this environment. Everything needed to
assess *what* was fine-tuned and *how well* is committed: `adapter_config.json` pins the LoRA
geometry (rank 32, alpha 64, the seven target projections), `log_history.json` carries the full
loss curve, and the comparison files carry the held-out outputs. The weights themselves are
assessed through the live endpoint registered as `model.endpoint` in `submission.json`, which the
submission guide names as an accepted assessment method.

The per-step training state under `checkpoints/run2/hf_run/` (~4.8 GB of intermediate adapters and
optimizer shards) is also excluded; the scripts, data and config above are what reproduce it.

Two runs are documented here:

- **run1** — an initial ~6.5-minute pipeline-validation run on a small (137-example) dataset,
  proving the data-prep → NeMo-container LoRA training → checkpointing → comparison pipeline
  works end to end.
- **run2** (current/recommended) — a follow-up run on a 6x larger, more diverse dataset (852
  examples spanning full RBA history, the full ASX ticker×year grid, rate-cycle detection,
  cross-dataset questions, and expanded AFR pattern/share/month coverage), trained for 300 steps
  (~29 min). This is the adapter recommended for serving.

Both are small relative to the reference baseline (`~48k/6k/6k` train/val/test) described in
the participant handout — scale up further with more event time if available.

## What Nemotron is being trained to do

Per `Participant_Package/Challenge_Brief.md` and `Setup_Instructions.md`, Nemotron is **not** the
tool-calling model. It receives the question plus the verified structured `query_data()` tool
result (already computed by the agent runtime) and must synthesize a concise final answer that:

- includes every requested component (values, dates, counts, %, direction),
- preserves exact numbers/dates from the tool result without inventing anything,
- matches the terse, single-paragraph style shown in `Challenge_Brief.md`'s "what full marks
  looks like" examples.

## 1. Data preparation

Script: `scripts/01_prepare_data.py`. Builds `(system, user, assistant)` chat examples directly
from the three approved local datasets — no external data, no hidden evaluation data. All facts
in the `assistant` target are computed deterministically from the raw files (dates, percentages,
counts) — the same values a real `query_data()` tool call would return — so the model learns
answer *composition and formatting*, not new facts.

| Run | Source | Examples | Metrics covered |
|---|---|---:|---|
| run1 | RBA (60 sampled dates) | 64 | `lookup_rate`, `count_changes`, `extremes`, `max_hold_streak` |
| run1 | ASX (18 tickers, 3 years/ticker sampled) | 72 | `annual_return`, `max_drawdown` |
| run1 | AFR (6 sampled months, 25 patterns) | 16 | `count` |
| **run2** | RBA (**all 175** effective dates, 2010–2026) | 361 | + `count_increases`/`count_decreases`, + hiking/easing **cycle detection** |
| **run2** | ASX (**all 18 tickers × all 7 years**, 2015–2021) | 320 | + `full_sample_return`, `volatility`, cross-ticker **annual-return ranking** |
| **run2** | Cross-dataset (RBA changes + ASX return, same calendar year, 4 tickers) | 56 | combined RBA+ASX synthesis |
| **run2** | AFR (14→30 sampled months, 50 patterns) | 115 | `count`, **`share`** (%), **`count_by_month`** breakdown |

run2 output: `data/train.jsonl` (725), `data/val.jsonl` (68), `data/test.jsonl` (59) — 852 total,
85/8/7 split, seed 7. (run1 used a 90/10 train/val split with no separate test file.)

Run:
```bash
python scripts/01_prepare_data.py \
  --afr_dir "../data set/AFR" --asx_dir "../data set/ASX" \
  --rba_file "../data set/RBA Rates/RBA-rates.jsonl" --out_dir data \
  --afr_n_files 30 --afr_n_patterns 50 --afr_month_n_patterns 15 \
  --cross_tickers CBA ANZ BHP Qantas
```
Wall time: 2m10s (dominated by regex-scanning ~84k AFR articles across 30 files × 65 pattern
checks).

## 2. Fine-tuning configuration

Script: `scripts/02_finetune_lora.py`, run inside `nvcr.io/nvidia/nemo:25.09` with `--gpus all`
on the single available NVIDIA GB10 node (130 GB unified memory). Follows the reference baseline
in `Setup_Instructions.md` / `01_training_guide.md`:

| Parameter | run1 | run2 | Note |
|---|---|---|---|
| Base model | `Llama-3.1-Nemotron-Nano-8B-v1` | same | supplied model, bf16 |
| Method | LoRA (PEFT) rank 32 | same | alpha 64, dropout 0.05, targets q/k/v/o/gate/up/down proj |
| Trainable params | 83.9M / 8.11B (1.03%) | same | |
| Max sequence length | 512 | 512 | matches reference (longer OOMs on single node) |
| Batch size × grad accum | 2 × 4 (eff. 8) | same | |
| Learning rate | 5e-5 | 5e-5 | reference value; 1e-4 documented to spike |
| Warmup steps | 5 | 15 | scaled with step count |
| Max steps | 50 | **300** | reference full run is 100–500 |
| Checkpoint interval | every 10 | every 50 | |

Run (run2):
```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD/..:/workspace" -v "/path/to/models:/models:ro" -w /workspace \
  nvcr.io/nvidia/nemo:25.09 \
  python training/scripts/02_finetune_lora.py \
    --model_path /models/Llama-3.1-Nemotron-Nano-8B-v1 \
    --train_file training/data/train.jsonl --val_file training/data/val.jsonl \
    --out_dir training/checkpoints/run2 \
    --max_steps 300 --checkpoint_every 50 \
    --batch_size 2 --grad_accum 4 --lr 5e-5 --lora_rank 32 --max_seq_len 512 --warmup_steps 15
```

## 3. Results

| Run | Wall time | Step | Eval loss |
|---|---|---:|---:|
| run1 | 393s (~6.5 min) | 10 / 20 / 30 / 40 / 50 | 1.756 / 0.384 / 0.240 / 0.212 / **0.204** |
| **run2** | **1738s (~29 min)** | 50 / 100 / 150 / 200 / 250 / 300 | 0.209 / 0.109 / 0.099 / 0.098 / 0.094 / **0.094** |

run2's train loss fell from ~3.9 (step 1) to a final train_loss of 0.312 (average over the run;
per-batch noise is higher with more diverse data than run1's narrow template set). Eval loss
flattens between step 200 (0.098) and step 300 (0.094) — a sign of approaching convergence on
this dataset size rather than runaway overfitting, and a reasonable point to stop. Checkpoints
for all six steps are saved under `checkpoints/run2/hf_run/checkpoint-{50,100,150,200,250,300}`;
the final adapter used for serving is `checkpoints/run2/hf_adapter`.

## 4. Base vs. fine-tuned comparison

Script: `scripts/03_compare_base_vs_finetuned.py` — runs the same held-out prompts through the
base model and then through the base model + LoRA adapter (greedy decoding, identical prompt).

| Run | Held-out set | n | Exact match (base) | Exact match (fine-tuned) |
|---|---|---:|---:|---:|
| run1 | `val.jsonl` | 10 | 0/10 | 9/10 |
| **run2** | `test.jsonl` (unseen during training, broader question mix) | 15 | **0/15** | **15/15** |

Saved to `comparison_base_vs_finetuned.json` (run1) and `comparison_base_vs_finetuned_run2.json`
(run2). Qualitative patterns, consistent across both runs:

- **Factual accuracy**: both base and fine-tuned models usually get the underlying numbers right,
  since the correct values are already present in the supplied tool-result JSON — this is
  expected and desired (grounding, not memorization).
- **Format/completeness**: the fine-tuned model reliably reproduces the target's exact
  multi-component, single-sentence style; the base model pads answers with unrequested
  explanation/hedging text and sometimes drops requested components (e.g. collapsing a
  multi-part ASX return answer to just the percentage).
- **Hallucination under distribution shift**: on a `0.10` cash-rate value, the base model twice
  (independently, in run1 and run2) misread the tool result and reported **"10.0%"** with an
  invented justification ("...as indicated by the cash rate target percentage... or 10%"). The
  fine-tuned model correctly reported `0.10%` both times — this is a repeatable base-model
  formatting failure, not a one-off.

Example (run2, full record set in `comparison_base_vs_finetuned_run2.json`):

> **Q:** In 2020, how many RBA cash-rate changes were there, and how did ANZ perform on the ASX that year?
> **Gold:** In 2020, the RBA made 3 cash-rate changes (0 increases, 3 decreases), while ANZ lost 4.13% on the ASX (from 2020-01-02 to 2020-12-31).
> **Base:** In 2020, there were 3 changes in the RBA cash-rate. ANZ performed with a return of -4.13% on the ASX that year, from January 2nd to December 31st. *(correct facts, but drops the increases/decreases breakdown and reformats dates)*
> **Fine-tuned:** In 2020, the RBA made 3 cash-rate changes (0 increases, 3 decreases), while ANZ lost 4.13% on the ASX (from 2020-01-02 to 2020-12-31). *(matches gold exactly)*

This directly targets the hidden-question rubric's component-based partial credit: dropped
components, reformatted dates, or a misread numeric value all lose points even when the
headline fact is correct.

## 5. Serving the adapter

The saved adapter at `checkpoints/run2/hf_adapter` (recommended; use `run1/hf_adapter` only for
pipeline-smoke-test purposes) is a standard PEFT/HF LoRA adapter (no weight merge needed) and can
be loaded directly by vLLM's `--enable-lora` flag, e.g.:

```bash
vllm serve /models/Llama-3.1-Nemotron-Nano-8B-v1 \
  --enable-lora --lora-modules domain-ft=training/checkpoints/run2/hf_adapter \
  --port 8001
```

Point the LiteLLM `domain-ft` route at this endpoint and set `DOMAIN_PREDICT_MODE=llm` before
evaluation, per `Setup_Instructions.md`.

## Known limitations

- Both runs remain small relative to the reference `~48k/6k/6k` split — scale up further
  (more RBA phrasing variants, more AFR months/patterns, more cross-dataset ticker/year combos)
  if more event time is available.
- No sentiment/market-direction training examples yet (the `DOMAIN_PREDICT_MODE=llm`
  sentiment-synthesis path from `Setup_Instructions.md`); add labeled sentiment examples before
  final submission if that question type is in scope.
- Comparison sets (10 and 15 prompts) are spot checks, not statistically robust benchmarks —
  useful as directional evidence, not a final accuracy claim.
- Single-node only (no 2-node distributed training was available/attempted in this environment).
