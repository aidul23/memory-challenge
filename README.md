# LEDGER: auditable agent memory under change (artifact)

Code, workloads, run logs and generated results for the paper *LEDGER:
Auditable Agent Memory Under Change*. The supplement referred to in the paper
(S1–S5) is [`docs/supplement.pdf`](docs/supplement.pdf). A longer walkthrough of
the design and code is in [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md).

Everything in the paper's tables can be regenerated from the logs in `runs/`
without an API key (step 2 below). Rerunning the LLM systems is also free with
the shipped response cache (step 3), except Mem0, whose cache is not shipped.

## Setup

Tested with Python 3.14 on macOS.

```
pip install -r requirements.txt
python -m pytest -q tests            # 107 offline tests
```

## 1. Layout

| Path | Contents |
|---|---|
| `interface/` | The `MemorySystem` API every system implements, logged data types, and the one tokenizer (`tokens.py`). |
| `systems/` | `no_memory`, `full_replay`, `transcript_rag` (BM25), `mem0_oss` (Mem0 1.0.11 adapter), `ledger` (our system: rules in `system.py`, LLM extractor and router in `llm.py`, rule-based extractor in `extract.py`), `hybrid` (LEDGER+EP). |
| `workloads/` | Timeline schema and ground truth (`timeline.py`), the generator (`generator.py`), the generated timelines used in the paper (`generated/`), and the LongMemEval loader with the sampled question ids (`longmemeval/main.json`, `pilot.json`). |
| `runner/` | Replay, scoring, shared answerer, prices, sweeps, the long-preset experiments, LongMemEval, and the token recount. |
| `runs/` | Every log behind the paper: per-probe evidence, verdicts, usage, latency and cost. |
| `analysis/make_tables.py` | Regenerates every results table and the supporting CSVs from `runs/`. |
| `results/` | Its output: `tables/*.tex`, `tables/*.csv`, and the long-preset figure data `figures/*.dat`. |
| `artifact/llm-cache.tar.xz` | The LLM response cache (4.8 MB compressed, 45k responses). |
| `docs/` | Supplement and project overview. |

### System names in code and in the paper

| Paper | Coding workload log | LongMemEval log |
|---|---|---|
| No memory / Full replay / Transcript RAG | `no_memory` / `full_replay` / `transcript_rag` | same |
| Mem0 / Mem0 (coding prompt) | `mem0` / `mem0_coding` | `mem0` |
| LEDGER (rules) | `ledger` | – |
| LEDGER (LLM) | `ledger_llm` | `ledger_llm_user` |
| LEDGER+EP fallback | `hybrid_llm_fb` | `hybrid_llm_user_fallback` |
| LEDGER+EP augment | `hybrid_llm_aug` | `hybrid_llm_user` |

Logs with the suffix `__nosrc` are runs in which deletion requests carry only a
description (no source event ids).

## 2. Regenerate the paper's numbers (no API key)

```
python analysis/make_tables.py       # results/tables/*  (prints every table)
python -m runner.recount_tokens      # token-count check, see below
```

| Paper item | Output | Logs |
|---|---|---|
| Workloads table | printed by `python -m workloads.generator --preset small --seed N --out regenerated/` (identical to the shipped timeline) | `workloads/generated/` |
| Coding workload, seeds 0–4 | `results/tables/coding_rows.tex` | `runs/synthetic-small-s{0..4}/`, Mem0 in `runs/mem0/` |
| Held-out seeds 5–9 | `results/tables/heldout_rows.tex` | `runs/synthetic-small-s{5..9}/`, Mem0 in `runs/mem0-heldout/` |
| LongMemEval (accuracy, CIs, write/read/answer cost) | `results/tables/lme_rows.tex`, paired CIs in `lme_paired.tex` | `runs/longmemeval/main/` |
| Pilot overlap (10 questions) | `results/tables/pilot_exclusion.csv` | `runs/longmemeval/main/`, ids in `workloads/longmemeval/` |
| Judge audit | – | `runs/longmemeval/audit/` |
| Long preset figure and per-checkpoint table (supplement S3) | `results/figures/*.dat`, `results/tables/growth_tokens.tex`; `python -m runner.longitudinal report` | `runs/synthetic-long-s{0..2}/`, Mem0 in `runs/mem0/`, `runs/longitudinal-long.json` |
| Restarts and model change | `python -m runner.longitudinal report` | `runs/restart-long.json`, `runs/synthetic-long-s*/hybrid_llm_aug__*.json` |
| Run-to-run noise and LLM timings | – | `runs/noise/` (repeats `__r2`, `__r3`), `runs/mem0-timing/` |
| Efficiency table | `results/tables/efficiency_rows.tex` | as above |
| Per-seed results, cost breakdown | `results/tables/per_seed.csv`, `cost_breakdown.csv` | as above |
| Ablations | see `docs/PROJECT_OVERVIEW.md` and the supplement S4 | – |

**Token counts.** All token counts in the paper are model tokens of the
answering model's tokenizer (`o200k_base`, via `tiktoken`), which
`interface/tokens.py` now uses. The logged runs were made with an earlier
word/punctuation counter (`"tokenizer": "regex-words/1"` in each log's
settings), which also fitted evidence into the budget. `runner.recount_tokens`
checks that the budget never cut any evidence in any logged run, so the
evidence is what a model-token budget would have selected, and
`make_tables.py` recounts every logged evidence set with the model tokenizer.

**Costs** use the provider-reported token counts and the prices pinned in
`runner/prices.py`.

## 3. Rerun from the response cache (free, no network)

```
tar -xJf artifact/llm-cache.tar.xz   # creates .cache/{llm,answer,lme-answer,lme-judge}
cp .env.example .env
# in .env set:  LLM_PROVIDER=openai   LLM_MODEL=gpt-4.1-mini   (leave the rest empty)
export OPENAI_API_KEY=sk-dummy       # any value: every call is a cache hit
```

The cache key includes the provider, model, sampling settings, prompt and
schema, so with these settings every LLM call of the logged runs is a hit. A
miss would call the API and fail with the dummy key; it is then recorded as an
error, not silently paid for. For example:

```
python -m runner.sweep --seeds 0 1 2 3 4 --systems ledger_llm hybrid_llm_fb hybrid_llm_aug --answer --out runs-rerun
python -m runner --timeline workloads/generated/synthetic-small-s0.json --systems ledger_llm --withhold-sources --answer --out runs-rerun
python -m runner.longitudinal restart --seeds 0 1 2 --out runs-rerun
```

LongMemEval needs the benchmark data, which is not redistributed here:

```
mkdir -p data/longmemeval
curl -L -o data/longmemeval/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
python -m runner.longmemeval --subset main --systems ledger_llm_user hybrid_llm_user_fallback hybrid_llm_user transcript_rag full_replay no_memory --out runs-rerun
```

**Not cached:** Mem0's own cache (LLM calls and embeddings, 2.1 GB) is not
shipped, so rerunning `mem0` or `mem0_coding` needs a real OpenAI key and
costs money. Their logs are in `runs/` and enter every table. Timings
(latency) are only meaningful from uncached runs; the ones in the paper come
from `runs/noise/` and `runs/mem0-timing/`.

## 4. Running new experiments

With a real key in `.env`, the same commands call the provider for any prompt
not in the cache. `python -m runner --help`, `python -m runner.sweep --help`,
`python -m runner.longitudinal --help` and `python -m runner.longmemeval --help`
list the options. New logs record `"tokenizer": "o200k_base/1"`.
