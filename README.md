# Agent memory challenge: interface and planted-fact schema (v0.1)

Full walkthrough of the project: [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md).

- `interface/memory_system.py`: the `MemorySystem` API every system implements
  (baselines, third-party systems, our ledger), plus all logged data types.
- `workloads/timeline.py`: the planted-fact `Timeline` schema. Ground truth is
  derived from mutations and every single-fact probe is re-checked at load time.
- `workloads/examples/build_example.py`: a 12-event coding-agent timeline that
  exercises every mutation kind and probe type.
- `workloads/generator.py`: seeded synthetic generator (presets `small`,
  `medium`). Several services per user, nine fact kinds, varied phrasings per
  source, filler and distractor events, and every probe type except
  `multi_fact`. `hygiene_errors` guarantees no event mentions a value it does
  not carry, so a surfaced value always points to one fact.
- `systems/no_memory/`: the lower-bound baseline and smallest reference adapter.
- `systems/full_replay/`, `systems/transcript_rag/` (BM25 over events): the
  transcript baselines, sharing `systems/transcript_store.py`.
- `systems/ledger/`: our system. A bitemporal, trust-ranked fact ledger with
  provenance: restatements merge, updates supersede (old values stay queryable
  by `as_of`), less-trusted contradictions are kept as disputes and never
  retrieved, deletion erases a slot's whole history, and unknown or deleted
  slots abstain. Fact extraction sits behind an `Extractor` (`extract.py`);
  the default `RuleExtractor` is a deterministic, model-free stand-in that
  covers only a handful of slots and assumes one project per user.
  `llm.py` has `LLMExtractor` (structured outputs, responses cached in
  `.cache/llm/`); run it as the `ledger_llm` system. Its provider comes from
  `.env` (`backends.py`): Anthropic, OpenAI, or any OpenAI-compatible
  endpoint via `OPENAI_BASE_URL`.
- `interface/tokens.py`: the single tokenizer for evidence budgets.
- `runner/replay.py`: replays a timeline (ingest per session, deletes at their
  stream position, probes at checkpoints), truncates over-budget evidence and
  logs every call. `runner/sweep.py` runs systems over several seeds and
  reports mean ± sd, and with `--repeats` the run-to-run noise of LLM systems.
  `runner/scoring.py` scores the injected evidence: verdicts
  `correct / polluted / wrong / missed / false_abstention / leaked /
  no_abstention`, with each forbidden value typed as `stale / future /
  contaminated / resurrected`.

```
pip install -r requirements.txt
cp .env.example .env                  # then set one provider's key (only ledger_llm needs it)
python -m workloads.examples.build_example
python -m workloads.generator         # workloads/generated/synthetic-small-s0.json
python -m runner                      # baselines + ledger; logs in runs/<workload>/
python -m runner --withhold-sources   # deletion by description only
python -m runner --timeline workloads/generated/synthetic-small-s0.json
python -m runner --systems ledger ledger_llm   # ledger_llm calls the configured LLM
python -m runner.sweep --seeds 0 1 2 3 4 --systems ledger ledger_llm --repeats 3   # mean ± sd over seeds, LLM noise
python -m pytest -q tests
```
