# Agent memory challenge: interface and planted-fact schema (v0.1)

- `interface/memory_system.py`: the `MemorySystem` API every system implements
  (baselines, third-party systems, our ledger), plus all logged data types.
- `workloads/timeline.py`: the planted-fact `Timeline` schema. Ground truth is
  derived from mutations and every single-fact probe is re-checked at load time.
- `workloads/examples/build_example.py`: a 12-event coding-agent timeline that
  exercises every mutation kind and probe type.
- `systems/no_memory/`: the lower-bound baseline and smallest reference adapter.
- `systems/full_replay/`, `systems/transcript_rag/` (BM25 over events): the
  transcript baselines, sharing `systems/transcript_store.py`.
- `interface/tokens.py`: the single tokenizer for evidence budgets.
- `runner/replay.py`: replays a timeline (ingest per session, deletes at their
  stream position, probes at checkpoints), truncates over-budget evidence and
  logs every call. `runner/scoring.py` scores the injected evidence: verdicts
  `correct / polluted / wrong / missed / false_abstention / leaked /
  no_abstention`, with each forbidden value typed as `stale / future /
  contaminated / resurrected`.

```
pip install -r requirements.txt
python -m workloads.examples.build_example
python -m runner                      # all baselines; logs in runs/<workload>/
python -m runner --withhold-sources   # deletion by description only
python -m pytest -q tests
```
