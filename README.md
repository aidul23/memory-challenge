# Agent memory challenge: interface and planted-fact schema (v0.1)

- `interface/memory_system.py`: the `MemorySystem` API every system implements
  (baselines, third-party systems, our ledger), plus all logged data types.
- `workloads/timeline.py`: the planted-fact `Timeline` schema. Ground truth is
  derived from mutations and every single-fact probe is re-checked at load time.
- `workloads/examples/build_example.py`: a 12-event coding-agent timeline that
  exercises every mutation kind and probe type.
- `systems/no_memory/`: the lower-bound baseline and smallest reference adapter.

```
pip install -r requirements.txt
python -m workloads.examples.build_example
python -m pytest -q tests
```
