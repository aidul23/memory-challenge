"""Run baselines on a timeline, save one JSON log per run, print a comparison.

    python -m runner                                  # all baselines, example timeline
    python -m runner --systems transcript_rag --budget 500
    python -m runner --withhold-sources               # description-only deletion
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from interface.memory_system import MemorySystem
from runner.replay import RunResult, run
from systems.full_replay.system import FullReplay
from systems.no_memory.system import NoMemory
from systems.transcript_rag.system import TranscriptRAG
from workloads.timeline import Timeline

# name -> (factory, unbounded). Full replay is the one system allowed an
# unbounded budget (see Query.token_budget).
SYSTEMS: dict[str, tuple[Callable[[], MemorySystem], bool]] = {
    "no_memory": (NoMemory, False),
    "full_replay": (FullReplay, True),
    "transcript_rag": (TranscriptRAG, False),
}

DEFAULT_TIMELINE = Path("workloads/examples/example_timeline.json")


def _cell(p) -> str:
    types = sorted({t.value for h in p.forbidden_hits for t in h.error_types})
    return p.verdict.value + (f" ({','.join(types)})" if types else "")


def print_report(runs: list[RunResult]) -> None:
    first = runs[0]
    rows = [["probe", "cp", "type", "expected"] + [r.system for r in runs]]
    for i, p in enumerate(first.probes):
        exp = p.expected.value or "<abstain>"
        rows.append([p.probe_id, p.checkpoint, p.probe_type.value, exp]
                    + [_cell(r.probes[i]) for r in runs])
    rows.append([""] * len(rows[0]))
    sums = [r.summary() for r in runs]
    for label, get in [
        ("correct", lambda s: f"{s['correct']}/{s['probes']}"),
        ("budget violations", lambda s: str(s["budget_violations"])),
        ("isolation violations", lambda s: str(s["isolation_violations"])),
        ("unsupported ops", lambda s: ",".join(s["unsupported"]) or "-"),
        ("injected tokens", lambda s: str(s["injected_tokens"])),
        ("mean retrieve ms", lambda s: f"{s['mean_retrieve_ms']:.2f}"),
        ("items at end", lambda s: str(s["final_items"])),
    ]:
        rows.append([label, "", "", ""] + [get(s) for s in sums])
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeline", type=Path, default=DEFAULT_TIMELINE)
    ap.add_argument("--systems", nargs="+", choices=sorted(SYSTEMS), default=list(SYSTEMS))
    ap.add_argument("--budget", type=int, default=2000, help="evidence token budget per query")
    ap.add_argument("--withhold-sources", action="store_true",
                    help="send deletion requests without source_event_ids")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()

    timeline = Timeline.load(args.timeline)
    out_dir = args.out / timeline.workload_id
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "__nosrc" if args.withhold_sources else ""

    runs = []
    for name in args.systems:
        factory, unbounded = SYSTEMS[name]
        result = run(
            factory(), timeline,
            token_budget=None if unbounded else args.budget,
            delete_with_sources=not args.withhold_sources,
        )
        (out_dir / f"{name}{suffix}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        runs.append(result)

    print(f"{timeline.workload_id}: {len(timeline.events)} events, {len(timeline.probes)} probes, "
          f"budget {args.budget}, deletion {'by description only' if args.withhold_sources else 'with sources'}")
    print_report(runs)
    print(f"\nlogs: {out_dir}/")


if __name__ == "__main__":
    main()
