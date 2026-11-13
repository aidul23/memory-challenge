"""Behaviour over time (Challenge 3, R3): restarts, model changes, and
quality, latency and store size per checkpoint as the store grows.

    python -m runner.sweep --preset long --seeds 0 1 2 --systems ... --answer   # base runs
    python -m runner.longitudinal restart --seeds 0 1 2
    python -m runner.longitudinal upgrade --seeds 0 1 2
    python -m runner.longitudinal report --seeds 0 1 2

``restart`` replays each system with a restart after every checkpoint: the
store is snapshotted to disk, a fresh instance is built and restored, and the
stream continues on it. Every probe's verdict and evidence must match the
uninterrupted base run (LLM responses come from the cache, so any difference
is the restart's). ``upgrade`` changes the hybrid's extraction model from
``gpt-4.1-mini`` to ``gpt-4.1`` halfway through the stream, keeping the
store, and compares it with runs on either model throughout. ``report``
prints accuracy (evidence and answer), error counts, store size and latency
per checkpoint, writes them to ``runs/longitudinal-<preset>.json`` and
writes one data file per system for the paper's pgfplots figures.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from interface.memory_system import MemorySystem
from runner.__main__ import SYSTEMS
from runner.answer import Answerer
from runner.replay import RunResult, run
from runner.sweep import load_workload
from systems.hybrid.system import LedgerHybrid
from systems.ledger.backends import OpenAIBackend, backend_from_env
from systems.ledger.llm import LLMExtractor

UPGRADE_MODEL = "gpt-4.1-2025-04-14"
RESTART_SYSTEMS = ["transcript_rag", "full_replay", "ledger", "ledger_llm", "hybrid_llm_aug"]


def _key(p) -> tuple:
    return (p["probe_id"], p["verdict"], tuple(e["content"] for e in p["evidence"]))


def restart(args) -> None:
    out: dict[str, Any] = {}
    for seed in args.seeds:
        tl = load_workload(args.preset, seed, args.workloads)
        for name in args.systems:
            factory, unbounded = SYSTEMS[name]
            sizes, restores = [], []

            def reload(system: MemorySystem, cp, factory=factory, sizes=sizes, restores=restores) -> MemorySystem:
                with tempfile.TemporaryDirectory() as d:
                    path = Path(d) / "store.json"
                    system.snapshot(path)
                    sizes.append(path.stat().st_size)
                    fresh = factory()
                    t0 = time.perf_counter()
                    fresh.restore(path)
                    restores.append(time.perf_counter() - t0)
                return fresh

            r = run(factory(), tl, token_budget=None if unbounded else args.budget, after_checkpoint=reload)
            base = json.loads((args.out / tl.workload_id / f"{name}.json").read_text(encoding="utf-8"))
            ours = json.loads(r.model_dump_json())
            same = sum(_key(a) == _key(b) for a, b in zip(base["probes"], ours["probes"]))
            calls = r.system_config.get("extractor", r.system_config.get("ledger", {}).get("extractor", {})).get("calls", 0)
            out[f"{tl.workload_id}/{name}"] = {
                "probes": len(ours["probes"]), "identical": same, "restarts": len(sizes),
                "snapshot_bytes": sizes, "restore_ms": [1000 * t for t in restores], "llm_calls": calls,
            }
            print(f"{tl.workload_id} {name}: {same}/{len(ours['probes'])} probes identical after "
                  f"{len(sizes)} restarts; last snapshot {sizes[-1] / 1024:.0f} KiB, "
                  f"max restore {1000 * max(restores):.1f} ms, live LLM calls {calls}", flush=True)
    path = args.out / f"restart-{args.preset}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"summary: {path}")


def _hybrid(backend=None) -> LedgerHybrid:
    return LedgerHybrid(LLMExtractor(backend=backend), mode="augment", extract_min_trust=2)


def upgrade(args) -> None:
    backend_from_env()  # loads .env (API key, base URL) before building a backend by hand
    big = OpenAIBackend(UPGRADE_MODEL)
    variants: dict[str, tuple[Callable[[], MemorySystem], Any]] = {
        "hybrid_llm_aug__gpt-4.1": (lambda: _hybrid(big), None),
        "hybrid_llm_aug__upgrade": (_hybrid, "swap"),
    }
    for seed in args.seeds:
        tl = load_workload(args.preset, seed, args.workloads)
        half = sorted(tl.checkpoints, key=lambda c: c.at)[len(tl.checkpoints) // 2 - 1].name

        def swap(system: MemorySystem, cp) -> MemorySystem:
            if cp.name == half:
                system.ledger.extractor = LLMExtractor(backend=big)
            return system

        for name, (factory, mode) in variants.items():
            r = run(factory(), tl, token_budget=args.budget, answerer=Answerer(),
                    after_checkpoint=swap if mode == "swap" else None)
            (args.out / tl.workload_id / f"{name}.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")
            s = r.summary()
            print(f"{tl.workload_id} {name}: answer {s['answer_correct']}/{s['probes']}, "
                  f"memory ${s['memory_cost_usd'] if mode != 'swap' else float('nan'):.3f}"
                  + (f" (model changed after {half})" if mode == "swap" else ""), flush=True)


def _per_checkpoint(log: dict[str, Any]) -> list[dict[str, Any]]:
    by_cp: dict[str, list] = defaultdict(list)
    for p in log["probes"]:
        by_cp[p["checkpoint"]].append(p)
    rows = []
    for cp, ps in by_cp.items():
        stats = log["checkpoint_stats"].get(cp, {})
        bad = [p for p in ps if p["forbidden_hits"]]
        rows.append({
            "checkpoint": cp, "probes": len(ps),
            "evidence_correct": sum(p["verdict"] == "correct" for p in ps) / len(ps),
            "answer_correct": (sum(p["answer_verdict"] == "correct" for p in ps) / len(ps)
                               if ps[0].get("answer_verdict") else None),
            "forbidden_in_evidence": len(bad),
            "stale_in_evidence": sum(any("stale" in h["error_types"] for h in p["forbidden_hits"]) for p in ps),
            "mean_retrieve_ms": 1000 * statistics.mean(p["latency_s"] for p in ps),
            "injected_tokens": statistics.mean(p["injected_tokens"] for p in ps),
            "store_items": stats.get("n_items"), "storage_bytes": stats.get("storage_bytes"),
        })
    return rows


def report(args) -> None:
    names = args.systems + ["hybrid_llm_aug__gpt-4.1", "hybrid_llm_aug__upgrade"]
    table: dict[str, dict[str, list]] = {}
    for name in names:
        per_seed = []
        for seed in args.seeds:
            path = args.out / f"synthetic-{args.preset}-s{seed}" / f"{name}.json"
            if path.exists():
                per_seed.append(_per_checkpoint(json.loads(path.read_text(encoding="utf-8"))))
        if not per_seed:
            continue
        cps = [r["checkpoint"] for r in per_seed[0]]
        agg = {}
        for i, cp in enumerate(cps):
            rows = [s[i] for s in per_seed if len(s) > i]
            agg[cp] = {k: statistics.mean(r[k] for r in rows) for k in rows[0]
                       if k != "checkpoint" and all(r[k] is not None for r in rows)}
        table[name] = agg
        print(f"\n{name} ({len(per_seed)} seeds)")
        print("cp    answer  evidence  forbidden  items   KiB     retrieve ms  tokens")
        for cp, a in agg.items():
            print(f"{cp:5} {a.get('answer_correct', float('nan')):6.1%}  {a['evidence_correct']:7.1%}  "
                  f"{a['forbidden_in_evidence']:9.1f}  {a['store_items']:6.0f}  {a['storage_bytes'] / 1024:6.1f}  "
                  f"{a['mean_retrieve_ms']:11.2f}  {a['injected_tokens']:6.0f}")
    path = args.out / f"longitudinal-{args.preset}.json"
    path.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"\nsummary: {path}")
    # Data files for pgfplots in the paper (no plotting dependency here).
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, agg in table.items():
        lines = ["cp answer evidence forbidden kib tokens retrievems"]
        for i, a in enumerate(agg.values(), 1):
            lines.append(f"{i} {100 * a.get('answer_correct', float('nan')):.2f} {100 * a['evidence_correct']:.2f} "
                         f"{a['forbidden_in_evidence']:.2f} {a['storage_bytes'] / 1024:.2f} "
                         f"{a['injected_tokens']:.1f} {a['mean_retrieve_ms']:.3f}")
        (fig_dir / f"long-{name}.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"pgfplots data: {fig_dir}/long-*.dat")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["restart", "upgrade", "report"])
    ap.add_argument("--preset", default="long")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--systems", nargs="+", default=["no_memory"] + RESTART_SYSTEMS)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--workloads", type=Path, default=Path("workloads/generated"))
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()
    if args.command == "restart":
        args.systems = [s for s in args.systems if s != "no_memory"]
    {"restart": restart, "upgrade": upgrade, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
