"""Run systems over several generated seeds and report mean and spread.

    python -m runner.sweep --seeds 0 1 2 3 4                       # free baselines
    python -m runner.sweep --seeds 0 1 2 --systems ledger ledger_llm
    python -m runner.sweep --seeds 0 1 2 --systems ledger_llm --repeats 3

A single run on a single seed cannot tell a better system from a lucky one.
Each seed's workload is loaded from ``workloads/generated/`` (generated and
saved first if missing), every system runs on it, and the report gives the
per-seed score, the mean and standard deviation across seeds, verdict totals
and accuracy per probe type.

``--repeats N`` re-runs the LLM systems N times per seed. Repeat 1 reads the
response cache like ``python -m runner`` does; later repeats bypass it, so
they measure run-to-run noise: the spread across repeats, and how many probes
change verdict. Deterministic systems run once. Logs go to
``runs/<workload>/<system>.json`` (``__r<k>`` for repeat k > 1) and a summary
to ``runs/sweep-<preset>.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from interface.memory_system import MemorySystem
from runner.__main__ import PAID, SYSTEMS
from runner.replay import RunResult, run
from systems.ledger.llm import LLMExtractor
from systems.ledger.system import Ledger
from workloads.generator import PRESETS, Config, generate
from workloads.timeline import Timeline

#: Factories that skip the LLM response cache, for repeats after the first.
UNCACHED: dict[str, Callable[[], MemorySystem]] = {
    "ledger_llm": lambda: Ledger(LLMExtractor(cache_dir=None)),
}


def load_workload(preset: str, seed: int, out: Path) -> Timeline:
    tl = generate(Config(**{**PRESETS[preset].__dict__, "seed": seed}))
    path = out / f"{tl.workload_id}.json"
    if path.exists():
        return Timeline.load(path)  # the saved file is the record runs refer to
    out.mkdir(parents=True, exist_ok=True)
    tl.save(path)
    return tl


def _pct(correct: int, total: int) -> float:
    return 100 * correct / total if total else 0.0


def _mean_sd(xs: list[float]) -> str:
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{statistics.mean(xs):.1f} ± {sd:.1f}"


def _table(rows: list[list[str]]) -> None:
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip())


def report(results: dict[str, dict[int, list[RunResult]]], seeds: list[int]) -> dict:
    """Print the comparison; return it as JSON-ready data."""
    out: dict = {"seeds": seeds, "systems": {}}
    rows = [["system"] + [f"s{s}" for s in seeds] + ["mean ± sd (% correct)"]]
    for name, by_seed in results.items():
        accs = []
        cells = []
        for s in seeds:
            runs = by_seed[s]
            scores = [r.summary()["correct"] for r in runs]
            accs.append(statistics.mean(_pct(c, len(runs[0].probes)) for c in scores))
            cell = f"{scores[0]}/{len(runs[0].probes)}"
            if len(runs) > 1:
                cell = f"{min(scores)}-{max(scores)}/{len(runs[0].probes)}"
            cells.append(cell)
        rows.append([name] + cells + [_mean_sd(accs)])
        out["systems"][name] = {"accuracy_pct": accs,
                                "correct": {s: [r.summary()["correct"] for r in by_seed[s]] for s in seeds},
                                "probes": {s: len(by_seed[s][0].probes) for s in seeds}}
    print("correct per seed (a range when repeated)")
    _table(rows)

    verdicts = sorted({p.verdict.value for b in results.values() for rs in b.values() for p in rs[0].probes})
    rows = [["system"] + verdicts]
    for name, by_seed in results.items():
        c = Counter(p.verdict.value for s in seeds for p in by_seed[s][0].probes)
        rows.append([name] + [str(c.get(v, 0)) for v in verdicts])
        out["systems"][name]["verdicts"] = dict(c)
    print("\nverdicts, all seeds (first repeat)")
    _table(rows)

    types = sorted({p.probe_type.value for b in results.values() for rs in b.values() for p in rs[0].probes})
    rows = [["system"] + types]
    for name, by_seed in results.items():
        tot, ok = Counter(), Counter()
        for s in seeds:
            for p in by_seed[s][0].probes:
                tot[p.probe_type.value] += 1
                ok[p.probe_type.value] += p.verdict.value == "correct"
        rows.append([name] + [f"{ok[t]}/{tot[t]}" for t in types])
        out["systems"][name]["by_type"] = {t: [ok[t], tot[t]] for t in types}
    print("\ncorrect by probe type, all seeds (first repeat)")
    _table(rows)

    repeated = {n: b for n, b in results.items() if len(b[seeds[0]]) > 1}
    if repeated:
        rows = [["system", "repeats", "probes that changed verdict", "mean sd across repeats (correct)"]]
        for name, by_seed in repeated.items():
            flips = total = 0
            sds = []
            for s in seeds:
                runs = by_seed[s]
                for i in range(len(runs[0].probes)):
                    total += 1
                    flips += len({r.probes[i].verdict for r in runs}) > 1
                sds.append(statistics.stdev(r.summary()["correct"] for r in runs))
            rows.append([name, str(len(by_seed[seeds[0]])), f"{flips}/{total}", f"{statistics.mean(sds):.2f}"])
            out["systems"][name]["noise"] = {"flipped": flips, "probes": total, "mean_sd": statistics.mean(sds)}
        print("\nrun-to-run noise (same seed, cache bypassed after the first repeat)")
        _table(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="small")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--systems", nargs="+", choices=sorted(SYSTEMS), default=[n for n in SYSTEMS if n not in PAID])
    ap.add_argument("--repeats", type=int, default=1, help="runs per seed for LLM systems")
    ap.add_argument("--budget", type=int, default=2000, help="evidence token budget per query")
    ap.add_argument("--workloads", type=Path, default=Path("workloads/generated"))
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be at least 1")

    results: dict[str, dict[int, list[RunResult]]] = defaultdict(dict)
    for seed in args.seeds:
        tl = load_workload(args.preset, seed, args.workloads)
        out_dir = args.out / tl.workload_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in args.systems:
            factory, unbounded = SYSTEMS[name]
            n = args.repeats if name in UNCACHED else 1
            runs = []
            for k in range(n):
                make = factory if k == 0 else UNCACHED[name]
                r = run(make(), tl, token_budget=None if unbounded else args.budget)
                log = out_dir / (f"{name}.json" if k == 0 else f"{name}__r{k + 1}.json")
                log.write_text(r.model_dump_json(indent=2), encoding="utf-8")
                runs.append(r)
                print(f"{tl.workload_id} {name}{f' r{k + 1}' if n > 1 else ''}: "
                      f"{r.summary()['correct']}/{len(r.probes)}", flush=True)
            results[name][seed] = runs

    print()
    summary = report(results, args.seeds)
    summary |= {"preset": args.preset, "budget": args.budget, "repeats": args.repeats,
                "configs": {n: results[n][args.seeds[0]][0].system_config for n in results}}
    path = args.out / f"sweep-{args.preset}.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nsummary: {path}")


if __name__ == "__main__":
    main()
