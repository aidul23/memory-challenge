"""Generate the paper's result tables from the run logs (no hand-copied numbers).

    python paper/make_tables.py        # writes paper/tables/*.tex

Reads runs/synthetic-small-s{0..4}/*.json (coding workload, both deletion
modes), runs/longmemeval/main/*.json and, when present, the long-preset logs.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from pathlib import Path

RUNS = Path("runs")
OUT = Path("paper/tables")
SEEDS = range(5)

#: log name -> label in the paper
CODING = {
    "no_memory": "No memory",
    "full_replay": "Full replay",
    "transcript_rag": "Transcript RAG",
    "ledger": "\\sys{} (rules)",
    "ledger_llm": "\\sys{} (LLM)",
    "hybrid_llm_fb": "\\hyb{} fallback",
    "hybrid_llm_aug": "\\hyb{} augment",
}
LME = {
    "no_memory": "No memory",
    "full_replay": "Full replay",
    "transcript_rag": "Transcript RAG",
    "ledger_llm_user": "\\sys{} (LLM)",
    "hybrid_llm_user_fallback": "\\hyb{} fallback",
    "hybrid_llm_user": "\\hyb{} augment",
}
LME_CATS = ["single-session-user", "single-session-assistant", "single-session-preference",
            "multi-session", "knowledge-update", "temporal-reasoning", "abstention"]
LME_SHORT = ["SS-U", "SS-A", "Pref", "Multi", "KU", "Temp", "Abs"]


def load(name: str, seed: int, suffix: str = "") -> dict:
    return json.loads((RUNS / f"synthetic-small-s{seed}" / f"{name}{suffix}.json").read_text(encoding="utf-8"))


def mean_sd(xs: list[float]) -> str:
    return f"{statistics.mean(xs):.1f} $\\pm$ {statistics.stdev(xs):.1f}"


def forbidden(probes: list[dict], field: str) -> Counter:
    c: Counter = Counter()
    for p in probes:
        for t in {t for h in p[field] for t in h["error_types"]}:
            c[t] += 1
    return c


def coding_table() -> str:
    rows = []
    for name, label in CODING.items():
        logs = [load(name, s) for s in SEEDS]
        nosrc = [load(name, s, "__nosrc") for s in SEEDS] if (RUNS / "synthetic-small-s0" / f"{name}__nosrc.json").exists() else None
        probes = [p for lg in logs for p in lg["probes"]]
        ev = [100 * sum(p["verdict"] == "correct" for p in lg["probes"]) / len(lg["probes"]) for lg in logs]
        ans = [100 * sum(p["answer_verdict"] == "correct" for p in lg["probes"]) / len(lg["probes"]) for lg in logs]
        fe, fa = forbidden(probes, "forbidden_hits"), forbidden(probes, "answer_forbidden_hits")
        wrong_ans = sum(bool(p["answer_forbidden_hits"]) for p in probes)
        tokens = statistics.mean(p["injected_tokens"] for p in probes)
        cost = 0.0
        for lg in logs:
            s = _summary(lg)
            cost += s["memory"] + s["answer"]
        leak = "--"
        if nosrc:
            np_ = [p for lg in nosrc for p in lg["probes"]]
            leak = (f"{sum(any('resurrected' in h['error_types'] for h in p['forbidden_hits']) for p in np_)} / "
                    f"{sum(any('resurrected' in h['error_types'] for h in p.get('answer_forbidden_hits', ())) for p in np_)}")
        rows.append(f"{label} & {mean_sd(ans)} & {mean_sd(ev)} & {wrong_ans} & "
                    f"{fe['stale']}/{fe['future']}/{fe['contaminated']} & {leak} & {tokens:.0f} & {cost:.2f} \\\\")
    return "\n".join(rows)


def _summary(log: dict) -> dict:
    from runner.prices import cost_usd
    from interface.memory_system import Usage
    mem = [Usage(**o["usage"]) for o in log["ops"]] + [Usage(**p["usage"]) for p in log["probes"]]
    ans = [Usage(**p["answer_usage"]) for p in log["probes"] if p.get("answer_usage")]
    from runner.replay import sum_usage
    return {"memory": cost_usd(sum_usage(mem)), "answer": cost_usd(sum_usage(ans)) if ans else 0.0}


def lme_table() -> str:
    rows = []
    rng = random.Random(0)
    for name, label in LME.items():
        path = RUNS / "longmemeval" / "main" / f"{name}.json"
        if not path.exists():
            continue
        inst = [r for r in json.loads(path.read_text(encoding="utf-8"))["instances"] if "error" not in r]
        acc = 100 * sum(r["correct"] for r in inst) / len(inst)
        boots = sorted(100 * sum(rng.choice(inst)["correct"] for _ in inst) / len(inst) for _ in range(2000))
        by = {c: [r for r in inst if r["category"] == c] for c in LME_CATS}
        cats = " & ".join(f"{100 * sum(r['correct'] for r in by[c]) / len(by[c]):.0f}" for c in LME_CATS)
        tokens = statistics.mean(r["injected_tokens"] for r in inst)
        mem = statistics.mean(r["cost_usd"]["memory"] for r in inst)
        ans = statistics.mean(r["cost_usd"]["answer"] for r in inst)
        rows.append(f"{label} & {acc:.1f} ({boots[50]:.0f}--{boots[1949]:.0f}) & {cats} & "
                    f"{tokens:,.0f} & {1000 * mem:.1f} & {1000 * ans:.1f} \\\\".replace(",", "{,}"))
    return "\n".join(rows)


def heldout_table() -> str:
    """Final systems on small-preset seeds 5--9, never used during development."""
    rows = []
    for name, label in CODING.items():
        logs = [json.loads((RUNS / f"synthetic-small-s{s}" / f"{name}.json").read_text(encoding="utf-8"))
                for s in range(5, 10)]
        ans = [100 * sum(p["answer_verdict"] == "correct" for p in lg["probes"]) / len(lg["probes"]) for lg in logs]
        ev = [100 * sum(p["verdict"] == "correct" for p in lg["probes"]) / len(lg["probes"]) for lg in logs]
        wrong = sum(bool(p["answer_forbidden_hits"]) for lg in logs for p in lg["probes"])
        rows.append(f"{label} & {mean_sd(ans)} & {mean_sd(ev)} & {wrong} \\\\")
    return "\n".join(rows)


def efficiency_table() -> str:
    """Write and read cost on the small preset, seeds 0--4. Timings of LLM
    systems come from the uncached repeats in runs/noise (a cached replay is not
    a timing); the others are timed in their own runs."""
    live = {"ledger_llm": "ledger_llm", "hybrid_llm_fb": "hybrid_llm_fb", "hybrid_llm_aug": "hybrid_llm_fb"}
    rows = []
    for name, label in CODING.items():
        logs = [load(name, s) for s in SEEDS]
        if name in live:
            timed = [json.loads((RUNS / "noise" / f"synthetic-small-s{s}" / f"{live[name]}__r{k}.json")
                                .read_text(encoding="utf-8")) for s in SEEDS for k in (2, 3)]
        else:
            timed = logs
        ingest = [o for lg in timed for o in lg["ops"] if o["op"] == "ingest"]
        ing_ms = 1000 * statistics.median(o["latency_s"] for o in ingest)
        calls = statistics.mean(o["usage"]["llm_calls"] for lg in logs for o in lg["ops"] if o["op"] == "ingest")
        ret_ms = 1000 * statistics.median(p["latency_s"] for lg in timed for p in lg["probes"])
        kib = statistics.mean(list(lg["checkpoint_stats"].values())[-1]["storage_bytes"] for lg in logs) / 1024
        tokens = statistics.mean(p["injected_tokens"] for lg in logs for p in lg["probes"])
        n = sum(len(lg["probes"]) for lg in logs)
        mem = sum(_summary(lg)["memory"] for lg in logs)
        ans = sum(_summary(lg)["answer"] for lg in logs)
        fmt = lambda ms: f"{ms:.0f}" if ms >= 10 else f"{ms:.2f}"
        rows.append(f"{label} & {calls:.2f} & {fmt(ing_ms)} & {fmt(ret_ms)} & {kib:.1f} & {tokens:.0f} & "
                    f"{1000 * mem / n:.2f} & {1000 * ans / n:.2f} \\\\")
    return "\n".join(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coding_rows.tex").write_text(coding_table() + "\n", encoding="utf-8")
    (OUT / "lme_rows.tex").write_text(lme_table() + "\n", encoding="utf-8")
    (OUT / "heldout_rows.tex").write_text(heldout_table() + "\n", encoding="utf-8")
    (OUT / "efficiency_rows.tex").write_text(efficiency_table() + "\n", encoding="utf-8")
    for f in sorted(OUT.glob("*.tex")):
        print(f"== {f}\n{f.read_text()}")


if __name__ == "__main__":
    main()
