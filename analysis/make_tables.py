"""Generate the paper's result tables from the run logs (no hand-copied numbers).

    python analysis/make_tables.py        # writes results/tables/*.tex

Reads runs/synthetic-small-s{0..4}/*.json (coding workload, both deletion
modes), runs/longmemeval/main/*.json and, when present, the long-preset logs.
Mem0's logs come from their own sweeps, runs/mem0 (seeds 0--4) and
runs/mem0-heldout (seeds 5--9).
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from the repo root without PYTHONPATH

from interface.tokens import count_tokens
from runner.recount_tokens import lme_full_replay_tokens

RUNS = Path("runs")
OUT = Path("results/tables")
SEEDS = range(5)

#: log name -> label in the paper
CODING = {
    "no_memory": "No memory",
    "full_replay": "Full replay",
    "transcript_rag": "Transcript RAG",
    "mem0": "Mem0",
    "mem0_coding": "Mem0 (coding prompt)",
    "ledger": "\\sys{} (rules)",
    "ledger_llm": "\\sys{} (LLM)",
    "hybrid_llm_fb": "\\hyb{} fallback",
    "hybrid_llm_aug": "\\hyb{} augment",
}
LME = {
    "no_memory": "No memory",
    "full_replay": "Full replay",
    "transcript_rag": "Transcript RAG",
    "mem0": "Mem0",
    "ledger_llm_user": "\\sys{} (LLM)",
    "hybrid_llm_user_fallback": "\\hyb{} fallback",
    "hybrid_llm_user": "\\hyb{} augment",
}
LME_CATS = ["single-session-user", "single-session-assistant", "single-session-preference",
            "multi-session", "knowledge-update", "temporal-reasoning", "abstention"]
#: Systems whose logs live in their own sweep folders: (seeds 0--4, seeds 5--9).
ROOTS = {"mem0": ("mem0", "mem0-heldout"), "mem0_coding": ("mem0", "mem0-heldout")}
#: Systems timed from uncached reruns, because their logged runs replayed a
#: response cache: (folder under runs/, log name, repeats). Augment mode shares
#: fallback's write and routing path; mem0_coding ran after mem0 on each seed
#: and hit its cached probe embeddings.
TIMED_FROM = {
    "ledger_llm": ("noise", "ledger_llm", (2, 3)),
    "hybrid_llm_fb": ("noise", "hybrid_llm_fb", (2, 3)),
    "hybrid_llm_aug": ("noise", "hybrid_llm_fb", (2, 3)),
    "mem0_coding": ("mem0-timing", "mem0_coding", (2,)),
}
LME_SHORT = ["SS-U", "SS-A", "Pref", "Multi", "KU", "Temp", "Abs"]


def log_path(name: str, seed: int, suffix: str = "") -> Path:
    root = RUNS / ROOTS[name][seed >= 5] if name in ROOTS else RUNS
    return root / f"synthetic-small-s{seed}" / f"{name}{suffix}.json"


def load(name: str, seed: int, suffix: str = "") -> dict:
    return json.loads(log_path(name, seed, suffix).read_text(encoding="utf-8"))


def tokens_of(p: dict) -> int:
    """Injected tokens of one logged probe, in model tokens (o200k_base).
    Older logs stored a word count in ``injected_tokens``; the budget never
    cut their evidence (runner/recount_tokens.py), so the logged evidence is
    what was injected."""
    return sum(count_tokens(e["content"]) for e in p["evidence"])


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
        nosrc = [load(name, s, "__nosrc") for s in SEEDS] if log_path(name, 0, "__nosrc").exists() else None
        probes = [p for lg in logs for p in lg["probes"]]
        ev = [100 * sum(p["verdict"] == "correct" for p in lg["probes"]) / len(lg["probes"]) for lg in logs]
        ans = [100 * sum(p["answer_verdict"] == "correct" for p in lg["probes"]) / len(lg["probes"]) for lg in logs]
        fe, fa = forbidden(probes, "forbidden_hits"), forbidden(probes, "answer_forbidden_hits")
        wrong_ans = sum(bool(p["answer_forbidden_hits"]) for p in probes)
        tokens = statistics.mean(tokens_of(p) for p in probes)
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
        if inst[0]["evidence"] is None:  # full replay: from the provider's counts
            full = lme_full_replay_tokens()
            tokens = statistics.mean(full[r["question_id"]] for r in inst)
        else:
            tokens = statistics.mean(sum(count_tokens(t) for t in r["evidence"]) for r in inst)
        write, read = lme_write_read(name, inst)
        ans = statistics.mean(r["cost_usd"]["answer"] for r in inst)
        rows.append(f"{label} & {acc:.1f} ({boots[50]:.0f}--{boots[1949]:.0f}) & {cats} & "
                    f"{tokens:,.0f} & {1000 * write:.1f} & {1000 * read:.2f} & {1000 * ans:.2f} \\\\".replace(",", "{,}"))
    return "\n".join(rows)


def lme_write_read(name: str, inst: list[dict]) -> tuple[float, float]:
    """Mean write cost per history (ingest, paid once) and read cost per
    question (retrieval, paid every time), in USD. Logs written before the
    split carry only their sum; the only such paid system is Mem0, whose read
    is one embedding of the question (text-embedding-3-small, cl100k_base)."""
    if "write" in inst[0]["cost_usd"]:
        return (statistics.mean(r["cost_usd"]["write"] for r in inst),
                statistics.mean(r["cost_usd"]["read"] for r in inst))
    mem = statistics.mean(r["cost_usd"]["memory"] for r in inst)
    if name != "mem0":
        return mem, 0.0
    import tiktoken
    from runner.prices import EMBEDDING_PRICES
    enc = tiktoken.get_encoding("cl100k_base")
    read = statistics.mean(len(enc.encode(r["question"])) for r in inst) * EMBEDDING_PRICES["text-embedding-3-small"] / 1e6
    return mem - read, read


#: LongMemEval comparisons reported in the text: (system, baseline)
LME_PAIRS = [("hybrid_llm_user", "full_replay"), ("hybrid_llm_user", "transcript_rag"),
             ("hybrid_llm_user", "mem0")]


def paired_ci(a: list[int], b: list[int], resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Accuracy difference a - b in points on the same questions, with a 95%
    paired percentile bootstrap: resample questions, keep each question's
    pair of verdicts together, take the 2.5th and 97.5th percentiles."""
    d = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    boots = sorted(100 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(resamples))
    lo, hi = round(0.025 * resamples), round(0.975 * resamples) - 1
    return 100 * sum(d) / len(d), boots[lo], boots[hi]


def lme_paired() -> str:
    """Paired differences on LongMemEval, joined on question id."""
    def verdicts(name: str) -> dict[str, int]:
        path = RUNS / "longmemeval" / "main" / f"{name}.json"
        inst = json.loads(path.read_text(encoding="utf-8"))["instances"]
        return {r["question_id"]: int(r["correct"]) for r in inst if "error" not in r}

    rows = []
    for sys_, base in LME_PAIRS:
        a, b = verdicts(sys_), verdicts(base)
        ids = sorted(a.keys() & b.keys())
        diff, lo, hi = paired_ci([a[i] for i in ids], [b[i] for i in ids])
        rows.append(f"{LME[sys_]} vs {LME[base]} & {diff:+.1f} & ${lo:+.1f}$ to ${hi:+.1f}$ & {len(ids)} \\\\")
    return "\n".join(rows)


def heldout_table() -> str:
    """Final systems on small-preset seeds 5--9, never used during development."""
    rows = []
    for name, label in CODING.items():
        logs = [load(name, s) for s in range(5, 10)]
        ans = [100 * sum(p["answer_verdict"] == "correct" for p in lg["probes"]) / len(lg["probes"]) for lg in logs]
        ev = [100 * sum(p["verdict"] == "correct" for p in lg["probes"]) / len(lg["probes"]) for lg in logs]
        wrong = sum(bool(p["answer_forbidden_hits"]) for lg in logs for p in lg["probes"])
        rows.append(f"{label} & {mean_sd(ans)} & {mean_sd(ev)} & {wrong} \\\\")
    return "\n".join(rows)


def timed_logs(name: str) -> list[dict]:
    """Logs whose op and probe latencies are timings (seeds 0--4)."""
    if name not in TIMED_FROM:
        return [load(name, s) for s in SEEDS]
    folder, log, repeats = TIMED_FROM[name]
    return [json.loads((RUNS / folder / f"synthetic-small-s{s}" / f"{log}__r{k}.json").read_text(encoding="utf-8"))
            for s in SEEDS for k in repeats]


def efficiency_table() -> str:
    """Write and read cost on the small preset, seeds 0--4. Timings of systems
    in TIMED_FROM come from uncached reruns (a cached replay is not a timing);
    the others are timed in their own runs."""
    rows = []
    for name, label in CODING.items():
        logs = [load(name, s) for s in SEEDS]
        timed = timed_logs(name)
        ingest = [o for lg in timed for o in lg["ops"] if o["op"] == "ingest"]
        ing_ms = 1000 * statistics.median(o["latency_s"] for o in ingest)
        calls = statistics.mean(o["usage"]["llm_calls"] for lg in logs for o in lg["ops"] if o["op"] == "ingest")
        ret_ms = 1000 * statistics.median(p["latency_s"] for lg in timed for p in lg["probes"])
        kib = statistics.mean(list(lg["checkpoint_stats"].values())[-1]["storage_bytes"] for lg in logs) / 1024
        tokens = statistics.mean(tokens_of(p) for lg in logs for p in lg["probes"])
        n = sum(len(lg["probes"]) for lg in logs)
        sessions = sum(o["op"] == "ingest" for lg in logs for o in lg["ops"])
        write, read = (sum(x) for x in zip(*(_write_read(lg) for lg in logs)))
        ans = sum(_summary(lg)["answer"] for lg in logs)
        fmt = lambda ms: f"{ms:.0f}" if ms >= 10 else f"{ms:.2f}"
        rows.append(f"{label} & {calls:.2f} & {fmt(ing_ms)} & {fmt(ret_ms)} & {kib:.1f} & {tokens:.0f} & "
                    f"{1000 * write / sessions:.2f} & {1000 * read / n:.2f} & {1000 * ans / n:.2f} \\\\")
    return "\n".join(rows)


def _write_read(log: dict) -> tuple[float, float]:
    """USD of the write path (ingest and delete ops, paid once per session or
    request) and of the read path (retrieval, paid per question)."""
    from runner.prices import cost_usd
    from interface.memory_system import Usage
    from runner.replay import sum_usage
    return (cost_usd(sum_usage([Usage(**o["usage"]) for o in log["ops"]])),
            cost_usd(sum_usage([Usage(**p["usage"]) for p in log["probes"]])))


#: long-preset logs: (label, log name)
LONG = [("Replay", "full_replay"), ("hyb", "hybrid_llm_fb")]


def growth_tokens() -> str:
    """Long preset, seeds 0--2: probes per checkpoint (all seeds) and mean
    injected model tokens per question for full replay and LEDGER+EP."""
    out = []
    per = {}
    for _, name in LONG:
        logs = [json.loads((RUNS / f"synthetic-long-s{s}" / f"{name}.json").read_text()) for s in range(3)]
        cps = sorted({p["checkpoint"] for lg in logs for p in lg["probes"]}, key=lambda c: int(c[1:]))
        per[name] = {c: [tokens_of(p) for lg in logs for p in lg["probes"] if p["checkpoint"] == c] for c in cps}
    for c in per["full_replay"]:
        out.append(f"{c[1:]} & {len(per['full_replay'][c])} & {statistics.mean(per['full_replay'][c]):,.0f} & "
                   f"{statistics.mean(per['hybrid_llm_fb'][c]):.0f} \\\\".replace(",", "{,}"))
    return "\n".join(out)


def artifact_csvs() -> None:
    """Supporting data for the paper's statements: per-seed results, the cost
    breakdown and the LongMemEval pilot-overlap analysis."""
    import csv
    from workloads.longmemeval import load_subset
    with open(OUT / "per_seed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "seed", "probes", "answer_correct", "evidence_correct", "answers_with_forbidden",
                    "mean_injected_tokens_o200k"])
        for name in CODING:
            for s in range(10):
                lg = load(name, s)
                ps = lg["probes"]
                w.writerow([name, s, len(ps), sum(p["answer_verdict"] == "correct" for p in ps),
                            sum(p["verdict"] == "correct" for p in ps),
                            sum(bool(p["answer_forbidden_hits"]) for p in ps),
                            round(statistics.mean(tokens_of(p) for p in ps), 1)])
    with open(OUT / "cost_breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "system", "unit", "write_usd", "read_usd", "answer_usd", "sessions", "questions"])
        for name in CODING:
            logs = [load(name, s) for s in SEEDS]
            wr = [_write_read(lg) for lg in logs]
            w.writerow(["coding seeds 0-4", name, "total", round(sum(x for x, _ in wr), 6),
                        round(sum(y for _, y in wr), 6), round(sum(_summary(lg)["answer"] for lg in logs), 6),
                        sum(o["op"] == "ingest" for lg in logs for o in lg["ops"]),
                        sum(len(lg["probes"]) for lg in logs)])
        for name in LME:
            inst = [r for r in json.loads((RUNS / "longmemeval" / "main" / f"{name}.json").read_text())["instances"]
                    if "error" not in r]
            write, read = lme_write_read(name, inst)
            w.writerow(["LongMemEval main", name, "mean per history/question", round(write, 6), round(read, 6),
                        round(statistics.mean(r["cost_usd"]["answer"] for r in inst), 6),
                        round(statistics.mean(r["sessions"] for r in inst), 1), len(inst)])
    overlap = sorted(set(load_subset("pilot")) & set(load_subset("main")))
    with open(OUT / "pilot_exclusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "accuracy_all_210", "accuracy_without_pilot", "questions_without_pilot",
                    "pilot_overlap_ids"])
        for name in LME:
            inst = [r for r in json.loads((RUNS / "longmemeval" / "main" / f"{name}.json").read_text())["instances"]
                    if "error" not in r]
            rest = [r for r in inst if r["question_id"] not in overlap]
            w.writerow([name, round(100 * sum(r["correct"] for r in inst) / len(inst), 1),
                        round(100 * sum(r["correct"] for r in rest) / len(rest), 1), len(rest), " ".join(overlap)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coding_rows.tex").write_text(coding_table() + "\n", encoding="utf-8")
    (OUT / "lme_rows.tex").write_text(lme_table() + "\n", encoding="utf-8")
    (OUT / "lme_paired.tex").write_text(lme_paired() + "\n", encoding="utf-8")
    (OUT / "heldout_rows.tex").write_text(heldout_table() + "\n", encoding="utf-8")
    (OUT / "efficiency_rows.tex").write_text(efficiency_table() + "\n", encoding="utf-8")
    (OUT / "growth_tokens.tex").write_text(growth_tokens() + "\n", encoding="utf-8")
    artifact_csvs()
    for f in sorted(OUT.glob("*.tex")):
        print(f"== {f}\n{f.read_text()}")


if __name__ == "__main__":
    main()
