"""Build an Excel workbook with the data and analysis behind the paper.

    python paper/make_workbook.py      # writes paper/results.xlsx

Raw per-probe, per-question and per-checkpoint rows come from the same run logs
as paper/make_tables.py; every table that the paper reports is recomputed in the
workbook with formulas over those rows. Values that cannot be formulas (medians
of timings over thousands of ops, bootstrap intervals, dollar costs priced by
runner/prices.py) are written as inputs in blue with a note on their source.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]
import make_tables as mt  # noqa: E402
from interface.memory_system import Usage  # noqa: E402
from runner.prices import cost_usd  # noqa: E402

RUNS = Path("runs")
OUT = Path("paper/results.xlsx")

FONT = "Arial"
BLUE = "0000FF"
NAVY = "14213D"
HEAD_FILL = PatternFill("solid", fgColor=NAVY)
SUB_FILL = PatternFill("solid", fgColor="E9E5DA")
OURS_FILL = PatternFill("solid", fgColor="FBEBDD")
THIN = Side(style="thin", color="BFBFBF")

LABEL = {"no_memory": "No memory", "full_replay": "Full replay", "transcript_rag": "Transcript RAG",
         "mem0": "Mem0", "mem0_coding": "Mem0 (coding prompt)", "ledger": "LEDGER (rules)",
         "ledger_llm": "LEDGER (LLM)", "hybrid_llm_fb": "LEDGER+EP fallback",
         "hybrid_llm_aug": "LEDGER+EP augment"}
LME_LABEL = {"no_memory": "No memory", "full_replay": "Full replay", "transcript_rag": "Transcript RAG",
             "mem0": "Mem0", "ledger_llm_user": "LEDGER (LLM)",
             "hybrid_llm_user_fallback": "LEDGER+EP fallback", "hybrid_llm_user": "LEDGER+EP augment"}
OURS = {"LEDGER (LLM)", "LEDGER+EP fallback", "LEDGER+EP augment", "LEDGER (rules)"}
PROBE_TYPES = ["current", "historical", "post_update", "contamination", "post_delete", "abstention"]
#: upgrade variant -> its label in Long_Checkpoints
UPGRADE_LONG = {"gpt-4.1-mini throughout": "LEDGER+EP augment",
                "mini, upgraded to gpt-4.1 after c4": "LEDGER+EP augment, upgraded after c4",
                "gpt-4.1 throughout": "LEDGER+EP augment, gpt-4.1 throughout"}
LONG_SYSTEMS = ["no_memory", "full_replay", "transcript_rag", "mem0", "mem0_coding",
                "ledger", "ledger_llm", "hybrid_llm_fb", "hybrid_llm_aug"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def price(usages: list[dict]) -> float:
    """Dollar cost, priced call by call so runs that mix models can be priced."""
    return sum(cost_usd(Usage(**u)) for u in usages if u.get("llm_calls") or u.get("embedding_tokens"))


def costs(log: dict) -> tuple[float, float]:
    mem = [o["usage"] for o in log["ops"]] + [p["usage"] for p in log["probes"]]
    ans = [p["answer_usage"] for p in log["probes"] if p.get("answer_usage")]
    return price(mem), price(ans)


def has(hits: list[dict], kind: str) -> int:
    return int(any(kind in h["error_types"] for h in hits))


# --------------------------------------------------------------------------- styling

def style_header(ws, row: int, ncols: int, col0: int = 1) -> None:
    for c in range(col0, col0 + ncols):
        cell = ws.cell(row, c)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF")
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def title(ws, text: str, sub: str | None = None) -> int:
    ws["A1"] = text
    ws["A1"].font = Font(name=FONT, bold=True, size=14, color=NAVY)
    if sub:
        ws["A2"] = sub
        ws["A2"].font = Font(name=FONT, italic=True, color="595959")
        ws["A2"].alignment = Alignment(wrap_text=False)
    return 4


def finish(ws, widths: dict[str, float] | None = None, freeze: str | None = None) -> None:
    for row in ws.iter_rows():
        for cell in row:
            f = cell.font
            if f.name != FONT:
                cell.font = Font(name=FONT, bold=f.bold, italic=f.italic, size=f.size, color=f.color)
    for col, w in (widths or {}).items():
        ws.column_dimensions[col].width = w
    if freeze:
        ws.freeze_panes = freeze


def put(ws, row: int, values: list, raw: bool = False, fmt: list[str | None] | None = None) -> None:
    for i, v in enumerate(values, 1):
        cell = ws.cell(row, i, v)
        cell.font = Font(name=FONT, color=BLUE if raw and not (isinstance(v, str) and v.startswith("=")) else None)
        if fmt and i <= len(fmt) and fmt[i - 1]:
            cell.number_format = fmt[i - 1]


def table_rows(ws, header: list[str], rows: list[list], start: int, raw: bool = True,
               fmt: list[str | None] | None = None) -> int:
    for i, h in enumerate(header, 1):
        ws.cell(start, i, h)
    style_header(ws, start, len(header))
    for k, r in enumerate(rows, start + 1):
        put(ws, k, r, raw=raw, fmt=fmt)
    return start + len(rows)


def note(ws, row: int, text: str) -> None:
    ws.cell(row, 1, text).font = Font(name=FONT, italic=True, color="595959")


# --------------------------------------------------------------------------- raw data

def coding_probes() -> tuple[list[list], list[list]]:
    """One row per probe (dev seeds 0-4 and held-out 5-9) and one row per run."""
    probes, runs = [], []
    for name in LABEL:
        for seed in range(10):
            log = load(mt.log_path(name, seed))
            split = "dev" if seed < 5 else "held-out"
            for p in log["probes"]:
                ev, ans = p["forbidden_hits"], p["answer_forbidden_hits"]
                probes.append([split, seed, LABEL[name], p["probe_id"], p["probe_type"], p["checkpoint"],
                               p["verdict"], p["answer_verdict"], int(p["verdict"] == "correct"),
                               int(p["answer_verdict"] == "correct"), int(bool(ans)), int(bool(ev)),
                               has(ev, "stale"), has(ev, "future"), has(ev, "contaminated"),
                               p["injected_tokens"], p["question"]])
            mem, ans = costs(log)
            ingest = [o for o in log["ops"] if o["op"] == "ingest"]
            last = list(log["checkpoint_stats"].values())[-1]
            runs.append([split, seed, LABEL[name], len(log["probes"]), len(ingest),
                         sum(o["usage"]["llm_calls"] for o in ingest), last["storage_bytes"], mem, ans])
    return probes, runs


def deletion_rows() -> list[list]:
    rows = []
    for name in LABEL:
        if not mt.log_path(name, 0, "__nosrc").exists():
            continue
        for seed in range(5):
            log = load(mt.log_path(name, seed, "__nosrc"))
            ps = log["probes"]
            rows.append([LABEL[name], seed, len(ps),
                         sum(has(p["forbidden_hits"], "resurrected") for p in ps),
                         sum(has(p.get("answer_forbidden_hits", []), "resurrected") for p in ps),
                         sum(p["answer_verdict"] == "correct" for p in ps)])
    return rows


def efficiency_inputs() -> list[list]:
    """Pooled medians of timings, computed exactly as make_tables.efficiency_table."""
    live = {"ledger_llm": "ledger_llm", "hybrid_llm_fb": "hybrid_llm_fb", "hybrid_llm_aug": "hybrid_llm_fb"}
    rows = []
    for name in LABEL:
        if name in mt.UNTIMED:
            rows.append([LABEL[name], None, None, "shares Mem0's embedding cache; not timed"])
            continue
        if name in live:
            timed = [load(RUNS / "noise" / f"synthetic-small-s{s}" / f"{live[name]}__r{k}.json")
                     for s in range(5) for k in (2, 3)]
            src = f"uncached repeats runs/noise/*/{live[name]}__r2,r3"
        else:
            timed = [mt.load(name, s) for s in range(5)]
            src = "own runs, seeds 0-4"
        ing = 1000 * statistics.median(o["latency_s"] for lg in timed for o in lg["ops"] if o["op"] == "ingest")
        ret = 1000 * statistics.median(p["latency_s"] for lg in timed for p in lg["probes"])
        rows.append([LABEL[name], ing, ret, src])
    return rows


def lme_rows() -> tuple[list[list], dict[str, tuple[float, float]]]:
    rows, ci = [], {}
    rng = random.Random(0)  # same draw order as make_tables.lme_table
    for name, label in LME_LABEL.items():
        path = RUNS / "longmemeval" / "main" / f"{name}.json"
        inst = [r for r in load(path)["instances"] if "error" not in r]
        boots = sorted(100 * sum(rng.choice(inst)["correct"] for _ in inst) / len(inst) for _ in range(2000))
        ci[label] = (boots[50], boots[1949])
        for r in inst:
            rows.append([label, r["question_id"], r["category"], int(r["correct"]), int(bool(r["abstained"])),
                         r["injected_tokens"], r["history_tokens"], r["cost_usd"]["memory"],
                         r["cost_usd"]["answer"], r["cost_usd"]["judge"], r["question"], r["gold"],
                         r["response"]])
    return rows, ci


def long_rows() -> list[list]:
    rows = []
    variants = LONG_SYSTEMS + ["hybrid_llm_aug__gpt-4.1", "hybrid_llm_aug__upgrade"]
    for name in variants:
        for seed in range(3):
            root = RUNS / "mem0" if name.startswith("mem0") else RUNS
            path = root / f"synthetic-long-s{seed}" / f"{name}.json"
            if not path.exists():
                continue
            log = load(path)
            by: dict[str, list] = {}
            for p in log["probes"]:
                by.setdefault(p["checkpoint"], []).append(p)
            label = LABEL.get(name, {"hybrid_llm_aug__gpt-4.1": "LEDGER+EP augment, gpt-4.1 throughout",
                                     "hybrid_llm_aug__upgrade": "LEDGER+EP augment, upgraded after c4"}.get(name))
            for cp in sorted(by, key=lambda c: int(c[1:])):
                ps = by[cp]
                st = log["checkpoint_stats"].get(cp, {})
                rows.append([label, seed, int(cp[1:]), len(ps),
                             sum(p["answer_verdict"] == "correct" for p in ps),
                             sum(p["verdict"] == "correct" for p in ps),
                             sum(bool(p["forbidden_hits"]) for p in ps),
                             sum(has(p["forbidden_hits"], "stale") for p in ps),
                             sum(has(p["forbidden_hits"], "future") for p in ps),
                             statistics.mean(p["injected_tokens"] for p in ps),
                             1000 * statistics.mean(p["latency_s"] for p in ps),
                             st.get("storage_bytes"), st.get("n_items")])
    return rows


def upgrade_rows() -> list[list]:
    rows = []
    for name, label in [("hybrid_llm_aug", "gpt-4.1-mini throughout"),
                        ("hybrid_llm_aug__upgrade", "mini, upgraded to gpt-4.1 after c4"),
                        ("hybrid_llm_aug__gpt-4.1", "gpt-4.1 throughout")]:
        for seed in (0, 1):
            log = load(RUNS / f"synthetic-long-s{seed}" / f"{name}.json")
            sec = [p for p in log["probes"] if int(p["checkpoint"][1:]) >= 5]
            mem, _ = costs(log)
            rows.append([label, seed, len(sec), sum(p["answer_verdict"] == "correct" for p in sec), mem])
    return rows


def noise_rows() -> list[list]:
    rows = []
    for name in ("ledger_llm", "hybrid_llm_fb"):
        for seed in range(5):
            logs = [load(RUNS / "noise" / f"synthetic-small-s{seed}" / f"{name}{suf}.json")
                    for suf in ("", "__r2", "__r3")]
            for ps in zip(*[lg["probes"] for lg in logs]):
                rows.append([LABEL[name], seed, ps[0]["probe_id"], ps[0]["probe_type"]]
                            + [int(p["answer_verdict"] == "correct") for p in ps])
    return rows


def restart_rows() -> list[list]:
    rows = []
    for key, r in load(RUNS / "restart-long.json").items():
        wl, name = key.split("/")
        rows.append([LABEL.get(name, name), int(wl[-1]), r["probes"], r["identical"], r["restarts"],
                     max(r["snapshot_bytes"]) / 1024, max(r["restore_ms"]), r["llm_calls"]])
    return rows


def audit_rows() -> list[list]:
    key = {k["item"]: k["system"] for k in load(RUNS / "longmemeval" / "audit" / "judge_audit_key.json")}
    rows = []
    with open(RUNS / "longmemeval" / "audit" / "judge_audit_human.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            item = int(r["item"])
            rows.append([item, LME_LABEL.get(key[item], key[item]), r["question_id"], r["category"],
                         r["judge_said_correct"], r["HUMAN_correct (yes/no)"], r["HUMAN_note"],
                         r["question"], r["gold_answer"], r["model_response"]])
    return rows


# --------------------------------------------------------------------------- sheets

def readme(wb: Workbook) -> None:
    ws = wb.active
    ws.title = "README"
    title(ws, "Agent memory paper: data and analysis",
          "Generated by paper/make_workbook.py from the run logs in runs/; regenerate after any rerun.")
    rows = [
        ("Summary", "Every number quoted in the abstract and results text, recomputed, next to the paper's value"),
        ("Coding_Results", "Tables 3 and 4: coding workload, dev seeds 0-4 and held-out seeds 5-9"),
        ("Coding_ByType", "Answer accuracy by probe type (dev seeds); where each system fails"),
        ("Deletion", "Description-only deletion: deleted value in evidence and in answers"),
        ("Efficiency", "Table 5: LLM calls, timings, store size, tokens, cost per probe"),
        ("LME_Results", "Table 6: LongMemEval_S subset, accuracy by category, tokens, cost"),
        ("Long_Summary", "Fig. 2: long preset, answer accuracy and forbidden evidence per checkpoint"),
        ("Restart_Upgrade", "Restart after every checkpoint; extraction model upgraded mid-stream"),
        ("Noise", "Uncached repeats: answer flips and per-seed SD"),
        ("Judge_Audit", "Human audit of 60 LLM-judge decisions: agreement and Cohen's kappa"),
        ("Workloads", "Table 2: workload sizes"),
        ("Ablations", "Table 7: what changed LEDGER's evidence accuracy"),
        ("Requirements", "Table 1: challenge requirements and where they are addressed"),
        ("Coding_Probes", "Raw: one row per probe, system and seed (10 seeds, 9 systems)"),
        ("Coding_Runs", "Raw: one row per run (probes, sessions, LLM calls, store size, $ cost)"),
        ("LME_Questions", "Raw: one row per LongMemEval question and system"),
        ("Long_Checkpoints", "Raw: one row per long-preset checkpoint, system and seed"),
    ]
    r = table_rows(ws, ["Sheet", "Contents"], [list(x) for x in rows], 4, raw=False)
    r += 2
    for line in [
        "Conventions",
        "Blue text: data copied from the run logs (inputs). Black text: formulas over those inputs.",
        "Accuracy = % of probes or questions judged correct. Mean ± SD is over seeds (sample SD).",
        "Evidence level = what memory put in the model's context; answer level = what the shared answerer said.",
        "All systems use one answering model (gpt-4.1-mini), one prompt and one price table (runner/prices.py).",
        "LongMemEval judge: gpt-4o-2024-08-06 with the benchmark's prompts.",
        "Costs are in US dollars; m$ = thousandths of a dollar.",
        "LEDGER = the paper's \\sys{} (fact ledger); LEDGER+EP = \\hyb{} (ledger plus episodic fallback).",
    ]:
        ws.cell(r, 1, line).font = Font(name=FONT, bold=line == "Conventions")
        r += 1
    finish(ws, {"A": 22, "B": 100})


def coding_raw(wb: Workbook, probes: list[list], runs: list[list]) -> tuple[int, int]:
    ws = wb.create_sheet("Coding_Probes")
    hdr = ["Split", "Seed", "System", "Probe", "Probe type", "Checkpoint", "Evidence verdict",
           "Answer verdict", "Evidence correct", "Answer correct", "Wrong answer (forbidden value)",
           "Forbidden in evidence", "Stale in evidence", "Future in evidence", "Contaminated in evidence",
           "Injected tokens", "Question"]
    last = table_rows(ws, hdr, probes, 1)
    ws.auto_filter.ref = f"A1:Q{last}"
    finish(ws, {"A": 9, "B": 6, "C": 22, "D": 8, "E": 14, "F": 10, "G": 16, "H": 16, "Q": 70}, "D2")
    ws2 = wb.create_sheet("Coding_Runs")
    hdr = ["Split", "Seed", "System", "Probes", "Sessions ingested", "Write-time LLM calls",
           "Store bytes at end", "Memory $", "Answer $"]
    last2 = table_rows(ws2, hdr, runs, 1, fmt=[None] * 6 + ["#,##0", "0.0000", "0.0000"])
    ws2.auto_filter.ref = f"A1:I{last2}"
    finish(ws2, {"A": 9, "C": 22, "E": 12, "F": 12, "G": 12, "H": 11, "I": 11}, "D2")
    return last, last2


def rng(col: str, last: int, sheet: str = "Coding_Probes") -> str:
    return f"{sheet}!${col}$2:${col}${last}"


def coding_results(wb: Workbook, last: int, lastr: int) -> dict[str, dict[str, str]]:
    ws = wb.create_sheet("Coding_Results", 1)
    title(ws, "Coding-agent workload (Tables 3 and 4)",
          "Small preset; dev seeds 0-4 (512 probes) were used during development, held-out seeds 5-9 (526) never were.")
    sysr, seedr, splitr = rng("C", last), rng("B", last), rng("A", last)
    refs: dict[str, dict[str, str]] = {}
    row = 4
    for split, seeds, label in [("dev", range(5), "Dev seeds 0-4"), ("held-out", range(5, 10), "Held-out seeds 5-9")]:
        ws.cell(row, 1, label).font = Font(name=FONT, bold=True, size=12)
        row += 1
        hdr = (["System"] + [f"Ans s{s}" for s in seeds] + ["Answer mean", "Answer SD"]
               + [f"Ev s{s}" for s in seeds] + ["Evidence mean", "Evidence SD", "Probes", "Wrong answers",
                                                "Stale in ev.", "Future in ev.", "Contam. in ev.",
                                                "Tokens/probe", "Memory+answer $"])
        for i, h in enumerate(hdr, 1):
            ws.cell(row, i, h)
        style_header(ws, row, len(hdr))
        for name, lab in LABEL.items():
            row += 1
            ws.cell(row, 1, lab)
            if lab in OURS:
                for c in range(1, len(hdr) + 1):
                    ws.cell(row, c).fill = OURS_FILL
            q = f'{sysr},$A{row},{splitr},"{split}"'
            for k, s in enumerate(seeds):
                base = f'{sysr},$A{row},{seedr},{s}'
                ws.cell(row, 2 + k, f"=100*COUNTIFS({base},{rng('J', last)},1)/COUNTIFS({base})")
                ws.cell(row, 9 + k, f"=100*COUNTIFS({base},{rng('I', last)},1)/COUNTIFS({base})")
            ws.cell(row, 7, f"=AVERAGE(B{row}:F{row})")
            ws.cell(row, 8, f"=STDEV(B{row}:F{row})")
            ws.cell(row, 14, f"=AVERAGE(I{row}:M{row})")
            ws.cell(row, 15, f"=STDEV(I{row}:M{row})")
            ws.cell(row, 16, f"=COUNTIFS({q})")
            for c, col in zip(range(17, 21), "KMNO"):
                ws.cell(row, c, f"=COUNTIFS({q},{rng(col, last)},1)")
            ws.cell(row, 21, f"=AVERAGEIFS({rng('P', last)},{q})")
            ws.cell(row, 22, f"=SUMIFS({rng('H', lastr, 'Coding_Runs')},{rng('C', lastr, 'Coding_Runs')},$A{row},"
                             f"{rng('A', lastr, 'Coding_Runs')},\"{split}\")"
                             f"+SUMIFS({rng('I', lastr, 'Coding_Runs')},{rng('C', lastr, 'Coding_Runs')},$A{row},"
                             f"{rng('A', lastr, 'Coding_Runs')},\"{split}\")")
            for c in list(range(2, 16)):
                ws.cell(row, c).number_format = "0.0"
            ws.cell(row, 21).number_format = "0"
            ws.cell(row, 22).number_format = "0.00"
            refs.setdefault(split, {})[name] = str(row)
        row += 3
    note(ws, row, "Wrong answers: answers containing a stale, future, contaminated or deleted value. "
                  "Forbidden in evidence: probes whose evidence carried such a value (a probe can count in several).")
    note(ws, row + 1, "Paper rounds means and SDs to one decimal; Table 4 (held-out) reports answer, evidence and wrong answers.")
    # chart: answer accuracy dev vs held-out
    ch = BarChart()
    ch.type = "bar"
    ch.title = "Answer accuracy (%), mean over seeds"
    ch.y_axis.title = "% correct"
    first_dev = int(refs["dev"]["no_memory"])
    first_ho = int(refs["held-out"]["no_memory"])
    n = len(LABEL)
    ch.add_data(Reference(ws, min_col=7, min_row=first_dev, max_row=first_dev + n - 1), titles_from_data=False)
    ch.add_data(Reference(ws, min_col=7, min_row=first_ho, max_row=first_ho + n - 1), titles_from_data=False)
    ch.set_categories(Reference(ws, min_col=1, min_row=first_dev, max_row=first_dev + n - 1))
    ch.series[0].tx = None
    from openpyxl.chart.series import SeriesLabel
    ch.series[0].tx = SeriesLabel(v="Dev seeds 0-4")
    ch.series[1].tx = SeriesLabel(v="Held-out seeds 5-9")
    ch.height, ch.width = 10, 20
    ws.add_chart(ch, f"A{row + 4}")
    finish(ws, {"A": 24, **{get_column_letter(c): 9 for c in range(2, 23)}}, "B4")
    return refs


def coding_by_type(wb: Workbook, last: int) -> None:
    ws = wb.create_sheet("Coding_ByType", 2)
    title(ws, "Answer accuracy by probe type (dev seeds 0-4)",
          "Correct / total per probe type, pooled over seeds. Historical = question about an earlier date.")
    hdr = ["System"] + [f"{t} %" for t in PROBE_TYPES] + [f"{t} n" for t in PROBE_TYPES]
    for i, h in enumerate(hdr, 1):
        ws.cell(4, i, h)
    style_header(ws, 4, len(hdr))
    for k, lab in enumerate(LABEL.values(), 5):
        ws.cell(k, 1, lab)
        for j, t in enumerate(PROBE_TYPES):
            base = f'{rng("C", last)},$A{k},{rng("A", last)},"dev",{rng("E", last)},"{t}"'
            ws.cell(k, 2 + j, f"=100*COUNTIFS({base},{rng('J', last)},1)/COUNTIFS({base})").number_format = "0.0"
            ws.cell(k, 8 + j, f"=COUNTIFS({base})")
    finish(ws, {"A": 24, **{get_column_letter(c): 14 for c in range(2, 14)}}, "B5")


def deletion(wb: Workbook, rows: list[list]) -> dict[str, str]:
    ws = wb.create_sheet("Deletion", 3)
    title(ws, "Description-only deletion (dev seeds 0-4)",
          "Deletion requests name the fact but not its source events; a baseline must find what to forget itself.")
    last = table_rows(ws, ["System", "Seed", "Probes", "Deleted value in evidence", "Deleted value in answer",
                           "Answer correct"], rows, 4)
    r = last + 3
    ws.cell(r - 1, 1, "Totals over seeds").font = Font(name=FONT, bold=True, size=12)
    hdr = ["System", "Probes", "Deleted value sent to model", "Deleted value repeated in answer", "Answer accuracy %"]
    for i, h in enumerate(hdr, 1):
        ws.cell(r, i, h)
    style_header(ws, r, len(hdr))
    refs = {}
    seen = []
    for row in rows:
        if row[0] not in seen:
            seen.append(row[0])
    for k, lab in enumerate(seen, r + 1):
        ws.cell(k, 1, lab)
        for c, col in zip((2, 3, 4), "CDE"):
            ws.cell(k, c, f"=SUMIFS(${col}$5:${col}${last},$A$5:$A${last},$A{k})")
        ws.cell(k, 5, f"=100*SUMIFS($F$5:$F${last},$A$5:$A${last},$A{k})/B{k}").number_format = "0.0"
        refs[lab] = str(k)
    finish(ws, {"A": 24, "B": 10, "C": 16, "D": 16, "E": 16, "F": 14}, "A5")
    return refs


def efficiency(wb: Workbook, inputs: list[list], last: int, lastr: int) -> dict[str, str]:
    ws = wb.create_sheet("Efficiency", 4)
    title(ws, "Efficiency on the coding workload (Table 5)",
          "Dev seeds 0-4. Timings are medians pooled over all ops (inputs, blue); everything else is a formula.")
    hdr = ["System", "LLM calls/session", "Ingest ms (median)", "Retrieve ms (median)", "Store KiB (mean per seed)",
           "Tokens/probe", "Memory m$/probe", "Answer m$/probe", "Total m$/probe", "Timing source"]
    for i, h in enumerate(hdr, 1):
        ws.cell(4, i, h)
    style_header(ws, 4, len(hdr))
    R = lambda c: rng(c, lastr, "Coding_Runs")  # noqa: E731
    refs = {}
    for k, (lab, ing, ret, src) in enumerate(inputs, 5):
        dev = f'{R("C")},$A{k},{R("A")},"dev"'
        ws.cell(k, 1, lab)
        ws.cell(k, 2, f"=SUMIFS({R('F')},{dev})/SUMIFS({R('E')},{dev})").number_format = "0.00"
        for c, v in ((3, ing), (4, ret)):
            cell = ws.cell(k, c, v if v is not None else "--")
            cell.font = Font(name=FONT, color=BLUE)
            cell.number_format = "0.00"
        ws.cell(k, 5, f"=AVERAGEIFS({R('G')},{dev})/1024").number_format = "0.0"
        ws.cell(k, 6, f'=AVERAGEIFS({rng("P", last)},{rng("C", last)},$A{k},{rng("A", last)},"dev")').number_format = "0"
        ws.cell(k, 7, f"=1000*SUMIFS({R('H')},{dev})/SUMIFS({R('D')},{dev})").number_format = "0.00"
        ws.cell(k, 8, f"=1000*SUMIFS({R('I')},{dev})/SUMIFS({R('D')},{dev})").number_format = "0.00"
        ws.cell(k, 9, f"=G{k}+H{k}").number_format = "0.00"
        ws.cell(k, 10, src).font = Font(name=FONT, color=BLUE)
        refs[lab] = str(k)
    note(ws, 6 + len(inputs), "Augment mode shares fallback's write and routing path, so it reuses fallback's uncached timings.")
    r = 8 + len(inputs)
    ws.cell(r, 1, "Routing call per question, LEDGER (LLM) and LEDGER+EP fallback pooled (as quoted in the text)"
            ).font = Font(name=FONT, bold=True)
    lat = sorted(p["latency_s"] for n in ("ledger_llm", "hybrid_llm_fb") for s in range(5) for k in (2, 3)
                 for p in load(RUNS / "noise" / f"synthetic-small-s{s}" / f"{n}__r{k}.json")["probes"])
    for i, (lab, v) in enumerate([("Median ms", 1000 * statistics.median(lat)),
                                  ("95th percentile ms", 1000 * lat[int(0.95 * len(lat))])], r + 1):
        ws.cell(i, 1, lab)
        c = ws.cell(i, 2, v)
        c.font = Font(name=FONT, color=BLUE)
        c.number_format = "0"
    refs["routing"] = str(r + 1)
    finish(ws, {"A": 24, **{get_column_letter(c): 13 for c in range(2, 10)}, "J": 44}, "B5")
    return refs


def lme(wb: Workbook, rows: list[list], ci: dict) -> dict[str, str]:
    raw = wb.create_sheet("LME_Questions")
    last = table_rows(raw, ["System", "Question id", "Category", "Correct", "Abstained", "Injected tokens",
                            "History tokens", "Memory $", "Answer $", "Judge $", "Question", "Gold answer",
                            "Response"], rows, 1, fmt=[None] * 7 + ["0.00000"] * 3)
    raw.auto_filter.ref = f"A1:M{last}"
    finish(raw, {"A": 22, "B": 14, "C": 24, "K": 60, "L": 30, "M": 60}, "B2")

    ws = wb.create_sheet("LME_Results", 5)
    title(ws, "LongMemEval_S, stratified subset of 210 questions (Table 6)",
          "30 questions per category; judge gpt-4o-2024-08-06 with the benchmark's prompts.")
    cats = mt.LME_CATS
    hdr = (["System", "Questions", "Accuracy %", "CI low", "CI high"] + mt.LME_SHORT
           + ["Tokens", "Memory m$/q", "Answer m$/q", "Judge m$/q"])
    for i, h in enumerate(hdr, 1):
        ws.cell(4, i, h)
    style_header(ws, 4, len(hdr))
    Q = lambda c: rng(c, last, "LME_Questions")  # noqa: E731
    refs = {}
    for k, lab in enumerate(LME_LABEL.values(), 5):
        s = f"{Q('A')},$A{k}"
        ws.cell(k, 1, lab)
        if lab in OURS:
            for c in range(1, len(hdr) + 1):
                ws.cell(k, c).fill = OURS_FILL
        ws.cell(k, 2, f"=COUNTIFS({s})")
        ws.cell(k, 3, f"=100*SUMIFS({Q('D')},{s})/B{k}").number_format = "0.0"
        for c, v in ((4, ci[lab][0]), (5, ci[lab][1])):
            cell = ws.cell(k, c, v)
            cell.font = Font(name=FONT, color=BLUE)
            cell.number_format = "0"
        for j, cat in enumerate(cats):
            ws.cell(k, 6 + j, f'=100*SUMIFS({Q("D")},{s},{Q("C")},"{cat}")/COUNTIFS({s},{Q("C")},"{cat}")'
                    ).number_format = "0"
        ws.cell(k, 13, f"=AVERAGEIFS({Q('F')},{s})").number_format = "#,##0"
        for c, col in ((14, "H"), (15, "I"), (16, "J")):
            ws.cell(k, c, f"=1000*AVERAGEIFS({Q(col)},{s})").number_format = "0.0"
        refs[lab] = str(k)
    ws.cell(4, 4).comment = Comment("95% bootstrap interval, 2,000 resamples, random.Random(0), as in "
                                    "paper/make_tables.py. Not recomputable by formula.", "make_workbook")
    note(ws, 13, "SS-U/SS-A/Pref = single-session user/assistant/preference; Multi = multi-session; "
                 "KU = knowledge update; Temp = temporal reasoning; Abs = abstention.")
    note(ws, 14, "Memory cost is write-time extraction, paid once per history; tokens = mean injected per question.")
    r = 16
    ws.cell(r, 1, "Paired differences vs full replay (same questions)").font = Font(name=FONT, bold=True, size=12)
    for i, h in enumerate(["System", "Both right", "Only this", "Only full replay", "Both wrong", "Diff (pts)"], 1):
        ws.cell(r + 1, i, h)
    style_header(ws, r + 1, 6)
    for k, lab in enumerate([x for x in LME_LABEL.values() if x != "Full replay"], r + 2):
        ws.cell(k, 1, lab)
        ws.cell(k, 6, f"=C{refs[lab]}-C{refs['Full replay']}").number_format = "+0.0;-0.0;0.0"
        # paired counts join on question id: COUNTIFS looks up full replay's verdict for each row
        a = f'({Q("A")}="{lab}")'
        ws.cell(k, 2, f'=SUMPRODUCT({a}*{Q("D")}*COUNTIFS({Q("A")},"Full replay",{Q("B")},{Q("B")},{Q("D")},1))')
        ws.cell(k, 3, f'=SUMPRODUCT({a}*{Q("D")}*COUNTIFS({Q("A")},"Full replay",{Q("B")},{Q("B")},{Q("D")},0))')
        ws.cell(k, 4, f'=SUMPRODUCT({a}*(1-{Q("D")})*COUNTIFS({Q("A")},"Full replay",{Q("B")},{Q("B")},{Q("D")},1))')
        ws.cell(k, 5, f'=SUMPRODUCT({a}*(1-{Q("D")})*COUNTIFS({Q("A")},"Full replay",{Q("B")},{Q("B")},{Q("D")},0))')
    ch = BarChart()
    ch.title = "LongMemEval accuracy (%)"
    ch.add_data(Reference(ws, min_col=3, min_row=4, max_row=4 + len(LME_LABEL)), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=4 + len(LME_LABEL)))
    ch.legend = None
    ch.height, ch.width = 8, 18
    ws.add_chart(ch, "H16")
    finish(ws, {"A": 24, **{get_column_letter(c): 10 for c in range(2, 17)}}, "B5")
    return refs


def long_sheets(wb: Workbook, rows: list[list]) -> dict[str, str]:
    raw = wb.create_sheet("Long_Checkpoints")
    hdr = ["System", "Seed", "Checkpoint", "Probes", "Answer correct", "Evidence correct",
           "Forbidden in evidence", "Stale in evidence", "Future in evidence", "Injected tokens (mean)",
           "Retrieve ms (mean)", "Store bytes", "Store items", "Answer %", "Evidence %"]
    last = table_rows(raw, hdr, [r + [None, None] for r in rows], 1, fmt=[None] * 9 + ["0.0", "0.000", "#,##0"])
    for k in range(2, last + 1):
        raw.cell(k, 14, f"=100*E{k}/D{k}").number_format = "0.0"
        raw.cell(k, 15, f"=100*F{k}/D{k}").number_format = "0.0"
    raw.auto_filter.ref = f"A1:O{last}"
    finish(raw, {"A": 36, "J": 12, "K": 12, "L": 12}, "D2")

    ws = wb.create_sheet("Long_Summary", 6)
    title(ws, "Long preset over time (Fig. 2)",
          "6 users, 60 sessions each over 540 days, 8 checkpoints; mean over 3 seeds of each seed's per-checkpoint value.")
    L = lambda c: rng(c, last, "Long_Checkpoints")  # noqa: E731
    systems = [LABEL[n] for n in LONG_SYSTEMS]
    refs = {}
    blocks = [("Answer accuracy % per checkpoint", "N"), ("Probes with forbidden value in evidence", "G"),
              ("Stale value in evidence", "H"), ("Store KiB", "L"), ("Injected tokens per probe", "J")]
    row = 4
    for name, col in blocks:
        ws.cell(row, 1, name).font = Font(name=FONT, bold=True, size=12)
        row += 1
        for i, h in enumerate(["System"] + [f"c{c}" for c in range(1, 9)] + ["First→last change"], 1):
            ws.cell(row, i, h)
        style_header(ws, row, 10)
        top = row
        for lab in systems:
            row += 1
            ws.cell(row, 1, lab)
            for c in range(1, 9):
                f = f"=AVERAGEIFS({L(col)},{L('A')},$A{row},{L('C')},{c})"
                if col == "L":
                    f = f"=AVERAGEIFS({L(col)},{L('A')},$A{row},{L('C')},{c})/1024"
                ws.cell(row, 1 + c, f).number_format = "0.0"
            ws.cell(row, 10, f"=I{row}-B{row}").number_format = "+0.0;-0.0;0.0"
            if col == "N":
                refs[lab] = str(row)
        if col == "N":
            ch = LineChart()
            ch.title = "Answer accuracy per checkpoint (%)"
            ch.y_axis.title = "% correct"
            ch.x_axis.title = "Checkpoint"
            data = Reference(ws, min_col=1, max_col=9, min_row=top + 1, max_row=row)
            ch.add_data(data, from_rows=True, titles_from_data=True)
            ch.set_categories(Reference(ws, min_col=2, max_col=9, min_row=top))
            ch.height, ch.width = 10, 22
            ws.add_chart(ch, "L4")
        row += 2
    note(ws, row, "Mem0 rows: runs/mem0/synthetic-long-s*. Restart and upgrade results: sheet Restart_Upgrade.")
    finish(ws, {"A": 24, **{get_column_letter(c): 9 for c in range(2, 11)}}, "B4")
    return refs


def restart_upgrade(wb: Workbook, restart: list[list], upgrade: list[list]) -> dict[str, str]:
    ws = wb.create_sheet("Restart_Upgrade", 7)
    title(ws, "Restarts and model upgrade (long preset)",
          "Restart: snapshot every store after each checkpoint, restore into a fresh instance, continue.")
    last = table_rows(ws, ["System", "Seed", "Probes", "Identical after restart", "Restarts",
                           "Largest snapshot KiB", "Slowest restore ms", "LLM calls"], restart, 4,
                      fmt=[None] * 5 + ["0.0", "0.0"])
    r = last + 1
    put(ws, r, ["Total", None, f"=SUM(C5:C{last})", f"=SUM(D5:D{last})", f"=SUM(E5:E{last})",
                f"=MAX(F5:F{last})", f"=MAX(G5:G{last})"], fmt=[None] * 5 + ["0.0", "0.0"])
    ws.cell(r, 1).font = Font(name=FONT, bold=True)
    refs = {"restart_total": str(r)}
    r += 3
    ws.cell(r - 1, 1, "Extraction model change after checkpoint 4 (LEDGER+EP augment, seeds 0-1)").font = \
        Font(name=FONT, bold=True, size=12)
    first = r + 1
    lastu = table_rows(ws, ["Variant", "Seed", "Probes c5-c8", "Answer correct c5-c8", "Extraction $ (whole run)"],
                       upgrade, r, fmt=[None] * 4 + ["0.000"])
    r = lastu + 2
    for i, h in enumerate(["Variant", "Probes", "Correct", "Second-half accuracy % (pooled)",
                           "Mean of per-checkpoint % (as in paper)", "Extraction $ (both seeds)",
                           "Cost vs gpt-4.1 throughout"], 1):
        ws.cell(r, i, h)
    style_header(ws, r, 7)
    variants = []
    for u in upgrade:
        if u[0] not in variants:
            variants.append(u[0])
    rng_ = lambda c: f"${c}${first}:${c}${lastu}"  # noqa: E731
    for k, v in enumerate(variants, r + 1):
        ws.cell(k, 1, v)
        ws.cell(k, 2, f"=SUMIFS({rng_('C')},{rng_('A')},$A{k})")
        ws.cell(k, 3, f"=SUMIFS({rng_('D')},{rng_('A')},$A{k})")
        ws.cell(k, 4, f"=100*C{k}/B{k}").number_format = "0.0"
        lc = lambda c: rng(c, 2000, "Long_Checkpoints")  # noqa: E731
        ws.cell(k, 5, f'=AVERAGEIFS({lc("N")},{lc("A")},"{UPGRADE_LONG[v]}",{lc("B")},"<=1",{lc("C")},">=5")'
                ).number_format = "0.0"
        ws.cell(k, 6, f"=SUMIFS({rng_('E')},{rng_('A')},$A{k})").number_format = "0.00"
        ws.cell(k, 7, f"=F{k}/$F${r + len(variants)}").number_format = "0%"
        refs[v] = str(k)
    finish(ws, {"A": 34, "B": 10, "C": 14, "D": 18, "E": 18, "F": 18, "G": 18}, "A5")
    return refs


def noise(wb: Workbook, rows: list[list]) -> dict[str, str]:
    ws = wb.create_sheet("Noise", 8)
    title(ws, "Run-to-run noise (uncached repeats)",
          "Three runs per seed with the response cache bypassed; a flip = the answer verdict differs across runs.")
    start = 30
    hdr = ["System", "Seed", "Probe", "Probe type", "Run 1 correct", "Run 2 correct", "Run 3 correct", "Flip"]
    last = table_rows(ws, hdr, [r + [None] for r in rows], start)
    for k in range(start + 1, last + 1):
        ws.cell(k, 8, f"=IF(AND(SUM(E{k}:G{k})>0,SUM(E{k}:G{k})<3),1,0)")
    R = lambda c: f"${c}${start + 1}:${c}${last}"  # noqa: E731
    for i, h in enumerate(["System", "Seed", "Run 1 %", "Run 2 %", "Run 3 %", "SD across runs (pts)"], 1):
        ws.cell(4, i, h)
    style_header(ws, 4, 6)
    row = 5
    refs = {}
    for lab in ("LEDGER (LLM)", "LEDGER+EP fallback"):
        top = row
        for s in range(5):
            ws.cell(row, 1, lab)
            ws.cell(row, 2, s)
            base = f"{R('A')},$A{row},{R('B')},$B{row}"
            for j, col in enumerate("EFG"):
                ws.cell(row, 3 + j, f"=100*SUMIFS({R(col)},{base})/COUNTIFS({base})").number_format = "0.0"
            ws.cell(row, 6, f"=STDEV(C{row}:E{row})").number_format = "0.00"
            row += 1
        refs[lab] = (top, row - 1)
    r = row + 1
    for i, h in enumerate(["System", "Probes", "Answer flips", "Mean per-seed SD (pts)", "Max per-seed SD (pts)"], 1):
        ws.cell(r, i, h)
    style_header(ws, r, 5)
    out = {}
    for k, lab in enumerate(refs, r + 1):
        top, bot = refs[lab]
        ws.cell(k, 1, lab)
        ws.cell(k, 2, f"=COUNTIFS({R('A')},$A{k})")
        ws.cell(k, 3, f"=SUMIFS({R('H')},{R('A')},$A{k})")
        ws.cell(k, 4, f"=AVERAGE(F{top}:F{bot})").number_format = "0.00"
        ws.cell(k, 5, f"=MAX(F{top}:F{bot})").number_format = "0.00"
        out[lab] = str(k)
    ws.cell(start - 1, 1, "Per-probe verdicts").font = Font(name=FONT, bold=True, size=12)
    finish(ws, {"A": 24, "B": 8, "C": 10, "D": 14, "E": 12, "F": 14, "G": 12, "H": 8}, None)
    return out


def judge_audit(wb: Workbook, rows: list[list]) -> dict[str, str]:
    ws = wb.create_sheet("Judge_Audit", 9)
    title(ws, "Human audit of LLM-judge decisions (LongMemEval)",
          "60 blinded items, 20 per system; a human labelled each response without seeing the judge's verdict.")
    start = 16
    last = table_rows(ws, ["Item", "System", "Question id", "Category", "Judge correct", "Human correct",
                           "Human note", "Question", "Gold answer", "Response", "Agree"],
                      [r + [None] for r in rows], start)
    for k in range(start + 1, last + 1):
        ws.cell(k, 11, f"=IF(E{k}=F{k},1,0)")
    E, F, K = (f"${c}${start + 1}:${c}${last}" for c in "EFK")
    stats = [
        ("Items", f"=COUNTA({E})", "0"),
        ("Agreement", f"=SUM({K})/B5", "0.0%"),
        ("Judge yes, human no", f'=COUNTIFS({E},"yes",{F},"no")', "0"),
        ("Judge no, human yes", f'=COUNTIFS({E},"no",{F},"yes")', "0"),
        ("Judge yes rate", f'=COUNTIF({E},"yes")/B5', "0.0%"),
        ("Human yes rate", f'=COUNTIF({F},"yes")/B5', "0.0%"),
        ("Chance agreement", "=B9*B10+(1-B9)*(1-B10)", "0.000"),
        ("Cohen's kappa", "=(B6-B11)/(1-B11)", "0.00"),
    ]
    for i, (lab, f, fmt) in enumerate(stats, 5):
        ws.cell(i, 1, lab).font = Font(name=FONT, bold=True)
        ws.cell(i, 2, f).number_format = fmt
    note(ws, 14, "Row 12 of the human file duplicated row 11; its label was copied (see runs/longmemeval/audit). "
                 "Mem0's judge decisions were added later and not audited.")
    finish(ws, {"A": 22, "B": 20, "C": 12, "D": 22, "E": 10, "F": 10, "G": 24, "H": 50, "I": 24, "J": 50}, None)
    return {"kappa": "12", "agree": "6"}


def static_sheets(wb: Workbook) -> None:
    ws = wb.create_sheet("Workloads", 10)
    title(ws, "Workloads (Table 2)", "Ranges over seeds; tokens counted with the runner's tokenizer over all events.")
    table_rows(ws, ["", "Small (s0-4)", "Long (s0-2)", "LongMemEval"], [
        ["Users (histories)", 3, 6, 210], ["Sessions", 48, 360, "≈48 each"],
        ["Events", "184-218", "1,151-1,203", "≈493 each"], ["History tokens", "2.0k-2.6k", "12.4k-13.2k", "96k-106k each"],
        ["Facts / mutations", "22-34 / 63-105", "60-81 / 242-329", "--"], ["Deletions", "4-7", "20-32", "--"],
        ["Checkpoints", 4, 8, "1 per history"], ["Probes", "=COUNTIFS(Coding_Probes!$A:$A,\"dev\",Coding_Probes!$C:$C,\"No memory\")",
                                                  "=SUMIFS(Long_Checkpoints!$D:$D,Long_Checkpoints!$A:$A,\"No memory\")", 210],
    ], 4)
    note(ws, 14, "Copied from the paper's Table 2 (workload generator statistics); the Probes row is recomputed from the logs.")
    finish(ws, {"A": 22, "B": 18, "C": 18, "D": 18})

    ws = wb.create_sheet("Ablations", 11)
    title(ws, "What changed LEDGER's evidence accuracy (Table 7)",
          "Development sweeps on dev seeds 0-4 (python -m runner.sweep); values copied from the sweep logs at the time.")
    table_rows(ws, ["Change", "Before", "After", "Kept?"], [
        ["Enumerate event ids in schema", "88.7 ± 4.0", "92.2 ± 4.1", "kept"],
        ["LLM slot-merge check", "92.2", "92.0", "removed"],
        ["Terse-value extraction prompt", "91.9", "90.7", "removed"],
        ["Neutral router examples", "91.9", "91.5", "kept"],
        ["Hold assistant-first values (trust 2)", "6 wrong", "4 wrong", "kept"],
        ["Hold document-first values too (trust 3)", "91.5", "89.7", "rejected"],
        ["Named-slot deletion (description only)", "2 leaked", "0 leaked", "kept"],
        ["Grounding check", "1 no-abstention", "0", "kept"],
    ], 4)
    note(ws, 14, "The two trust rows replay every extraction from cache, so their difference is exact.")
    finish(ws, {"A": 42, "B": 16, "C": 16, "D": 10})

    ws = wb.create_sheet("Requirements", 12)
    title(ws, "Challenge requirements and where the paper addresses them (Table 1)")
    table_rows(ws, ["Requirement", "Addressed by"], [
        ["R1 lifecycle, provenance, deletion, abstention", "Ledger rules; evidence carries source events and validity; tombstones; abstention reasons"],
        ["R1 persistence, restarts", "Restart after every checkpoint changes no probe (sheet Restart_Upgrade)"],
        ["R2 established benchmark", "LongMemEval_S, stratified subset of 210 questions (sheet LME_Results)"],
        ["R2 multi-session agent workload", "Generated coding-agent timelines, 5 + 5 held-out seeds (sheet Coding_Results)"],
        ["R2 same model, prompts, budgets", "Shared answerer, one token budget, one tokenizer"],
        ["R2 quality, tokens, $, ingest, latency, storage", "Measured by the runner, never self-reported (sheet Efficiency)"],
        ["R3 checkpoints, degradation, staleness", "8 checkpoints over 540 days; typed errors, store size, latency (sheet Long_Summary)"],
        ["R3 pinned, repeated, audited", "Pinned snapshots, response cache, seeds × repeats (Noise); 60 judge decisions audited (Judge_Audit)"],
    ], 4, raw=False)
    finish(ws, {"A": 46, "B": 100})


def summary(wb: Workbook, cr: dict, dl: dict, ef: dict, lm: dict, lg: dict, ru: dict, nz: dict, ja: dict) -> None:
    ws = wb.create_sheet("Summary", 1)
    title(ws, "Numbers quoted in the paper, recomputed",
          "Paper value is as printed (input); Workbook value is a formula over the raw sheets; Check flags a rounding mismatch.")
    d, h = cr["dev"], cr["held-out"]
    C = "Coding_Results"
    claims = [
        ("Coding", "LEDGER+EP fallback answer accuracy, dev", 94.3, f"={C}!G{d['hybrid_llm_fb']}", 1),
        ("Coding", "LEDGER+EP augment answer accuracy, dev", 94.4, f"={C}!G{d['hybrid_llm_aug']}", 1),
        ("Coding", "LEDGER (LLM) answer accuracy, dev", 90.9, f"={C}!G{d['ledger_llm']}", 1),
        ("Coding", "LEDGER (LLM) evidence accuracy, dev", 91.7, f"={C}!N{d['ledger_llm']}", 1),
        ("Coding", "Full replay answer accuracy, dev", 82.1, f"={C}!G{d['full_replay']}", 1),
        ("Coding", "Full replay wrong answers (stale/guesses), dev", 65, f"={C}!Q{d['full_replay']}", 0),
        ("Coding", "Full replay clean evidence, dev", 21.9, f"={C}!N{d['full_replay']}", 1),
        ("Coding", "Transcript RAG answer accuracy, dev", 74.8, f"={C}!G{d['transcript_rag']}", 1),
        ("Coding", "Mem0 answer accuracy, dev", 66.8, f"={C}!G{d['mem0']}", 1),
        ("Coding", "Mem0 (coding prompt) answer accuracy, dev", 70.6, f"={C}!G{d['mem0_coding']}", 1),
        ("Coding", "No memory answer accuracy, dev", 29.5, f"={C}!G{d['no_memory']}", 1),
        ("Held-out", "LEDGER+EP fallback answer accuracy", 91.1, f"={C}!G{h['hybrid_llm_fb']}", 1),
        ("Held-out", "LEDGER+EP augment answer accuracy", 91.8, f"={C}!G{h['hybrid_llm_aug']}", 1),
        ("Held-out", "LEDGER (LLM) answer accuracy", 88.2, f"={C}!G{h['ledger_llm']}", 1),
        ("Held-out", "Full replay answer accuracy", 83.3, f"={C}!G{h['full_replay']}", 1),
        ("Held-out", "Transcript RAG answer accuracy", 78.5, f"={C}!G{h['transcript_rag']}", 1),
        ("Held-out", "Mem0 answer accuracy", 66.2, f"={C}!G{h['mem0']}", 1),
        ("Deletion", "Full replay: deleted value sent to model (probes)", 46, f"=Deletion!C{dl['Full replay']}", 0),
        ("Deletion", "Full replay: deleted value repeated in answer", 4, f"=Deletion!D{dl['Full replay']}", 0),
        ("Deletion", "Transcript RAG: deleted value sent to model", 39, f"=Deletion!C{dl['Transcript RAG']}", 0),
        ("Deletion", "Mem0: deleted value repeated in answer", 17, f"=Deletion!D{dl['Mem0']}", 0),
        ("Deletion", "LEDGER+EP fallback: sent / repeated", 0, f"=Deletion!C{dl['LEDGER+EP fallback']}+Deletion!D{dl['LEDGER+EP fallback']}", 0),
        ("Deletion", "LEDGER+EP augment: sent / repeated", 0, f"=Deletion!C{dl['LEDGER+EP augment']}+Deletion!D{dl['LEDGER+EP augment']}", 0),
        ("Efficiency", "Tokens per probe, full replay", 982, f"=Efficiency!F{ef['Full replay']}", 0),
        ("Efficiency", "Tokens per probe, transcript RAG", 120, f"=Efficiency!F{ef['Transcript RAG']}", 0),
        ("Efficiency", "Tokens per probe, LEDGER (LLM)", 15, f"=Efficiency!F{ef['LEDGER (LLM)']}", 0),
        ("Efficiency", "Tokens per probe, LEDGER+EP fallback", 56, f"=Efficiency!F{ef['LEDGER+EP fallback']}", 0),
        ("Efficiency", "Total m$ per probe, LEDGER+EP fallback", 0.45, f"=Efficiency!I{ef['LEDGER+EP fallback']}", 2),
        ("Efficiency", "Total m$ per probe, full replay", 0.55, f"=Efficiency!I{ef['Full replay']}", 2),
        ("Efficiency", "Median routing s, LLM systems pooled", 0.66, f"=Efficiency!B{ef['routing']}/1000", 2),
        ("Efficiency", "95th percentile routing s, LLM systems pooled", 0.96, f"=Efficiency!B{int(ef['routing']) + 1}/1000", 2),
        ("LongMemEval", "LEDGER+EP augment accuracy", 74.8, f"=LME_Results!C{lm['LEDGER+EP augment']}", 1),
        ("LongMemEval", "Full replay accuracy", 74.8, f"=LME_Results!C{lm['Full replay']}", 1),
        ("LongMemEval", "Transcript RAG accuracy", 71.4, f"=LME_Results!C{lm['Transcript RAG']}", 1),
        ("LongMemEval", "Mem0 accuracy", 71.0, f"=LME_Results!C{lm['Mem0']}", 1),
        ("LongMemEval", "LEDGER+EP fallback accuracy", 70.5, f"=LME_Results!C{lm['LEDGER+EP fallback']}", 1),
        ("LongMemEval", "LEDGER (LLM) accuracy", 41.0, f"=LME_Results!C{lm['LEDGER (LLM)']}", 1),
        ("LongMemEval", "Tokens, LEDGER+EP augment", 1094, f"=LME_Results!M{lm['LEDGER+EP augment']}", 0),
        ("LongMemEval", "Tokens, full replay", 108028, f"=LME_Results!M{lm['Full replay']}", 0),
        ("LongMemEval", "Context of LEDGER+EP augment vs full replay (%)", 1,
         f"=100*LME_Results!M{lm['LEDGER+EP augment']}/LME_Results!M{lm['Full replay']}", 0),
        ("Long", "Full replay answer accuracy, c1", 89.4, f"=Long_Summary!B{lg['Full replay']}", 1),
        ("Long", "Full replay answer accuracy, c8", 70.5, f"=Long_Summary!I{lg['Full replay']}", 1),
        ("Long", "Transcript RAG answer accuracy, c1", 88.0, f"=Long_Summary!B{lg['Transcript RAG']}", 1),
        ("Long", "Transcript RAG answer accuracy, c8", 66.5, f"=Long_Summary!I{lg['Transcript RAG']}", 1),
        ("Long", "Mem0 answer accuracy, c1", 78.3, f"=Long_Summary!B{lg['Mem0']}", 1),
        ("Long", "LEDGER+EP fallback, lowest checkpoint", 89.7, f"=MIN(Long_Summary!B{lg['LEDGER+EP fallback']}:I{lg['LEDGER+EP fallback']})", 1),
        ("Long", "LEDGER+EP fallback, highest checkpoint", 96.2, f"=MAX(Long_Summary!B{lg['LEDGER+EP fallback']}:I{lg['LEDGER+EP fallback']})", 1),
        ("Restart", "Probes per system over 3 seeds", 1379,
         f'=SUMIFS(Restart_Upgrade!C5:C{int(ru["restart_total"]) - 1},Restart_Upgrade!A5:A{int(ru["restart_total"]) - 1},"Full replay")', 0),
        ("Restart", "Probes changed by a restart (all systems)", 0,
         f"=Restart_Upgrade!C{ru['restart_total']}-Restart_Upgrade!D{ru['restart_total']}", 0),
        ("Restart", "Slowest restore (ms, ≤)", 63, f"=Restart_Upgrade!G{ru['restart_total']}", 0),
        ("Upgrade", "Second half, mini throughout (%)", 92.0, f"=Restart_Upgrade!E{ru['gpt-4.1-mini throughout']}", 1),
        ("Upgrade", "Second half, upgraded midway (%)", 95.4, f"=Restart_Upgrade!E{ru['mini, upgraded to gpt-4.1 after c4']}", 1),
        ("Upgrade", "Second half, gpt-4.1 throughout (%)", 95.7, f"=Restart_Upgrade!E{ru['gpt-4.1 throughout']}", 1),
        ("Upgrade", "Extraction $, upgraded (both seeds)", 1.18, f"=Restart_Upgrade!F{ru['mini, upgraded to gpt-4.1 after c4']}", 2),
        ("Upgrade", "Extraction $, gpt-4.1 throughout (both seeds)", 1.89, f"=Restart_Upgrade!F{ru['gpt-4.1 throughout']}", 2),
        ("Noise", "Answer flips, LEDGER (LLM), of 512", 12, f"=Noise!C{nz['LEDGER (LLM)']}", 0),
        ("Noise", "Answer flips, LEDGER+EP fallback, of 512", 8, f"=Noise!C{nz['LEDGER+EP fallback']}", 0),
        ("Noise", "Mean per-seed SD, LEDGER (LLM)", 0.7, f"=Noise!D{nz['LEDGER (LLM)']}", 1),
        ("Noise", "Mean per-seed SD, LEDGER+EP fallback", 0.4, f"=Noise!D{nz['LEDGER+EP fallback']}", 1),
        ("Judge audit", "Cohen's kappa", 0.97, f"=Judge_Audit!B{ja['kappa']}", 2),
        ("Judge audit", "Agreement (items of 60)", 59, f"=Judge_Audit!B{ja['agree']}*60", 0),
    ]
    hdr = ["Area", "Claim", "Paper value", "Workbook value", "Decimals in paper", "Check"]
    for i, x in enumerate(hdr, 1):
        ws.cell(4, i, x)
    style_header(ws, 4, len(hdr))
    for k, (area, claim, paper, formula, dec) in enumerate(claims, 5):
        ws.cell(k, 1, area)
        ws.cell(k, 2, claim)
        c = ws.cell(k, 3, paper)
        c.font = Font(name=FONT, color=BLUE)
        ws.cell(k, 4, formula).number_format = "#,##0.00"
        ws.cell(k, 5, dec).font = Font(name=FONT, color=BLUE)
        ws.cell(k, 6, f'=IF(ROUND(D{k},E{k})=C{k},"ok","CHECK")')
    last = 4 + len(claims)
    ws.cell(last + 2, 1, "Claims flagged").font = Font(name=FONT, bold=True)
    ws.cell(last + 2, 2, f'=COUNTIF(F5:F{last},"CHECK")')
    from openpyxl.formatting.rule import CellIsRule
    ws.conditional_formatting.add(f"F5:F{last}", CellIsRule(operator="equal", formula=['"CHECK"'],
                                                            fill=PatternFill("solid", fgColor="F8CBAD")))
    note(ws, last + 4, "Decimals = how the paper rounds the value (-1 = to tens). Long-preset values are means over "
                       "3 seeds of per-checkpoint accuracy, as in runner/longitudinal.py.")
    finish(ws, {"A": 14, "B": 54, "C": 13, "D": 15, "E": 10, "F": 9}, "A5")


def main() -> None:
    wb = Workbook()
    readme(wb)
    probes, runs = coding_probes()
    last, lastr = coding_raw(wb, probes, runs)
    cr = coding_results(wb, last, lastr)
    coding_by_type(wb, last)
    dl = deletion(wb, deletion_rows())
    ef = efficiency(wb, efficiency_inputs(), last, lastr)
    lrows, ci = lme_rows()
    lm = lme(wb, lrows, ci)
    lg = long_sheets(wb, long_rows())
    ru = restart_upgrade(wb, restart_rows(), upgrade_rows())
    nz = noise(wb, noise_rows())
    ja = judge_audit(wb, audit_rows())
    static_sheets(wb)
    summary(wb, cr, dl, ef, lm, lg, ru, nz, ja)
    order = ["README", "Summary", "Coding_Results", "Coding_ByType", "Deletion", "Efficiency", "LME_Results",
             "Long_Summary", "Restart_Upgrade", "Noise", "Judge_Audit", "Workloads", "Ablations", "Requirements",
             "Coding_Probes", "Coding_Runs", "LME_Questions", "Long_Checkpoints"]
    wb._sheets = [wb[n] for n in order]
    wb.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
