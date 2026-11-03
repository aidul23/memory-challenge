"""Run memory systems on LongMemEval: replay, answer with the shared model,
judge with the benchmark's own judge.

    python -m runner.longmemeval --make-subset pilot --per-category 2
    python -m runner.longmemeval --subset pilot --systems no_memory transcript_rag full_replay
    python -m runner.longmemeval --subset pilot --systems ledger_llm --workers 8

For each instance and system: a fresh store ingests the history session by
session in date order, the question is asked at its date, the (budget-
truncated) evidence goes to the shared answerer, and the answer is judged
with LongMemEval's prompts (``evaluate_qa.py``, MIT licence, copied verbatim
below) on the benchmark's judge model, ``gpt-4o-2024-08-06``, at
temperature 0. The one change is the response format: the judge returns a
JSON ``{"answer": "yes" | "no"}`` instead of free text, so the label is read
from a field rather than by searching for "yes".

Every answer, judge decision, token count, latency and cost is logged to
``runs/longmemeval/<subset>/<system>.json``; all LLM responses are cached, so
a rerun is free. Instances run in parallel threads (``--workers``), each with
its own system instance; they share nothing but the response cache, and no two
instances send the same prompt.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

from interface.memory_system import MemorySystem, Query, Usage
from interface.tokens import count_tokens
from runner.__main__ import SYSTEMS
from runner.prices import PRICES_AS_OF, cost_usd
from runner.replay import _fit, sum_usage
from systems.ledger.backends import ROOT, OpenAIBackend, backend_from_env
from systems.ledger.llm import LLMExtractor, cached_json
from systems.hybrid.system import LedgerHybrid
from systems.ledger.system import Ledger
from workloads import longmemeval as lme

JUDGE_MODEL = "gpt-4o-2024-08-06"
CACHE = ROOT / ".cache"

#: LongMemEval systems: the synthetic ones, with the ledger's extractor on its
#: personal-chat profile.
LME_SYSTEMS: dict[str, tuple[Callable[[], MemorySystem], bool]] = {
    **{n: SYSTEMS[n] for n in ("no_memory", "full_replay", "transcript_rag")},
    "ledger_llm": (lambda: Ledger(LLMExtractor(profile="personal")), False),
    "hybrid_llm": (lambda: LedgerHybrid(LLMExtractor(profile="personal")), False),
    "hybrid_llm_augment": (lambda: LedgerHybrid(LLMExtractor(profile="personal"), mode="augment"), False),
    # Extract from the user's turns only (87% of LongMemEval's tokens are the
    # assistant's, whose claims the ledger would only hold or dispute).
    "ledger_llm_user": (lambda: Ledger(LLMExtractor(profile="personal"), extract_min_trust=2), False),
    "hybrid_llm_user_fallback": (lambda: LedgerHybrid(LLMExtractor(profile="personal"), mode="fallback",
                                                      extract_min_trust=2), False),
    "hybrid_llm_user": (lambda: LedgerHybrid(LLMExtractor(profile="personal"), mode="augment",
                                             extract_min_trust=2), False),
}

ANSWER_SYSTEM = """\
You are a chat assistant with memory of your past conversations with this
user. Answer the user's question, given after the memory, using the memory.
Memory items may carry dates; use them and today's date for questions about
time, order or duration, and when statements changed over time, answer with
the latest. If the memory does not contain the information the question
needs, say that you do not know or that it was not mentioned. Answer
concisely.
"""

ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
                 "required": ["answer"], "additionalProperties": False}

JUDGE_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
                "required": ["answer"], "additionalProperties": False}


def judge_prompt(task: str, question: str, answer: str, response: str, abstention: bool) -> str:
    """``get_anscheck_prompt`` from LongMemEval's ``evaluate_qa.py``, verbatim."""
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        else:
            raise NotImplementedError(task)
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    return template.format(question, answer, response)


def answer_prompt(question: str, today: datetime, evidence: list[str]) -> str:
    # Memory first and the question last: after ~100k tokens of chat logs, a
    # question placed first is lost and the model continues the last chat.
    memory = "\n\n".join(evidence) if evidence else "(nothing)"
    return (f"Memory:\n{memory}\n\n---\nToday is {today:%Y-%m-%d (%a) %H:%M}.\n"
            f"Question: {question}\nAnswer the question above using the memory.")


def run_instance(name: str, inst: lme.Instance, budget: int, answer_backend, judge_backend) -> dict[str, Any]:
    """One instance end to end; an exception is recorded, not raised, so one
    bad instance does not stop a paid run (completed calls are cached)."""
    try:
        return _run_instance(name, inst, budget, answer_backend, judge_backend)
    except Exception as exc:  # noqa: BLE001 - logged and reported per instance
        return {"question_id": inst.question_id, "category": inst.category, "error": repr(exc)[:500]}


def _run_instance(name: str, inst: lme.Instance, budget: int, answer_backend, judge_backend) -> dict[str, Any]:
    factory, unbounded = LME_SYSTEMS[name]
    system = factory()
    system.reset()
    ingest_s, ingest_usage = 0.0, []
    sessions: dict[str, list] = defaultdict(list)
    for e in inst.events:
        sessions[e.session_id].append(e)
    for batch in sessions.values():  # insertion order = date order
        t0 = time.perf_counter()
        report = system.ingest(batch)
        ingest_s += time.perf_counter() - t0
        ingest_usage.append(report.usage)

    token_budget = None if unbounded else budget
    t0 = time.perf_counter()
    res = system.retrieve(Query(query_id=inst.question_id, user_id=inst.question_id, text=inst.question,
                                asked_at=inst.question_date, token_budget=token_budget))
    retrieve_s = time.perf_counter() - t0
    evidence, total, used = _fit(res.evidence, token_budget)
    texts = [ev.content for ev in evidence]

    data, ans_usage, _ = cached_json(answer_backend, CACHE / "lme-answer", ANSWER_SYSTEM,
                                     answer_prompt(inst.question, inst.question_date, texts),
                                     ANSWER_SCHEMA, "answer")
    response = data.get("answer", "") if isinstance(data.get("answer"), str) else ""
    verdict, judge_usage, _ = cached_json(
        judge_backend, CACHE / "lme-judge", "",
        judge_prompt(inst.question_type, inst.question, inst.answer, response, inst.abstention),
        JUDGE_SCHEMA, "judgement")
    stats = system.stats(inst.question_id)
    memory_usage = sum_usage(ingest_usage + [res.usage])
    return {
        "question_id": inst.question_id, "category": inst.category, "question_type": inst.question_type,
        "question": inst.question, "gold": inst.answer, "response": response,
        "correct": verdict.get("answer") == "yes",
        "abstained": res.abstained, "abstain_reason": res.abstain_reason,
        "evidence": texts if not unbounded else None,
        "evidence_items": len(evidence), "evidence_tokens": total, "injected_tokens": used,
        "history_tokens": sum(count_tokens(e.content) for e in inst.events),
        "sessions": inst.sessions, "ingest_s": ingest_s, "retrieve_s": retrieve_s,
        "store_items": stats.n_items, "storage_bytes": stats.storage_bytes,
        "memory_usage": memory_usage.model_dump(), "answer_usage": ans_usage.model_dump(),
        "judge_usage": judge_usage.model_dump(),
        "cost_usd": {"memory": cost_usd(memory_usage), "answer": cost_usd(ans_usage),
                     "judge": cost_usd(judge_usage)},
        "system_config": system.config(),
    }


def report(name: str, all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in all_rows if "error" not in r]
    cats = sorted({r["category"] for r in rows})
    by = {c: [r for r in rows if r["category"] == c] for c in cats}
    acc = {c: sum(r["correct"] for r in rs) / len(rs) for c, rs in by.items()}
    cost = {k: sum(r["cost_usd"][k] for r in rows) for k in ("memory", "answer", "judge")}
    out = {
        "system": name, "questions": len(rows), "errors": len(all_rows) - len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "accuracy_by_category": acc, "n_by_category": {c: len(rs) for c, rs in by.items()},
        "mean_injected_tokens": statistics.mean(r["injected_tokens"] for r in rows),
        "mean_ingest_s": statistics.mean(r["ingest_s"] for r in rows),
        "mean_retrieve_ms": 1000 * statistics.mean(r["retrieve_s"] for r in rows),
        "mean_storage_bytes": statistics.mean(r["storage_bytes"] for r in rows),
        "cost_usd": cost,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=lme.DEFAULT_DATA)
    ap.add_argument("--make-subset", metavar="NAME", help="draw and save a stratified subset, then exit")
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", default="pilot")
    ap.add_argument("--systems", nargs="+", choices=sorted(LME_SYSTEMS), default=["no_memory", "transcript_rag"])
    ap.add_argument("--budget", type=int, default=4000, help="evidence token budget per question")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/longmemeval")
    args = ap.parse_args()

    if args.make_subset:
        ids = lme.stratified_sample(args.data, args.per_category, args.seed)
        print(f"{len(ids)} questions -> {lme.save_subset(ids, args.make_subset)}")
        return

    instances = lme.load(args.data, lme.load_subset(args.subset))
    answer_backend = backend_from_env()
    judge_backend = OpenAIBackend(JUDGE_MODEL)
    from pathlib import Path
    out_dir = Path(args.out) / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in args.systems:
        started = datetime.now(timezone.utc)
        with ThreadPoolExecutor(args.workers) as pool:
            rows = list(pool.map(lambda i: run_instance(name, i, args.budget, answer_backend, judge_backend),
                                 instances))
        s = report(name, rows)
        summaries.append(s)
        log = {"benchmark": "LongMemEval_S (cleaned)", "subset": args.subset, "system": name,
               "started_at": started.isoformat(), "budget": args.budget, "prices_as_of": PRICES_AS_OF,
               "answer_backend": answer_backend.config(), "judge_backend": judge_backend.config(),
               "answer_prompt_system": ANSWER_SYSTEM, "summary": s, "instances": rows}
        (out_dir / f"{name}.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
        if s["errors"]:
            print(f"{name}: {s['errors']} instance(s) failed; see 'error' in the log", flush=True)
        print(f"{name}: {s['accuracy']:.1%} of {s['questions']}, cost ${sum(s['cost_usd'].values()):.3f} "
              f"(memory {s['cost_usd']['memory']:.3f}, answer {s['cost_usd']['answer']:.3f}, "
              f"judge {s['cost_usd']['judge']:.3f}), {s['mean_injected_tokens']:.0f} tokens injected", flush=True)

    cats = sorted({c for s in summaries for c in s["accuracy_by_category"]})
    print("\n" + "  ".join(["system".ljust(15)] + [c[:14].ljust(14) for c in cats]))
    for s in summaries:
        print("  ".join([s["system"].ljust(15)] + [f"{s['accuracy_by_category'].get(c, 0):.0%}".ljust(14) for c in cats]))


if __name__ == "__main__":
    main()
