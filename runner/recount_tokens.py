"""Recount injected tokens of logged runs with the model tokenizer.

    python -m runner.recount_tokens            # prints the check and the counts

The logged runs counted tokens with ``regex-words/1`` (words and punctuation
marks), and systems used that count to fit evidence into the budget (2,000
tokens on the coding workload, 4,000 on LongMemEval). This script checks
that the budget never cut any evidence, so a model-token budget would have
selected the same items, and recounts every logged evidence set with
``o200k_base``, the tokenizer of ``gpt-4.1-mini`` (``interface.tokens``):

* no probe's or question's logged evidence exceeds the budget in model
  tokens, and the runner never truncated it (``evidence_tokens`` equals
  ``injected_tokens``);
* on LongMemEval, where single turns are long, every retriever returned its
  full top-k (BM25 5 turns, Mem0 10 memories), so no item was dropped inside
  a system either. On the coding workload the largest bounded evidence set is
  about a tenth of the budget, so nothing could have been dropped.

Full replay is unbounded; on LongMemEval its evidence is not logged, so its
injected tokens are the provider-reported input tokens of its answering call
minus those of the no-memory call for the same question (same prompt with
``(nothing)`` as memory).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from interface.tokens import count_tokens

RUNS = Path("runs")
TOP_K = {"transcript_rag": 5, "mem0": 10}


def evidence_tokens(texts) -> int:
    return sum(count_tokens(t) for t in texts)


def check_synthetic() -> list[str]:
    problems = []
    for path in sorted(RUNS.glob("**/synthetic-*/*.json")):
        log = json.loads(path.read_text(encoding="utf-8"))
        budget = log["settings"]["token_budget"]
        for p in log["probes"]:
            n = evidence_tokens(e["content"] for e in p["evidence"])
            if p["budget_exceeded"] or (budget is not None and n > budget // 2):
                problems.append(f"{path}: {p['probe_id']} ({n} tokens)")
    return problems


def check_lme() -> list[str]:
    problems = []
    for path in sorted((RUNS / "longmemeval" / "main").glob("*.json")):
        log = json.loads(path.read_text(encoding="utf-8"))
        for r in log["instances"]:
            if r.get("evidence") is None:
                continue
            n = evidence_tokens(r["evidence"])
            k = TOP_K.get(log["system"])
            if n > log["budget"] or r["evidence_tokens"] > r["injected_tokens"] or (k and r["evidence_items"] < k):
                problems.append(f"{path}: {r['question_id']} ({n} tokens, {r['evidence_items']} items)")
    return problems


def lme_full_replay_tokens() -> dict[str, int]:
    """Question id -> injected model tokens of full replay on LongMemEval."""
    main = RUNS / "longmemeval" / "main"
    full = {r["question_id"]: r for r in json.loads((main / "full_replay.json").read_text())["instances"]}
    none = {r["question_id"]: r for r in json.loads((main / "no_memory.json").read_text())["instances"]}
    return {q: full[q]["answer_usage"]["llm_input_tokens"] - none[q]["answer_usage"]["llm_input_tokens"]
            + count_tokens("(nothing)") for q in full}


def main() -> None:
    syn, lme = check_synthetic(), check_lme()
    print(f"synthetic logs: {len(syn)} probe(s) near or over the budget")
    print(f"LongMemEval logs: {len(lme)} question(s) truncated or over the budget")
    for line in (syn + lme)[:20]:
        print("  ", line)
    for path in sorted((RUNS / "longmemeval" / "main").glob("*.json")):
        log = json.loads(path.read_text(encoding="utf-8"))
        rows = [r for r in log["instances"] if "error" not in r]
        if rows[0].get("evidence") is None:
            mean = statistics.mean(lme_full_replay_tokens().values())
        else:
            mean = statistics.mean(evidence_tokens(r["evidence"]) for r in rows)
        old = statistics.mean(r["injected_tokens"] for r in rows)
        print(f"LongMemEval {log['system']:26s} regex {old:9.1f}  o200k {mean:9.1f}")


if __name__ == "__main__":
    main()
