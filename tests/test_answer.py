"""The shared answerer and dollar costs: prompt contents, structured
abstention, caching, answer-level scoring in a replay, and the price table.
No network: the backend is scripted."""

from datetime import datetime, timezone

import pytest

from interface.memory_system import Evidence, Usage
from runner.answer import Answerer, answer_prompt
from runner.prices import cost_usd, price_of
from runner.replay import run
from runner.scoring import Verdict
from systems.full_replay.system import FullReplay
from systems.ledger.backends import JSONResult
from systems.ledger.system import Ledger
from systems.no_memory.system import NoMemory
from workloads.examples.build_example import build

T = datetime(2026, 3, 1, 12, tzinfo=timezone.utc)


class ScriptedBackend:
    """Answers with the first evidence line's value, or abstains on empty memory."""

    provider, model = "fake", "gpt-4.1-mini"

    def __init__(self):
        self.prompts: list[str] = []

    def complete_json(self, system, prompt, schema, name):
        self.prompts.append(prompt)
        memory = prompt.split("Memory:\n", 1)[1]
        usage = Usage(llm_calls=1, llm_input_tokens=100, llm_output_tokens=5,
                      model_ids=("gpt-4.1-mini-2025-04-14",))
        if memory.startswith("(nothing)"):
            return JSONResult({"answer": "", "abstain": True}, usage)
        line = memory.splitlines()[0]
        value = line.split(": ", 1)[1].split(" (since")[0] if ": " in line else line
        return JSONResult({"answer": value, "abstain": False}, usage)

    def config(self):
        return {"provider": self.provider, "model": self.model}


def test_prompt_carries_date_question_evidence_and_abstain_reason():
    p = answer_prompt("Which database?", T, [Evidence(content="database: PostgreSQL 16")])
    assert p.startswith("Today is 2026-03-01.") and "Question: Which database?" in p
    assert p.endswith("Memory:\n- database: PostgreSQL 16")
    assert answer_prompt("q", T, [], "deleted at the user's request").endswith(
        "(nothing) The memory system reported: deleted at the user's request.")


def test_answer_is_structured_and_cached(tmp_path):
    b = ScriptedBackend()
    a = Answerer(b, cache_dir=tmp_path)
    ans, usage = a.answer("Which database?", T, [Evidence(content="database: PostgreSQL 16 (since 2026-02-01)")])
    assert (ans.text, ans.abstained, usage.llm_calls) == ("PostgreSQL 16", False, 1)
    again, _ = Answerer(b, cache_dir=tmp_path).answer(
        "Which database?", T, [Evidence(content="database: PostgreSQL 16 (since 2026-02-01)")])
    assert again == ans and len(b.prompts) == 1  # served from the cache
    none, _ = a.answer("Which database?", T, [])
    assert none.abstained and none.text == ""
    assert a.config()["calls"] == 2 and a.config()["model"] == "gpt-4.1-mini"


@pytest.mark.parametrize("data", [{"answer": "  ", "abstain": False}, {"answer": "x", "abstain": True}, {}])
def test_blank_or_abstaining_answer_counts_as_abstention(tmp_path, data):
    class Fixed(ScriptedBackend):
        def complete_json(self, *args):
            return JSONResult(data, Usage())
    ans, _ = Answerer(Fixed(), cache_dir=tmp_path).answer("q", T, [Evidence(content="x")])
    assert ans.abstained and ans.text == ""


def test_replay_scores_answers_and_reports_cost(tmp_path):
    tl = build()
    ledger = run(Ledger(), tl, answerer=Answerer(ScriptedBackend(), cache_dir=tmp_path))
    assert all(p.answer_verdict == p.verdict == Verdict.CORRECT for p in ledger.probes)
    s = ledger.summary()
    assert s["answer_correct"] == 8 and s["memory_cost_usd"] == 0.0
    assert s["answer_cost_usd"] == pytest.approx(8 * (100 * 0.40 + 5 * 1.60) / 1e6)
    assert ledger.settings.answerer["model"] == "gpt-4.1-mini"

    # With no memory the answerer is still asked, and abstains everywhere.
    none = run(NoMemory(), tl, answerer=Answerer(ScriptedBackend(), cache_dir=tmp_path))
    assert {p.answer_verdict for p in none.probes} <= {Verdict.FALSE_ABSTENTION, Verdict.CORRECT}
    # Full replay's first line is the oldest event: answering from it is stale.
    full = run(FullReplay(), tl, token_budget=None, answerer=Answerer(ScriptedBackend(), cache_dir=tmp_path))
    assert any(p.answer_verdict != Verdict.CORRECT for p in full.probes)
    # Without an answerer nothing changes in the log.
    assert run(Ledger(), tl).probes[0].answer_verdict is None


def test_prices():
    assert price_of("gpt-4.1-mini-2025-04-14") == (0.40, 1.60)
    assert cost_usd(Usage()) == 0.0
    assert cost_usd(Usage(llm_input_tokens=1_000_000, llm_output_tokens=1_000_000,
                          model_ids=("gpt-4.1-mini-2025-04-14",))) == pytest.approx(2.0)
    with pytest.raises(KeyError):
        price_of("unknown-model")
    with pytest.raises(ValueError):
        cost_usd(Usage(llm_input_tokens=1, model_ids=("gpt-4.1-mini", "other")))


def test_summary_prices_a_run_whose_model_changed():
    from runner.replay import OpRecord, RunResult, RunSettings
    from interface.memory_system import Capabilities
    def op(model, n):
        return OpRecord(op="ingest", at=T, ref="s", usage=Usage(llm_calls=1, llm_input_tokens=n, model_ids=(model,)))
    r = RunResult(workload_id="w", generator="g", seed=0, system="x", system_version="1", system_config={},
                  capabilities=Capabilities(), settings=RunSettings(token_budget=None, delete_with_sources=True),
                  started_at=T, ops=(op("gpt-4.1-mini-2025-04-14", 1_000_000), op("gpt-4.1-2025-04-14", 1_000_000)),
                  probes=(), checkpoint_stats={})
    assert r.summary()["memory_cost_usd"] == pytest.approx(0.40 + 2.00)
