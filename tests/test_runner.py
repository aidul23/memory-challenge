"""Runner tests: delivery protocol, scoring, truncation, and the transcript
baselines' expected behaviour on the example timeline."""

from datetime import datetime, timezone
from typing import Any, Sequence

import pytest

from interface.memory_system import (
    Capabilities, DeleteReport, DeleteRequest, Event, Evidence, MemorySystem,
    Query, RetrievalResult, StoreStats, WriteReport,
)
from interface.tokens import count_tokens
from runner.replay import run
from runner.scoring import ErrorType, Verdict, mentions
from systems.full_replay.system import FullReplay
from systems.no_memory.system import NoMemory
from systems.transcript_rag.system import TranscriptRAG
from workloads.examples.build_example import build
from workloads.timeline import ExpectedKind, Timeline


@pytest.fixture
def tl() -> Timeline:
    return build()


def _by_id(result):
    return {p.probe_id: p for p in result.probes}


def _types(p) -> set[ErrorType]:
    return {t for h in p.forbidden_hits for t in h.error_types}


class Recorder(MemorySystem):
    """Logs every call the runner makes."""

    name, version = "recorder", "1"
    capabilities = Capabilities(deletion=True)

    def __init__(self):
        self.calls: list[tuple[str, Any]] = []

    def ingest(self, events: Sequence[Event]) -> WriteReport:
        self.calls.append(("ingest", list(events)))
        return WriteReport()

    def delete(self, request: DeleteRequest) -> DeleteReport:
        self.calls.append(("delete", request))
        return DeleteReport()

    def retrieve(self, query: Query) -> RetrievalResult:
        self.calls.append(("retrieve", query))
        return RetrievalResult()

    def stats(self, user_id=None) -> StoreStats:
        return StoreStats()

    def reset(self) -> None:
        self.calls.clear()

    def config(self) -> dict[str, Any]:
        return {}


class Oracle(Recorder):
    """Answers every probe from ground truth: the scorer must call it perfect."""

    name = "oracle"
    capabilities = Capabilities(abstention=True)

    def __init__(self, tl: Timeline):
        super().__init__()
        self.expected = {p.probe_id: p.expected for p in tl.probes}

    def retrieve(self, query: Query) -> RetrievalResult:
        exp = self.expected[query.query_id]
        if exp.kind is ExpectedKind.ABSTAIN:
            return RetrievalResult(abstained=True)
        return RetrievalResult(evidence=(Evidence(content=f"Answer: {exp.value}"),))


def test_delivery_protocol(tl):
    rec = Recorder()
    run(rec, tl)
    seen: list[str] = []
    for op, arg in rec.calls:
        if op == "ingest":
            assert len({(e.user_id, e.session_id) for e in arg}) == 1, "one session per call"
            seen += [e.event_id for e in arg]
        elif op == "retrieve":
            ingested = {e.event_id for e in tl.events if e.timestamp <= arg.asked_at}
            assert set(seen) == ingested, "exactly the events up to asked_at"
        elif op == "delete":
            assert arg.request_id == "m09"
            assert seen[-1] == "e10", "delivered right after the user's request"
            assert arg.source_event_ids == ("e06",)
            assert "forget the staging hostname" in arg.description
    assert seen == [e.event_id for e in tl.events], "every event once, in order"


def test_sources_can_be_withheld(tl):
    rec = Recorder()
    run(rec, tl, delete_with_sources=False)
    (req,) = [a for op, a in rec.calls if op == "delete"]
    assert req.source_event_ids == ()


def test_oracle_scores_perfectly(tl):
    result = run(Oracle(tl), tl)
    assert {p.verdict for p in result.probes} == {Verdict.CORRECT}


def test_no_memory_lower_bound(tl):
    result = run(NoMemory(), tl)
    for p in result.probes:
        expected = Verdict.NO_ABSTENTION if p.expected.kind is ExpectedKind.ABSTAIN else Verdict.MISSED
        assert p.verdict is expected
    assert [o.error is not None for o in result.ops if o.op == "delete"] == [True]


def test_full_replay_errors_are_typed(tl):
    p = _by_id(run(FullReplay(), tl, token_budget=None))
    assert p["p01"].verdict is Verdict.CORRECT
    assert p["p02"].verdict is Verdict.POLLUTED and _types(p["p02"]) == {ErrorType.CONTAMINATED}
    assert p["p03"].verdict is Verdict.POLLUTED and _types(p["p03"]) == {ErrorType.STALE}
    assert p["p05"].verdict is Verdict.POLLUTED and _types(p["p05"]) == {ErrorType.FUTURE}
    assert p["p06"].verdict is Verdict.NO_ABSTENTION
    assert p["p03"].forbidden_hits[0].source_event_ids == ("e01", "e07")  # "Postgres 14" alias


def test_deletion_without_sources_leaks(tl):
    p = _by_id(run(FullReplay(), tl, token_budget=None, delete_with_sources=False))
    assert p["p06"].verdict is Verdict.LEAKED
    assert _types(p["p06"]) == {ErrorType.RESURRECTED}


def test_runner_truncates_over_budget(tl):
    p = _by_id(run(FullReplay(), tl, token_budget=20))["p03"]
    assert p.budget_exceeded and p.injected_tokens <= 20 < p.evidence_tokens
    assert sum(count_tokens(e.content) for e in p.evidence) == p.injected_tokens


def test_rag_fits_its_own_budget(tl):
    result = run(TranscriptRAG(), tl, token_budget=30)
    assert not any(p.budget_exceeded for p in result.probes)
    assert all(p.injected_tokens <= 30 for p in result.probes)


@pytest.mark.parametrize("cls", [FullReplay, TranscriptRAG])
def test_transcript_store_is_idempotent_and_deletion_sticks(tl, cls):
    sys = cls()
    sys.ingest(tl.events)
    sys.ingest(tl.events)
    assert sys.stats("u1").n_items == len(tl.events)
    sys.delete(DeleteRequest(request_id="d", user_id="u1", requested_at=datetime.now(timezone.utc),
                             description="forget it", source_event_ids=("e06",)))
    sys.ingest(tl.events)  # crash-and-replay must not resurrect it
    assert sys.stats("u1").n_items == len(tl.events) - 1


def test_mentions_matches_whole_phrases():
    assert mentions("We deploy from the main branch.", "main")
    assert not mentions("Remains to maintain the domain.", "main")
    assert mentions("run `pytest   -x` now", "pytest -x")
    assert mentions("on postgres 14.", "Postgres 14")
    assert not mentions("PostgreSQL 140", "PostgreSQL 14")
