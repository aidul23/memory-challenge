"""Ledger tests: the example timeline end to end, and each write/read rule
(restate, supersede, dispute, delete, as_of, isolation, persistence) in
isolation."""

from datetime import datetime, timezone

import pytest

from interface.memory_system import DeleteRequest, Event, EventKind, MemoryItemStatus, Query
from runner.replay import run
from runner.scoring import Verdict
from systems.ledger.system import Ledger
from workloads.examples.build_example import build


def ts(day: int) -> datetime:
    return datetime(2026, 1, day, 12, tzinfo=timezone.utc)


def ev(eid: str, day: int, content: str, kind=EventKind.USER_MESSAGE, user="u1") -> Event:
    return Event(event_id=eid, user_id=user, session_id=f"s{day}", timestamp=ts(day), kind=kind, content=content)


def ask(sys: Ledger, text: str, day: int = 28, as_of: int | None = None, user="u1"):
    return sys.retrieve(Query(query_id="q", user_id=user, text=text, asked_at=ts(day),
                              as_of=ts(as_of) if as_of else None))


def values(res) -> list[str]:
    return [e.content.split(": ", 1)[1].split(" (")[0] for e in res.evidence]


DB_Q = "Which database do we use?"


@pytest.mark.parametrize("with_sources", [True, False])
def test_example_all_correct(with_sources):
    result = run(Ledger(), build(), delete_with_sources=with_sources)
    assert {p.probe_id: p.verdict for p in result.probes} == {p.probe_id: Verdict.CORRECT for p in result.probes}


def test_restate_adds_provenance_not_items():
    s = Ledger()
    s.ingest([ev("a", 1, "We're on PostgreSQL 14.")])
    rep = s.ingest([ev("b", 2, "Still on Postgres 14.")])
    assert (rep.items_added, rep.items_updated) == (0, 1)
    assert s.stats().n_items == 1
    assert ask(s, DB_Q).evidence[0].source_event_ids == ("a", "b")


def test_reingest_is_idempotent():
    s = Ledger()
    batch = [ev("a", 1, "We're on PostgreSQL 14."), ev("b", 2, "Moved to PostgreSQL 16.")]
    s.ingest(batch)
    rep = s.ingest(batch)
    assert rep.items_added == 0 and s.stats().n_items == 2


def test_supersede_and_as_of():
    s = Ledger()
    s.ingest([ev("a", 1, "We're on PostgreSQL 14.")])
    s.ingest([ev("b", 10, "migrate.log: migrated to PostgreSQL 16", kind=EventKind.TOOL_RESULT)])
    assert values(ask(s, DB_Q)) == ["PostgreSQL 16"]
    assert values(ask(s, DB_Q, as_of=5)) == ["PostgreSQL 14"]
    assert values(ask(s, DB_Q, day=5)) == ["PostgreSQL 14"]  # nothing from after asked_at
    statuses = {i.content: i.status for i in s.export_items("u1")}
    assert statuses == {"database: PostgreSQL 14": MemoryItemStatus.SUPERSEDED,
                        "database: PostgreSQL 16": MemoryItemStatus.ACTIVE}


def test_as_of_before_first_statement_abstains():
    s = Ledger()
    s.ingest([ev("a", 10, "We're on PostgreSQL 14.")])
    res = ask(s, DB_Q, as_of=3)
    assert res.abstained and "no value held" in res.abstain_reason


@pytest.mark.parametrize("kind", [EventKind.DOCUMENT, EventKind.ASSISTANT_MESSAGE])
def test_less_trusted_source_is_disputed_not_retrieved(kind):
    s = Ledger()
    s.ingest([ev("a", 1, "We deploy from the main branch.")])
    rep = s.ingest([ev("b", 2, "Deploys go out from the `master` branch.", kind=kind)])
    assert rep.items_rejected == 1
    assert values(ask(s, "Which branch do we deploy from?")) == ["main"]
    assert s.stats().extra["disputes"] == 1


def test_less_trusted_source_may_introduce():
    s = Ledger()
    s.ingest([ev("a", 1, "README: Deploys go out from the `master` branch.", kind=EventKind.DOCUMENT)])
    s.ingest([ev("b", 2, "We deploy from the main branch.")])
    assert values(ask(s, "Which branch do we deploy from?")) == ["main"]


@pytest.mark.parametrize("sources", [("a",), ()])
def test_delete_erases_history_and_blocks_replay(sources):
    s = Ledger()
    s.ingest([ev("a", 1, "Staging lives at staging-7.internal.")])
    s.ingest([ev("b", 2, "Staging moved: staging lives at staging-9.internal now.")])
    rep = s.delete(DeleteRequest(request_id="d", user_id="u1", requested_at=ts(3),
                                 description="Please forget the staging hostname.",
                                 source_event_ids=sources))
    assert rep.items_deleted == 2
    for as_of in (None, 1, 2):
        res = ask(s, "What's the staging hostname?", as_of=as_of)
        assert res.abstained and not res.evidence
    assert ask(s, "What's the staging hostname?").abstain_reason == "deleted at the user's request"
    s.ingest([ev("a", 1, "Staging lives at staging-7.internal.")])
    assert ask(s, "What's the staging hostname?").abstained
    # A later statement starts a fresh, visible history.
    s.ingest([ev("c", 5, "Staging lives at staging-12.internal.")])
    assert values(ask(s, "What's the staging hostname?")) == ["staging-12.internal"]


def test_unknown_topic_abstains():
    s = Ledger()
    s.ingest([ev("a", 1, "We're on PostgreSQL 14.")])
    assert ask(s, "Which message queue do we use?").abstained


def test_users_are_isolated():
    s = Ledger()
    s.ingest([ev("a", 1, "We're on PostgreSQL 14.", user="u1")])
    s.ingest([ev("b", 2, "We're on MySQL 8.", user="u2")])
    assert values(ask(s, DB_Q, user="u1")) == ["PostgreSQL 14"]
    assert values(ask(s, DB_Q, user="u2")) == ["MySQL 8"]
    assert ask(s, DB_Q, user="u3").abstained
    s.delete(DeleteRequest(request_id="d", user_id="u2", requested_at=ts(3),
                           description="forget the database", source_event_ids=("a",)))
    assert values(ask(s, DB_Q, user="u1")) == ["PostgreSQL 14"]
    assert ask(s, DB_Q, user="u2").abstained
    # u2's request named u1's event; that must not block it for u1.
    s.reset()
    s.delete(DeleteRequest(request_id="d", user_id="u2", requested_at=ts(3),
                           description="forget the database", source_event_ids=("a",)))
    s.ingest([ev("a", 4, "We're on PostgreSQL 14.", user="u1")])
    assert values(ask(s, DB_Q, user="u1")) == ["PostgreSQL 14"]


def test_snapshot_restore_roundtrip(tmp_path):
    s = Ledger()
    s.ingest([ev("a", 1, "We're on PostgreSQL 14."), ev("b", 2, "Staging lives at staging-7.internal.")])
    s.ingest([ev("c", 10, "Moved to PostgreSQL 16.")])
    s.ingest([ev("d", 11, "Deploys go out from `master`.", kind=EventKind.DOCUMENT),
              ev("e", 11, "We deploy from the main branch.")])
    s.delete(DeleteRequest(request_id="x", user_id="u1", requested_at=ts(12),
                           description="forget the staging hostname"))
    path = tmp_path / "ledger.json"
    s.snapshot(path)

    r = Ledger()
    r.restore(path)
    for q, as_of in [(DB_Q, None), (DB_Q, 5), ("Which branch?", None), ("staging hostname?", None)]:
        a, b = ask(s, q, as_of=as_of), ask(r, q, as_of=as_of)
        assert (a.evidence, a.abstained, a.abstain_reason) == (b.evidence, b.abstained, b.abstain_reason)
    assert r.stats() == s.stats()
    assert r.ingest([ev("b", 2, "Staging lives at staging-7.internal.")]).items_rejected == 1
