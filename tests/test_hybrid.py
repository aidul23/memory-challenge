"""The ledger with an episodic fallback: facts win when the ledger has them,
raw turns fill in when it has nothing, and deletion clears both stores."""

from datetime import datetime, timezone

from interface.memory_system import DeleteRequest, Event, EventKind, Query
from runner.replay import run
from runner.scoring import Verdict
from systems.hybrid.system import LedgerHybrid
from workloads.examples.build_example import build


def ts(day: int) -> datetime:
    return datetime(2026, 1, day, 12, tzinfo=timezone.utc)


def ev(eid: str, day: int, content: str, kind=EventKind.USER_MESSAGE) -> Event:
    return Event(event_id=eid, user_id="u1", session_id=f"s{day}", timestamp=ts(day), kind=kind, content=content)


def ask(s, text: str, day: int = 28):
    return s.retrieve(Query(query_id="q", user_id="u1", text=text, asked_at=ts(day)))


def test_facts_win_and_old_turns_stay_out():
    s = LedgerHybrid()
    s.ingest([ev("a", 1, "We're on PostgreSQL 14.")])
    s.ingest([ev("b", 5, "Moved to PostgreSQL 16.")])
    res = ask(s, "Which database do we use?")
    assert [e.content.split(" (")[0] for e in res.evidence] == ["database: PostgreSQL 16"]


def test_falls_back_to_turns_when_ledger_has_nothing():
    s = LedgerHybrid()
    s.ingest([ev("a", 1, "My sister's wedding is in Lisbon this June.")])
    res = ask(s, "Where is my sister's wedding?")
    assert not res.abstained and "Lisbon" in res.evidence[0].content
    assert res.evidence[0].source_event_ids == ("a",)


def test_assistant_guess_is_only_shown_as_a_raw_turn():
    s = LedgerHybrid()
    s.ingest([ev("a", 1, "Deploys go out from the `master` branch.", kind=EventKind.ASSISTANT_MESSAGE)])
    res = ask(s, "Which branch do we deploy from?")
    assert res.evidence[0].content.startswith("[2026-01-01 12:00 assistant_message]")


def test_deletion_clears_turns_the_extractor_missed():
    s = LedgerHybrid()
    s.ingest([ev("a", 1, "Staging lives at staging-7.internal."),
              ev("b", 1, "Noted: I'll point the tests at staging-7.internal.", kind=EventKind.ASSISTANT_MESSAGE),
              ev("c", 1, "The CI runs on Fridays.")])
    s.delete(DeleteRequest(request_id="d", user_id="u1", requested_at=ts(3),
                           description="Please forget the staging hostname."))
    res = ask(s, "What's the staging hostname?")
    assert res.abstained and res.abstain_reason == "deleted at the user's request" and not res.evidence
    # The assistant's echo was never a ledger entry, but it named the value.
    assert {e.event_id for e in s.episodes._history("u1")} == {"c"}
    assert "staging-7" not in str(ask(s, "Where do the tests point?").evidence)
    # A replay of the erased events is rejected by both stores.
    s.ingest([ev("a", 1, "Staging lives at staging-7.internal."),
              ev("b", 1, "Noted: I'll point the tests at staging-7.internal.", kind=EventKind.ASSISTANT_MESSAGE)])
    assert {e.event_id for e in s.episodes._history("u1")} == {"c"}
    assert ask(s, "What's the staging hostname?").abstained


def test_augment_appends_turns_after_facts():
    s = LedgerHybrid(mode="augment")
    s.ingest([ev("a", 1, "We're on PostgreSQL 14."), ev("b", 1, "The database team meets on Mondays.")])
    res = ask(s, "Which database do we use?")
    assert res.evidence[0].content.startswith("database: PostgreSQL 14")
    assert len(res.evidence) > 1 and all(e.content.startswith("[") for e in res.evidence[1:])


def test_example_keeps_the_ledger_guarantees():
    for with_sources in (True, False):
        r = run(LedgerHybrid(), build(), delete_with_sources=with_sources)
        v = {p.probe_id: p.verdict for p in r.probes}
        assert v.pop("p08") == Verdict.NO_ABSTENTION  # unknown topic: the fallback offers turns
        assert set(v.values()) == {Verdict.CORRECT}


def test_restart_at_every_checkpoint_changes_nothing(tmp_path):
    from systems.full_replay.system import FullReplay
    from systems.ledger.system import Ledger
    from systems.transcript_rag.system import TranscriptRAG

    tl = build()
    for factory in (Ledger, LedgerHybrid, TranscriptRAG, FullReplay):
        restarts = []

        def reload(system, cp):
            path = tmp_path / f"{system.name}-{cp.name}.json"
            system.snapshot(path)
            fresh = factory()
            fresh.restore(path)
            restarts.append(cp.name)
            return fresh

        base = run(factory(), tl, token_budget=None, delete_with_sources=False)
        again = run(factory(), tl, token_budget=None, delete_with_sources=False, after_checkpoint=reload)
        assert restarts == [c.name for c in tl.checkpoints]
        assert [(p.verdict, p.evidence) for p in again.probes] == [(p.verdict, p.evidence) for p in base.probes]


def test_hybrid_restore_keeps_deletion(tmp_path):
    s = LedgerHybrid()
    s.ingest([ev("a", 1, "Staging lives at staging-7.internal."), ev("c", 1, "The CI runs on Fridays.")])
    s.delete(DeleteRequest(request_id="d", user_id="u1", requested_at=ts(3), description="forget the staging hostname"))
    s.snapshot(tmp_path / "h.json")
    r = LedgerHybrid()
    r.restore(tmp_path / "h.json")
    assert ask(r, "What's the staging hostname?").abstained
    r.ingest([ev("a", 1, "Staging lives at staging-7.internal.")])  # replay of an erased event
    assert ask(r, "What's the staging hostname?").abstained
    assert r.stats() == s.stats()


def test_deletion_clears_turns_about_the_slot_never_extracted():
    # Trusted-only extraction never reads the assistant's guess, so its value
    # is no ledger entry; the turn is still about the deleted fact.
    from tests.test_ledger import SilentRouter
    s = LedgerHybrid(SilentRouter(), extract_min_trust=2)
    s.ingest([ev("a", 1, "gateway-service deploy_branch is main."),
              ev("b", 2, "I'll assume the gateway-service deploy branch is master.", kind=EventKind.ASSISTANT_MESSAGE),
              ev("c", 2, "The CI runs on Fridays.")])
    s.delete(DeleteRequest(request_id="d", user_id="u1", requested_at=ts(3),
                           description="Forget gateway-service's deploy branch."))
    assert {e.event_id for e in s.episodes._history("u1")} == {"c"}
    assert "master" not in str(ask(s, "Which branch did you assume for gateway-service?").evidence)


def test_deletion_clears_assistant_turns_about_the_entity():
    from tests.test_ledger import SilentRouter
    s = LedgerHybrid(SilentRouter(), extract_min_trust=2)
    s.ingest([ev("a", 1, "inventory-service test_command is make test."),
              ev("b", 2, "Running `npm test` for inventory-service like last time.", kind=EventKind.ASSISTANT_MESSAGE),
              ev("c", 2, "Done. I opened a draft PR for cart-service.", kind=EventKind.ASSISTANT_MESSAGE)])
    s.delete(DeleteRequest(request_id="d", user_id="u1", requested_at=ts(3),
                           description="Please forget the test command of inventory-service."))
    assert {e.event_id for e in s.episodes._history("u1")} == {"c"}  # other services untouched
