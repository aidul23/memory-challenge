"""Schema self-tests: the example timeline is valid, round-trips through JSON,
and the validator catches the mistakes a generator is likely to make."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from interface.memory_system import Evidence, Query, RetrievalResult
from systems.no_memory.system import NoMemory
from workloads.examples.build_example import build
from workloads.timeline import Expected, ExpectedKind, Timeline


@pytest.fixture
def tl() -> Timeline:
    return build()


def _rebuild(tl: Timeline, **changes) -> Timeline:
    data = tl.model_dump()
    data.update(changes)
    return Timeline.model_validate(data)


def test_example_is_valid_and_round_trips(tl, tmp_path):
    path = tmp_path / "t.json"
    tl.save(path)
    assert Timeline.load(path) == tl


def test_ground_truth_semantics(tl):
    by_id = {p.probe_id: p.expected for p in tl.probes}
    assert by_id["p01"].value == "PostgreSQL 14"
    assert by_id["p02"].value == "main"
    assert "master" in by_id["p07"].forbidden                      # contamination
    assert by_id["p03"].value == "PostgreSQL 16"
    assert "PostgreSQL 14" in by_id["p03"].forbidden               # supersession
    assert by_id["p05"].value == "PostgreSQL 14"                   # historical
    assert "PostgreSQL 16" in by_id["p05"].forbidden
    assert by_id["p06"].kind is ExpectedKind.ABSTAIN               # deletion
    assert "staging-7.internal" in by_id["p06"].forbidden


def test_deleted_history_stays_hidden(tl):
    end = tl.checkpoint("c3").at
    before_delete = datetime(2026, 2, 1, tzinfo=timezone.utc)
    exp = tl.expected_for("f_staging", asked_at=end, as_of=before_delete)
    assert exp.kind is ExpectedKind.ABSTAIN
    assert "staging-7.internal" in exp.forbidden


def test_rejects_wrong_expected_answer(tl):
    probes = [p.model_dump() for p in tl.probes]
    probes[2]["expected"] = Expected(kind=ExpectedKind.VALUE, value="PostgreSQL 14").model_dump()
    with pytest.raises(ValidationError, match="ground truth is value='PostgreSQL 16'"):
        _rebuild(tl, probes=probes)


def test_rejects_missing_forbidden_value(tl):
    probes = [p.model_dump() for p in tl.probes]
    probes[2]["expected"]["forbidden"] = ()
    with pytest.raises(ValidationError, match="forbidden list omits"):
        _rebuild(tl, probes=probes)


def test_rejects_update_to_same_value(tl):
    muts = [m.model_dump() for m in tl.mutations]
    m07 = next(m for m in muts if m["mutation_id"] == "m07")
    m07["value"] = "PostgreSQL 14"
    with pytest.raises(ValidationError, match="UPDATE to the same value"):
        _rebuild(tl, mutations=muts, probes=())


def test_rejects_source_that_does_not_match_carrier(tl):
    muts = [m.model_dump() for m in tl.mutations]
    m04 = next(m for m in muts if m["mutation_id"] == "m04")
    m04["source"] = "tool"  # carrier e05 is a document
    with pytest.raises(ValidationError, match="does not match carrier kind"):
        _rebuild(tl, mutations=muts)


def test_contradiction_cannot_come_from_user(tl):
    muts = [m.model_dump() for m in tl.mutations]
    m04 = next(m for m in muts if m["mutation_id"] == "m04")
    m04["source"] = "user"
    with pytest.raises(ValidationError, match="is an UPDATE, not a CONTRADICT"):
        _rebuild(tl, mutations=muts)


def test_interface_rules():
    with pytest.raises(ValidationError, match="must not carry evidence"):
        RetrievalResult(abstained=True, evidence=(Evidence(content="x"),))
    sys = NoMemory()
    q = Query(query_id="q", user_id="u1", text="?", asked_at=datetime.now(timezone.utc))
    assert sys.retrieve(q).evidence == ()
    assert sys.capabilities.deletion is False
