"""Generator tests: determinism, schema validity across seeds, coverage of
mutation and probe types, and the text-hygiene guarantees scoring relies on."""

from collections import defaultdict

import pytest

from runner.replay import run
from systems.full_replay.system import FullReplay
from workloads.generator import PRESETS, Config, generate, hygiene_errors
from workloads.timeline import MutationKind, ProbeType, SourceTrust, Timeline


@pytest.fixture(scope="module")
def tl() -> Timeline:
    return generate(PRESETS["small"])


def test_same_seed_same_timeline(tl):
    assert generate(PRESETS["small"]) == tl
    assert generate(Config(seed=1)).events != tl.events


@pytest.mark.parametrize("seed", range(10))
def test_valid_and_clean_across_seeds(seed):
    t = generate(Config(seed=seed))  # the Timeline validator re-derives every probe
    assert hygiene_errors(t) == []


def test_covers_every_mutation_and_probe_type(tl):
    assert {m.kind for m in tl.mutations} == set(MutationKind)
    assert {p.probe_type for p in tl.probes} == set(ProbeType) - {ProbeType.MULTI_FACT}
    assert len({e.user_id for e in tl.events}) == 3


def test_contradictions_only_against_trusted_values(tl):
    for f in tl.facts:
        setter = None
        for m in tl.mutations_for(f.fact_id):
            if m.kind is MutationKind.CONTRADICT:
                assert m.source in (SourceTrust.DOCUMENT, SourceTrust.MODEL)
                assert setter in (SourceTrust.USER, SourceTrust.TOOL)
            elif m.kind in (MutationKind.INTRODUCE, MutationKind.UPDATE):
                setter = m.source
            elif m.kind is MutationKind.DELETE:
                setter = None


def test_values_belong_to_one_fact_per_user(tl):
    owner: dict[tuple[str, str], str] = {}
    users = {f.fact_id: f.user_id for f in tl.facts}
    for m in tl.mutations:
        if m.value:
            assert owner.setdefault((users[m.fact_id], m.value), m.fact_id) == m.fact_id


def test_hygiene_catches_uncarried_value(tl):
    m = next(m for m in tl.mutations if m.value)
    e = next(e for e in tl.events if e.user_id == m.fact_id.split("/")[0] and e.event_id not in m.carrier_event_ids
             and not any(e.event_id in x.carrier_event_ids for x in tl.mutations))
    leaky = e.model_copy(update={"content": f"{e.content} By the way, {m.value}."})
    bad = tl.model_copy(update={"events": tuple(leaky if x.event_id == e.event_id else x for x in tl.events)})
    assert any(e.event_id in err for err in hygiene_errors(bad))


def test_runner_replays_generated_timeline(tl):
    result = run(FullReplay(), tl, token_budget=None)
    assert len(result.probes) == len(tl.probes)
    assert result.summary()["isolation_violations"] == 0
    by_type = defaultdict(set)
    for p in result.probes:
        by_type[p.probe_type].add(p.verdict.value)
    assert by_type[ProbeType.CURRENT] == {"correct"}  # everything is in the transcript
