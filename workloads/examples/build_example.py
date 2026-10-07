"""A tiny hand-built coding-agent timeline that exercises every mutation kind
and probe type. Use it to smoke-test adapters and the runner before the real
generator exists.

    python -m workloads.examples.build_example  # writes example_timeline.json
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from interface.memory_system import Event, EventKind
from workloads.timeline import (
    KIND_TO_SOURCE,
    Checkpoint,
    Expected,
    ExpectedKind,
    Fact,
    Mutation,
    MutationKind,
    Probe,
    ProbeType,
    Timeline,
    expected_answer,
)

U = "u1"


def ts(month: int, day: int, hour: int = 10, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=timezone.utc)


# (event_id, session_id, time, kind, content)
STREAM = [
    ("e01", "s1", ts(1, 5, 10, 0), EventKind.USER_MESSAGE, "For payments-service we're on PostgreSQL 14."),
    ("e02", "s1", ts(1, 5, 10, 1), EventKind.USER_MESSAGE, "Run the tests with `pytest -x`, please."),
    ("e03", "s1", ts(1, 5, 10, 2), EventKind.USER_MESSAGE, "We deploy from the main branch."),
    ("e04", "s1", ts(1, 5, 10, 3), EventKind.ASSISTANT_MESSAGE, "Got it, I'll keep that in mind."),
    ("e05", "s2", ts(1, 20, 9, 0), EventKind.DOCUMENT, "README.md: Deploys go out from the `master` branch."),
    ("e06", "s2", ts(1, 20, 9, 5), EventKind.USER_MESSAGE, "Staging lives at staging-7.internal."),
    ("e07", "s2", ts(1, 20, 9, 9), EventKind.USER_MESSAGE, "Reminder: payments still runs on Postgres 14."),
    ("e08", "s3", ts(2, 15, 14, 0), EventKind.TOOL_RESULT, "migrate.log: payments-service migrated to PostgreSQL 16. OK"),
    ("e09", "s3", ts(2, 15, 14, 10), EventKind.USER_MESSAGE, "Tests now run with `make test`."),
    ("e10", "s4", ts(3, 10, 11, 0), EventKind.USER_MESSAGE, "Please forget the staging hostname, it shouldn't be stored."),
    ("e11", "s4", ts(3, 10, 11, 2), EventKind.ASSISTANT_MESSAGE, "Understood. Deploying from master as usual, then?"),
    ("e12", "s5", ts(3, 30, 16, 0), EventKind.USER_MESSAGE, "Let's refactor the cart module next."),
]

FACTS = [
    Fact(fact_id="f_db", user_id=U, entity="payments-service", attribute="database", category="codebase"),
    Fact(fact_id="f_test", user_id=U, entity="repo", attribute="test_command", category="workflow"),
    Fact(fact_id="f_branch", user_id=U, entity="repo", attribute="deploy_branch", category="workflow"),
    Fact(fact_id="f_staging", user_id=U, entity="staging", attribute="hostname", category="infrastructure"),
]

# (mutation_id, fact_id, kind, value, carrier event)
CHANGES = [
    ("m01", "f_db", MutationKind.INTRODUCE, "PostgreSQL 14", "e01"),
    ("m02", "f_test", MutationKind.INTRODUCE, "pytest -x", "e02"),
    ("m03", "f_branch", MutationKind.INTRODUCE, "main", "e03"),
    ("m04", "f_branch", MutationKind.CONTRADICT, "master", "e05"),
    ("m05", "f_staging", MutationKind.INTRODUCE, "staging-7.internal", "e06"),
    ("m06", "f_db", MutationKind.RESTATE, "PostgreSQL 14", "e07"),
    ("m07", "f_db", MutationKind.UPDATE, "PostgreSQL 16", "e08"),
    ("m08", "f_test", MutationKind.UPDATE, "make test", "e09"),
    ("m09", "f_staging", MutationKind.DELETE, None, "e10"),
    ("m10", "f_branch", MutationKind.CONTRADICT, "master", "e11"),
]

ALIASES = {
    "PostgreSQL 14": ("Postgres 14", "PG 14"),
    "PostgreSQL 16": ("Postgres 16", "PG 16"),
}


def build() -> Timeline:
    events = [
        Event(event_id=i, user_id=U, session_id=s, timestamp=t, kind=k, content=c)
        for i, s, t, k, c in STREAM
    ]
    by_id = {e.event_id: e for e in events}
    mutations = [
        Mutation(
            mutation_id=mid,
            fact_id=fid,
            kind=kind,
            value=value,
            at=by_id[eid].timestamp,
            source=KIND_TO_SOURCE[by_id[eid].kind],
            carrier_event_ids=(eid,),
        )
        for mid, fid, kind, value, eid in CHANGES
    ]

    n = len(events)

    def checkpoint(name: str, last_event: str) -> Checkpoint:
        idx = next(i for i, e in enumerate(events) if e.event_id == last_event)
        return Checkpoint(name=name, fraction=round((idx + 1) / n, 4), at=events[idx].timestamp)

    checkpoints = [checkpoint("c1", "e07"), checkpoint("c2", "e09"), checkpoint("c3", "e12")]
    cp = {c.name: c for c in checkpoints}

    def probe(pid, ptype, question, cp_name, fact_id, as_of=None) -> Probe:
        muts = [m for m in mutations if m.fact_id == fact_id]
        exp = expected_answer(muts, cp[cp_name].at, as_of)
        if exp.value in ALIASES:
            exp = exp.model_copy(update={"aliases": ALIASES[exp.value]})
        return Probe(
            probe_id=pid, user_id=U, probe_type=ptype, question=question,
            checkpoint=cp_name, fact_ids=(fact_id,), as_of=as_of, expected=exp,
        )

    probes = [
        probe("p01", ProbeType.CURRENT, "Which database version does payments-service use?", "c1", "f_db"),
        probe("p02", ProbeType.CONTAMINATION, "Which branch do we deploy from?", "c1", "f_branch"),
        probe("p03", ProbeType.POST_UPDATE, "Which database version does payments-service use?", "c2", "f_db"),
        probe("p04", ProbeType.POST_UPDATE, "How do I run the test suite?", "c2", "f_test"),
        probe("p05", ProbeType.HISTORICAL, "Which database version was payments-service on at the end of January?",
              "c3", "f_db", as_of=ts(1, 31, 23, 59)),
        probe("p06", ProbeType.POST_DELETE, "What's the staging hostname?", "c3", "f_staging"),
        probe("p07", ProbeType.CONTAMINATION, "Which branch do we deploy from?", "c3", "f_branch"),
        Probe(
            probe_id="p08", user_id=U, probe_type=ProbeType.ABSTENTION,
            question="Which message queue does payments-service use?", checkpoint="c3",
            expected=Expected(kind=ExpectedKind.ABSTAIN),
        ),
    ]

    return Timeline(
        workload_id="example-coding-agent",
        generator="hand-built/0.1",
        seed=0,
        events=tuple(events),
        facts=tuple(FACTS),
        mutations=tuple(mutations),
        checkpoints=tuple(checkpoints),
        probes=tuple(probes),
    )


if __name__ == "__main__":
    out = Path(__file__).with_name("example_timeline.json")
    tl = build()
    tl.save(out)
    print(f"wrote {out} ({len(tl.events)} events, {len(tl.mutations)} mutations, {len(tl.probes)} probes)")
    for p in tl.probes:
        e = p.expected
        print(f"  {p.probe_id} [{p.checkpoint}] {p.probe_type.value:<13} -> "
              f"{e.kind.value}={e.value!r} forbidden={list(e.forbidden)}")
