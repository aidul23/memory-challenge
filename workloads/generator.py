"""Seeded synthetic generator for coding-agent timelines.

    python -m workloads.generator                       # small preset, seed 0
    python -m workloads.generator --preset medium --seed 3
    python -m runner --timeline workloads/generated/synthetic-small-s0.json

Each user owns a few services; each service gets a random subset of the fact
kinds in ``CATALOG`` (database, test command, deploy branch, ...). Every fact
gets a mutation sequence from a small state machine that respects the schema
rules (INTRODUCE, then RESTATE / UPDATE / CONTRADICT / DELETE, with a fresh
INTRODUCE after a deletion). Each mutation is written as one event in its own
session, phrased by a randomly chosen template for its source, and sessions
are padded with filler and distractor events that mention topics but never
values. Probes are sampled per checkpoint from facts whose state makes each
probe type meaningful, and their expectations come from
``workloads.timeline.expected_answer``, so the schema's own consistency check
re-verifies them.

Design choices that keep the ground truth defensible:

* A value belongs to exactly one fact per user, and no value is a sub-phrase
  of another, so "value X surfaced" always means one specific fact.
* No event mentions a value it does not carry (checked by ``hygiene_errors``).
* CONTRADICT comes only from a document or the agent, and only against a value
  established by the user or a tool. A stale README contradicting a value
  that itself came from a document would be a judgement call, not ground truth.
* Phrasings vary on purpose (generic templates plus kind-specific ones), so an
  extractor tuned to the hand-built example will not cover them all.

MULTI_FACT probes are not generated: they need hand-written expectations.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

from interface.memory_system import Event, EventKind
from runner.scoring import mentions
from workloads.timeline import (
    KIND_TO_SOURCE, Checkpoint, Expected, ExpectedKind, Fact, Mutation,
    MutationKind, Probe, ProbeType, Timeline, expected_answer,
)

GENERATOR = "synthetic/0.1"

# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------

#: Template slots: {svc} service name, {v} value, {label} attribute label,
#: {attr} attribute id, {short} service name without "-service".
Templates = tuple[str, ...]


@dataclass(frozen=True)
class FactKind:
    attribute: str
    label: str
    category: str
    values: Callable[[str], Sequence[str]]
    """Value pool for one service (a function so hostnames can embed it)."""
    aliases: Callable[[str], tuple[str, ...]] = lambda v: ()
    state: Templates = ()
    update: Templates = ()
    tool: Templates = ()
    """TOOL_RESULT or CODE_CHANGE phrasings (introduce, update, restate)."""
    document: Templates = ()
    model: Templates = ()
    question: Templates = ()
    delete_weight: float = 1.0
    """Relative chance a mutation step is a deletion (private data is higher)."""


def _pg_alias(v: str) -> tuple[str, ...]:
    if v.startswith("PostgreSQL "):
        n = v.split()[1]
        return (f"Postgres {n}", f"PG {n}")
    return ()


CATALOG: tuple[FactKind, ...] = (
    FactKind(
        "database", "database", "codebase",
        lambda s: [f"PostgreSQL {n}" for n in range(12, 18)]
        + ["MySQL 5.7", "MySQL 8.0", "MySQL 8.4", "MariaDB 10.11", "MariaDB 11.4", "MongoDB 6.0", "MongoDB 7.0"],
        aliases=_pg_alias,
        state=("{svc} runs on {v}.", "We're on {v} for {svc}."),
        update=("We upgraded {svc} to {v} last night.", "{svc} has been moved over to {v}."),
        tool=("migrate.log: {svc} migrated to {v}. OK", "db-check: {svc} server_version={v}"),
        document=("docs/architecture.md: {svc} stores its data in {v}.",),
        model=("I'll write the {svc} migration against {v}, as before.",),
        question=("Which database does {svc} use?", "What DB version is {svc} running?"),
    ),
    FactKind(
        "test_command", "test command", "workflow",
        lambda s: ["pytest -x", "make test", "npm test", "tox -e py312", "go test ./...",
                   "cargo test", "./gradlew test", "bazel test //...", "pnpm vitest run", "pytest -n auto"],
        state=("Run the {svc} tests with `{v}`.", "For {svc}, tests run via `{v}`."),
        update=("{svc} tests now run with `{v}`.", "From today, use `{v}` to test {svc}."),
        tool=("diff {short}/.github/workflows/ci.yml: + run: {v}",),
        document=("CONTRIBUTING.md ({svc}): run `{v}` before pushing.",),
        model=("Running `{v}` for {svc} like last time.",),
        question=("How do I run the {svc} test suite?", "What's the command to test {svc}?"),
    ),
    FactKind(
        "deploy_branch", "deploy branch", "workflow",
        lambda s: ["main", "master", "trunk", "stable", "prod", "develop", "release-train"],
        state=("We deploy {svc} from the {v} branch.", "{svc} ships from `{v}`."),
        update=("{svc} deploys from `{v}` now.",),
        tool=("git: default branch of {svc} changed to '{v}'",),
        document=("CONTRIBUTING.md: {svc} deploys go out from the `{v}` branch.",),
        model=("Deploying {svc} from {v} as usual, then?",),
        question=("Which branch do we deploy {svc} from?", "What's the deploy branch for {svc}?"),
    ),
    FactKind(
        "staging_host", "staging hostname", "infrastructure",
        lambda s: [f"{s.removesuffix('-service')}-stg-{n}.internal" for n in range(1, 10)],
        state=("{svc} staging lives at {v}.", "The staging box for {svc} is {v}."),
        update=("{svc} staging moved to {v}.",),
        tool=("dns: {v} -> {svc} staging (A record created)",),
        document=("runbook.md: {svc} staging is reachable at {v}.",),
        model=("I'll point the {svc} smoke tests at {v}.",),
        question=("What's the staging hostname for {svc}?", "Where does {svc} staging live?"),
        delete_weight=4.0,
    ),
    FactKind(
        "runtime", "runtime", "codebase",
        lambda s: ["Python 3.10", "Python 3.11", "Python 3.12", "Python 3.13", "Node 18", "Node 20",
                   "Node 22", "Go 1.22", "Go 1.23", "Java 17", "Java 21"],
        aliases=lambda v: (f"Node.js {v.split()[1]}",) if v.startswith("Node ") else (),
        state=("{svc} is built on {v}.", "{svc} targets {v}."),
        update=("We bumped {svc} to {v}.",),
        tool=("Dockerfile ({svc}): FROM image updated to {v}",),
        document=("README.md ({svc}): requires {v}.",),
        model=("I'll keep the {svc} code compatible with {v}.",),
        question=("Which runtime does {svc} target?", "What language version is {svc} on?"),
    ),
    FactKind(
        "message_queue", "message queue", "infrastructure",
        lambda s: ["RabbitMQ", "Kafka", "Amazon SQS", "NATS", "Redis Streams", "Google Pub/Sub", "ActiveMQ"],
        state=("{svc} publishes its events to {v}.", "{svc} uses {v} for async jobs."),
        update=("We migrated {svc} events to {v}.",),
        tool=("deploy.log: {svc} broker endpoint switched to {v}",),
        document=("docs/events.md: {svc} events flow through {v}.",),
        model=("I'll add a {v} consumer for {svc}.",),
        question=("Which message queue does {svc} use?", "Where does {svc} publish its events?"),
    ),
    FactKind(
        "ci_provider", "CI provider", "workflow",
        lambda s: ["GitHub Actions", "GitLab CI", "CircleCI", "Jenkins", "Buildkite", "Travis CI", "Drone CI"],
        state=("CI for {svc} runs on {v}.", "{svc} builds on {v}."),
        update=("We moved {svc} CI over to {v}.",),
        tool=("webhook: {svc} build #812 reported by {v}",),
        document=("README.md ({svc}): CI badge links to {v}.",),
        model=("I'll update the {v} pipeline for {svc}.",),
        question=("Which CI does {svc} use?", "Where do the {svc} builds run?"),
    ),
    FactKind(
        "cloud_region", "cloud region", "infrastructure",
        lambda s: ["eu-north-1", "eu-west-1", "eu-central-1", "us-east-1", "us-west-2", "ap-southeast-2"],
        state=("{svc} is deployed in {v}.", "{svc} runs in the {v} region."),
        update=("{svc} was relocated to {v}.",),
        tool=("terraform apply ({svc}): region = \"{v}\"",),
        document=("infra/README.md: {svc} lives in {v}.",),
        model=("I'll create the {svc} bucket in {v}.",),
        question=("Which region is {svc} deployed in?", "Where is {svc} hosted?"),
    ),
    FactKind(
        "oncall", "on-call owner", "team",
        lambda s: ["Aino Virtanen", "Mikko Laine", "Sara Nyström", "Jonas Berg", "Leena Koski",
                   "Tomás Ruiz", "Priya Nair"],
        state=("{v} is on call for {svc}.", "Page {v} if {svc} breaks."),
        update=("{v} took over the {svc} on-call rotation.",),
        tool=("pagerduty: {svc} escalation policy -> {v}",),
        document=("OWNERS ({svc}): {v}",),
        model=("I'll ping {v} about the {svc} alert.",),
        question=("Who is on call for {svc}?", "Who owns {svc} incidents?"),
        delete_weight=3.0,
    ),
)

GENERIC = {
    "state": ("For {svc}, the {label} is {v}.", "Note for later: {svc} uses {v} as its {label}.",
              "{svc}'s {label}: {v}."),
    "restate": ("Reminder: {svc} still uses {v} as its {label}.",
                "Just confirming, the {label} for {svc} is still {v}."),
    "update": ("Change of plan: {svc} now uses {v} as its {label}.",
               "We switched the {label} for {svc} to {v}.",
               "Update: {svc}'s {label} is {v} from now on."),
    "tool": ("config-sync: {short}.{attr} = {v}",),
    "document": ("wiki/{short}.md: {label}: {v}", "README.md ({svc}): the {label} is {v}."),
    "model": ("Got it. I'll assume {v} for the {label} of {svc}.",
              "Okay, using {v} as the {svc} {label}."),
    "delete": ("Please forget the {label} of {svc}; don't store it.",
               "Remove what you know about {svc}'s {label}, please.",
               "Don't keep {svc}'s {label} in memory anymore."),
    "question": ("What is the {label} of {svc}?", "Which {label} does {svc} use?",
                 "Remind me: {svc}'s {label}?"),
    "historical": ("What was the {label} of {svc} on {date}?",
                   "Back on {date}, which {label} did {svc} use?"),
}

FILLER: dict[EventKind, Templates] = {
    EventKind.USER_MESSAGE: (
        "Can you look into the flaky checkout test in {svc}?", "Let's clean up the logging in {svc}.",
        "How is the {svc} refactor going?", "Please add type hints to the {svc} client.",
        "The {label} for {svc} came up in standup today.", "Let's not touch the {svc} {label} this sprint.",
        "Thanks, that looks good.", "Can you summarise yesterday's changes?",
    ),
    EventKind.ASSISTANT_MESSAGE: (
        "Done. I opened a draft PR for {svc}.", "I've added the missing type hints.",
        "The refactor is about halfway done.", "Looking into it now.",
        "I couldn't reproduce the failure locally.",
    ),
    EventKind.TOOL_CALL: (
        'run_shell("git log -5 --oneline")', 'read_file("{short}/src/handlers.py")',
        'grep("TODO", "{short}/")',
    ),
    EventKind.TOOL_RESULT: (
        "3 files changed, 41 insertions(+), 7 deletions(-)", "All checks passed.", "No matches found.",
    ),
}

SERVICES = ("payments-service", "cart-service", "auth-service", "search-service", "billing-service",
            "inventory-service", "notifications-service", "gateway-service")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    seed: int = 0
    users: int = 3
    services_per_user: tuple[int, int] = (2, 3)
    kinds_per_service: tuple[int, int] = (3, 5)
    sessions_per_user: int = 16
    mutations_per_fact: tuple[int, int] = (1, 5)
    fillers_per_session: tuple[int, int] = (1, 4)
    span_days: int = 180
    checkpoints: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    probes_per_type: int = 2
    """Per user, per checkpoint, per probe type (when eligible facts exist)."""
    preset: str = "custom"
    start: datetime = datetime(2026, 1, 5, tzinfo=timezone.utc)


PRESETS = {
    "small": Config(preset="small"),
    "medium": Config(preset="medium", users=8, services_per_user=(2, 4), sessions_per_user=40,
                     mutations_per_fact=(1, 8), span_days=365),
}


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


@dataclass
class _Step:
    """One planned mutation before it has an event."""

    fact_id: str
    kind: MutationKind
    value: str | None
    event_kind: EventKind
    text: str


@dataclass
class _FactPlan:
    fact: Fact
    fk: FactKind
    svc: str
    steps: list[_Step] = field(default_factory=list)


def _fill(t: str, svc: str, fk: FactKind | None = None, v: str = "", **kw: str) -> str:
    return t.format(svc=svc, short=svc.removesuffix("-service"), v=v,
                    label=fk.label if fk else "", attr=fk.attribute if fk else "", **kw)


def _plan_fact(rng: random.Random, plan: _FactPlan, n_steps: int, used: set[str]) -> None:
    """State machine over mutation kinds. ``used`` holds the user's values for
    this attribute across all facts, so no value is shared between facts."""
    fk, svc, fid = plan.fk, plan.svc, plan.fact.fact_id
    pool = list(fk.values(svc))
    own: list[str] = []          # values this fact has held (contradictions may reuse them)
    active: str | None = None
    trusted = False              # active value set by the user or a tool

    def fresh() -> str | None:
        free = [v for v in pool if v not in used]
        if not free:
            return None
        v = rng.choice(free)
        used.add(v)
        own.append(v)
        return v

    def step(kind: MutationKind, v: str | None, ek: EventKind, key: str) -> None:
        templates = {"state": fk.state + GENERIC["state"], "update": fk.update + GENERIC["update"],
                     "restate": GENERIC["restate"], "tool": fk.tool + GENERIC["tool"],
                     "document": fk.document + GENERIC["document"], "model": fk.model + GENERIC["model"],
                     "delete": GENERIC["delete"]}[key]
        plan.steps.append(_Step(fid, kind, v, ek, _fill(rng.choice(templates), svc, fk, v or "")))

    for _ in range(n_steps):
        if active is None:
            v = fresh()
            if v is None:
                break
            r = rng.random()
            if r < 0.8:
                step(MutationKind.INTRODUCE, v, EventKind.USER_MESSAGE, "state")
            elif r < 0.9:
                step(MutationKind.INTRODUCE, v, rng.choice([EventKind.TOOL_RESULT, EventKind.CODE_CHANGE]), "tool")
            else:
                step(MutationKind.INTRODUCE, v, EventKind.DOCUMENT, "document")
            active, trusted = v, plan.steps[-1].event_kind is not EventKind.DOCUMENT
            continue

        options = {"restate": 1.0, "update": 1.4, "delete": 0.3 * fk.delete_weight}
        if trusted:
            options["contradict"] = 1.0
        kind = rng.choices(list(options), weights=list(options.values()))[0]
        if kind == "restate":
            if rng.random() < 0.7:
                step(MutationKind.RESTATE, active, EventKind.USER_MESSAGE, "restate")
            else:
                step(MutationKind.RESTATE, active, rng.choice([EventKind.TOOL_RESULT, EventKind.CODE_CHANGE]), "tool")
        elif kind == "update":
            v = fresh()
            if v is None:
                continue
            if rng.random() < 0.6:
                step(MutationKind.UPDATE, v, EventKind.USER_MESSAGE, "update")
            else:
                step(MutationKind.UPDATE, v, rng.choice([EventKind.TOOL_RESULT, EventKind.CODE_CHANGE]), "tool")
            active, trusted = v, True
        elif kind == "contradict":
            # Half the time a stale source repeats one of this fact's old values.
            stale = [v for v in own if v != active]
            v = rng.choice(stale) if stale and rng.random() < 0.5 else fresh()
            if v is None:
                continue
            if rng.random() < 0.5:
                step(MutationKind.CONTRADICT, v, EventKind.DOCUMENT, "document")
            else:
                step(MutationKind.CONTRADICT, v, EventKind.ASSISTANT_MESSAGE, "model")
        else:
            step(MutationKind.DELETE, None, EventKind.USER_MESSAGE, "delete")
            active, trusted = None, False


def _filler(rng: random.Random, svc: str, fk: FactKind) -> tuple[EventKind, str]:
    kind = rng.choice(list(FILLER))
    return kind, _fill(rng.choice(FILLER[kind]), svc, fk)


def generate(cfg: Config) -> Timeline:
    rng = random.Random(cfg.seed)
    users = [f"u{i + 1}" for i in range(cfg.users)]

    plans: dict[str, list[_FactPlan]] = {}
    facts: list[Fact] = []
    for u in users:
        plans[u] = []
        used: dict[str, set[str]] = defaultdict(set)
        for svc in rng.sample(SERVICES, rng.randint(*cfg.services_per_user)):
            for fk in rng.sample(CATALOG, rng.randint(*cfg.kinds_per_service)):
                fact = Fact(fact_id=f"{u}/{svc}/{fk.attribute}", user_id=u, entity=svc,
                            attribute=fk.attribute, category=fk.category, description=f"{fk.label} of {svc}")
                plan = _FactPlan(fact, fk, svc)
                n = min(rng.randint(*cfg.mutations_per_fact), cfg.sessions_per_user)
                _plan_fact(rng, plan, n, used[fk.attribute])
                if plan.steps:
                    plans[u].append(plan)
                    facts.append(fact)

    # Schedule: each user's sessions on distinct days; a fact's steps go to
    # increasing sessions, so its mutations are strictly ordered in time.
    raw: list[tuple[datetime, int, str, str, EventKind, str, _Step | None]] = []
    for ui, u in enumerate(users):
        days = sorted(rng.sample(range(cfg.span_days), cfg.sessions_per_user))
        sessions: list[list[tuple[EventKind, str, _Step | None]]] = [[] for _ in days]
        for plan in plans[u]:
            for s, st in zip(sorted(rng.sample(range(len(days)), len(plan.steps))), plan.steps):
                sessions[s].append((st.event_kind, st.text, st))
        all_plans = plans[u]
        for s, items in enumerate(sessions):
            for _ in range(rng.randint(*cfg.fillers_per_session)):
                p = rng.choice(all_plans)
                items.append((*_filler(rng, p.svc, p.fk), None))
            rng.shuffle(items)
            t = cfg.start + timedelta(days=days[s], hours=rng.randint(8, 17), minutes=rng.randint(0, 50),
                                      seconds=ui)
            for kind, text, st in items:
                raw.append((t, ui, f"{u}-s{s + 1:03d}", u, kind, text, st))
                t += timedelta(minutes=rng.randint(1, 6))
    raw.sort(key=lambda r: (r[0], r[1]))

    events: list[Event] = []
    mutations: list[Mutation] = []
    for i, (t, _, sid, u, kind, text, st) in enumerate(raw):
        eid = f"e{i + 1:05d}"
        events.append(Event(event_id=eid, user_id=u, session_id=sid, timestamp=t, kind=kind, content=text))
        if st is not None:
            mutations.append(Mutation(
                mutation_id=f"m{len(mutations) + 1:05d}", fact_id=st.fact_id, kind=st.kind, value=st.value,
                at=t, source=KIND_TO_SOURCE[kind], carrier_event_ids=(eid,),
            ))

    checkpoints = []
    for n, frac in enumerate(cfg.checkpoints):
        idx = max(0, round(frac * len(events)) - 1)
        checkpoints.append(Checkpoint(name=f"c{n + 1}", fraction=round((idx + 1) / len(events), 4),
                                      at=events[idx].timestamp))

    plan_of = {p.fact.fact_id: p for ps in plans.values() for p in ps}
    probes = _probes(rng, cfg, users, plan_of, mutations, checkpoints)
    tl = Timeline(
        workload_id=f"synthetic-{cfg.preset}-s{cfg.seed}", generator=GENERATOR, seed=cfg.seed,
        events=tuple(events), facts=tuple(facts), mutations=tuple(mutations),
        checkpoints=tuple(checkpoints), probes=tuple(probes),
    )
    errors = hygiene_errors(tl)
    if errors:
        raise ValueError("generator produced ambiguous text:\n  " + "\n  ".join(errors[:20]))
    return tl


def _probes(rng: random.Random, cfg: Config, users: Sequence[str], plan_of: dict[str, _FactPlan],
            mutations: Sequence[Mutation], checkpoints: Sequence[Checkpoint]) -> list[Probe]:
    by_fact: dict[str, list[Mutation]] = defaultdict(list)
    for m in mutations:
        by_fact[m.fact_id].append(m)
    probes: list[Probe] = []

    def add(ptype: ProbeType, user: str, question: str, cp: Checkpoint, exp: Expected,
            fact_ids: tuple[str, ...] = (), as_of: datetime | None = None) -> None:
        probes.append(Probe(probe_id=f"p{len(probes) + 1:04d}", user_id=user, probe_type=ptype,
                            question=question, checkpoint=cp.name, fact_ids=fact_ids, as_of=as_of, expected=exp))

    def question(p: _FactPlan) -> str:
        return _fill(rng.choice(p.fk.question + GENERIC["question"]), p.svc, p.fk)

    def expect(fid: str, cp: Checkpoint, as_of: datetime | None = None) -> Expected:
        exp = expected_answer(by_fact[fid], cp.at, as_of)
        if exp.value:
            exp = exp.model_copy(update={"aliases": plan_of[fid].fk.aliases(exp.value)})
        return exp

    for cp in checkpoints:
        for u in users:
            pools: dict[ProbeType, list[tuple[str, datetime | None]]] = defaultdict(list)
            for fid, muts in by_fact.items():
                if plan_of[fid].fact.user_id != u:
                    continue
                known = [m for m in muts if m.at <= cp.at]
                if not known:
                    continue
                cut = max((i for i, m in enumerate(known) if m.kind is MutationKind.DELETE), default=-1)
                visible = known[cut + 1:]
                if known[-1].kind is MutationKind.DELETE:
                    pools[ProbeType.POST_DELETE].append((fid, None))
                    continue
                kinds = {m.kind for m in visible}
                if MutationKind.UPDATE in kinds:
                    pools[ProbeType.POST_UPDATE].append((fid, None))
                if MutationKind.CONTRADICT in kinds:
                    pools[ProbeType.CONTAMINATION].append((fid, None))
                if not kinds & {MutationKind.UPDATE, MutationKind.CONTRADICT}:
                    pools[ProbeType.CURRENT].append((fid, None))
                setters = [m for m in visible if m.kind in (MutationKind.INTRODUCE, MutationKind.UPDATE)]
                for a, b in zip(setters, setters[1:]):
                    gap = (b.at - a.at).total_seconds()
                    as_of = a.at + timedelta(seconds=int(gap * rng.uniform(0.2, 0.8)))
                    pools[ProbeType.HISTORICAL].append((fid, as_of))

            for ptype, pool in pools.items():
                for fid, as_of in rng.sample(pool, min(cfg.probes_per_type, len(pool))):
                    p = plan_of[fid]
                    if as_of is None:
                        q = question(p)
                    else:
                        date = f"{as_of:%B} {as_of.day}, {as_of.year}"
                        q = _fill(rng.choice(GENERIC["historical"]), p.svc, p.fk, date=date)
                    add(ptype, u, q, cp, expect(fid, cp, as_of), (fid,), as_of)

            # Abstention: a fact kind never stated for one of this user's services.
            svcs = sorted({p.svc for p in plan_of.values() if p.fact.user_id == u})
            have = {(p.svc, p.fk.attribute) for p in plan_of.values() if p.fact.user_id == u}
            unknown = [(s, fk) for s in svcs for fk in CATALOG if (s, fk.attribute) not in have]
            for svc, fk in rng.sample(unknown, min(cfg.probes_per_type, len(unknown))):
                q = _fill(rng.choice(fk.question + GENERIC["question"]), svc, fk)
                add(ProbeType.ABSTENTION, u, q, cp, Expected(kind=ExpectedKind.ABSTAIN))
    return probes


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def hygiene_errors(tl: Timeline) -> list[str]:
    """Text-level checks the schema cannot make: every event mentions only the
    values it carries, probe questions mention none, and no value of a user is
    a sub-phrase of another."""
    errors: list[str] = []
    fact_user = {f.fact_id: f.user_id for f in tl.facts}
    kind_of = {f.fact_id: next(k for k in CATALOG if k.attribute == f.attribute) for f in tl.facts}
    forms: dict[str, dict[str, tuple[str, ...]]] = defaultdict(dict)  # user -> value -> surface forms
    for m in tl.mutations:
        if m.value:
            forms[fact_user[m.fact_id]][m.value] = (m.value, *kind_of[m.fact_id].aliases(m.value))
    carried: dict[str, set[str]] = defaultdict(set)
    for m in tl.mutations:
        for eid in m.carrier_event_ids:
            if m.value:
                carried[eid].add(m.value)

    def mentioned(user: str, text: str) -> set[str]:
        return {v for v, fs in forms[user].items() if any(mentions(text, f) for f in fs)}

    for e in tl.events:
        extra = mentioned(e.user_id, e.content) - carried[e.event_id]
        if extra:
            errors.append(f"{e.event_id} mentions uncarried values {sorted(extra)}: {e.content!r}")
    for p in tl.probes:
        if mentioned(p.user_id, p.question):
            errors.append(f"{p.probe_id} question mentions a value: {p.question!r}")
    for u, vs in forms.items():
        for a in vs:
            for b in vs:
                if a != b and any(mentions(fa, fb) for fa in vs[a] for fb in vs[b]):
                    errors.append(f"{u}: value {b!r} is a sub-phrase of {a!r}")
    return errors


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("workloads/generated"))
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    tl = generate(Config(**{**cfg.__dict__, "seed": args.seed}))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{tl.workload_id}.json"
    tl.save(path)

    counts = defaultdict(int)
    for p in tl.probes:
        counts[p.probe_type.value] += 1
    muts = defaultdict(int)
    for m in tl.mutations:
        muts[m.kind.value] += 1
    print(f"wrote {path}")
    print(f"  {len({e.user_id for e in tl.events})} users, {len(tl.events)} events, {len(tl.facts)} facts, "
          f"{len(tl.mutations)} mutations, {len(tl.probes)} probes")
    print(f"  mutations: {dict(muts)}")
    print(f"  probes:    {dict(counts)}")


if __name__ == "__main__":
    main()
