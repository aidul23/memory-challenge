"""Planted-fact timeline: the ground truth behind every workload.

A ``Timeline`` bundles an interaction stream (``events``) with a ground-truth
record of which facts the stream expresses and how they change over time
(``facts`` + ``mutations``), plus the questions asked at checkpoints
(``probes``). Because the truth is planted, we can score not only "right or
wrong" but *how* a system is wrong: stale, contaminated, resurrected after
deletion, or hallucinated.

Mutation semantics
------------------
INTRODUCE   fact first stated. Truth becomes ``value``.
UPDATE      the world changed. Truth becomes ``value``; the old value is
            superseded (C1 supersession).
RESTATE     same value, new wording. Truth unchanged. Tests deduplication.
CONTRADICT  a less trusted source (stale document, tool, or the agent itself)
            asserts a conflicting value. Truth unchanged. Tests contamination.
DELETE      the user asks to forget the fact. Every value stated before the
            deletion must become unretrievable, including for historical
            questions. A later INTRODUCE starts a fresh, visible history.

How probes map to invariants
----------------------------
I1 deletion       POST_DELETE probes: expect abstention; deleted values forbidden.
I2 supersession   POST_UPDATE probes: superseded values forbidden.
I3 temporal       HISTORICAL probes: ``as_of`` set; later values forbidden.
I4 deduplication  RESTATE mutations + store growth from ``stats()`` (no probe).
I5 contamination  CONTAMINATION probes: contradicting values forbidden.
I6 abstention     ABSTENTION probes: never-stated facts; expect abstention.
I7 isolation      runner-level check: probes for user B must never surface
                  user A's values (no per-probe field needed).

The consistency check at load time recomputes every single-fact probe's
expected answer from the mutations, so a hand-edited or generator-bugged
probe cannot silently corrupt results.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interface.memory_system import Event, EventKind


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------
# Sources and mutation kinds
# --------------------------------------------------------------------------


class SourceTrust(str, Enum):
    USER = "user"
    TOOL = "tool"
    DOCUMENT = "document"
    MODEL = "model"


#: The source of a mutation is fixed by the kind of event that carries it, so
#: trust is something a production system could know from the trace alone.
KIND_TO_SOURCE: dict[EventKind, SourceTrust] = {
    EventKind.USER_MESSAGE: SourceTrust.USER,
    EventKind.ASSISTANT_MESSAGE: SourceTrust.MODEL,
    EventKind.TOOL_CALL: SourceTrust.MODEL,
    EventKind.TOOL_RESULT: SourceTrust.TOOL,
    EventKind.DOCUMENT: SourceTrust.DOCUMENT,
    EventKind.CODE_CHANGE: SourceTrust.TOOL,
}


class MutationKind(str, Enum):
    INTRODUCE = "introduce"
    UPDATE = "update"
    RESTATE = "restate"
    CONTRADICT = "contradict"
    DELETE = "delete"


_VALUE_REQUIRED = {
    MutationKind.INTRODUCE,
    MutationKind.UPDATE,
    MutationKind.RESTATE,
    MutationKind.CONTRADICT,
}
_SETS_TRUTH = {MutationKind.INTRODUCE, MutationKind.UPDATE}


# --------------------------------------------------------------------------
# Facts and mutations
# --------------------------------------------------------------------------


class Fact(_Model):
    """A slot whose value the stream sets and changes. ``(user_id, entity,
    attribute)`` must be unique within a timeline."""

    fact_id: str
    user_id: str
    entity: str
    attribute: str
    category: str
    description: str = ""


class Mutation(_Model):
    mutation_id: str
    fact_id: str
    kind: MutationKind
    at: datetime
    """World time; must equal the timestamp of the first carrier event."""
    value: str | None = None
    source: SourceTrust
    carrier_event_ids: tuple[str, ...]
    """Events that express this mutation, so provenance can be scored."""

    @model_validator(mode="after")
    def _check(self) -> "Mutation":
        if self.kind in _VALUE_REQUIRED and not self.value:
            raise ValueError(f"{self.mutation_id}: {self.kind.value} needs a value")
        if self.kind is MutationKind.DELETE and self.value is not None:
            raise ValueError(f"{self.mutation_id}: DELETE carries no value")
        if self.kind is MutationKind.DELETE and self.source is not SourceTrust.USER:
            raise ValueError(f"{self.mutation_id}: deletion requests come from the user")
        if self.kind is MutationKind.CONTRADICT and self.source is SourceTrust.USER:
            raise ValueError(
                f"{self.mutation_id}: a user-stated new value is an UPDATE, not a CONTRADICT"
            )
        if not self.carrier_event_ids:
            raise ValueError(f"{self.mutation_id}: needs at least one carrier event")
        return self


# --------------------------------------------------------------------------
# Checkpoints and probes
# --------------------------------------------------------------------------


class Checkpoint(_Model):
    name: str
    fraction: float = Field(gt=0, le=1)
    """Share of the event stream ingested at this checkpoint."""
    at: datetime
    """Runner ingests every event with ``timestamp <= at``, then asks probes."""


class ProbeType(str, Enum):
    CURRENT = "current"
    POST_UPDATE = "post_update"
    HISTORICAL = "historical"
    POST_DELETE = "post_delete"
    CONTAMINATION = "contamination"
    ABSTENTION = "abstention"
    MULTI_FACT = "multi_fact"


class ExpectedKind(str, Enum):
    VALUE = "value"
    ABSTAIN = "abstain"


class Expected(_Model):
    kind: ExpectedKind
    value: str | None = None
    aliases: tuple[str, ...] = ()
    """Accepted alternative surface forms of ``value``."""
    forbidden: tuple[str, ...] = ()
    """Values whose appearance in evidence or answer counts as a typed error
    (stale, contaminated, resurrected)."""

    @model_validator(mode="after")
    def _check(self) -> "Expected":
        if self.kind is ExpectedKind.VALUE and not self.value:
            raise ValueError("VALUE expectation needs a value")
        if self.kind is ExpectedKind.ABSTAIN and self.value is not None:
            raise ValueError("ABSTAIN expectation carries no value")
        if self.value is not None and self.value in self.forbidden:
            raise ValueError("expected value cannot also be forbidden")
        return self


class Probe(_Model):
    probe_id: str
    user_id: str
    probe_type: ProbeType
    question: str
    checkpoint: str
    fact_ids: tuple[str, ...] = ()
    """Empty only for ABSTENTION probes about never-stated facts."""
    as_of: datetime | None = None
    expected: Expected


# --------------------------------------------------------------------------
# Ground-truth derivation
# --------------------------------------------------------------------------


def expected_answer(
    mutations: Sequence[Mutation],
    asked_at: datetime,
    as_of: datetime | None = None,
    aliases: Sequence[str] = (),
) -> Expected:
    """Derive the correct answer for one fact, asked at ``asked_at`` about the
    state at ``as_of`` (default: ``asked_at``)."""
    if as_of is not None and as_of > asked_at:
        raise ValueError("as_of cannot be later than asked_at")

    known = [m for m in sorted(mutations, key=lambda m: m.at) if m.at <= asked_at]
    last_delete = max(
        (i for i, m in enumerate(known) if m.kind is MutationKind.DELETE), default=-1
    )
    erased = known[: last_delete + 1]
    visible = known[last_delete + 1 :]

    target = as_of or asked_at
    truth: str | None = None
    for m in visible:
        if m.at > target:
            break
        if m.kind in _SETS_TRUTH:
            truth = m.value

    forbidden = {m.value for m in erased if m.value}
    forbidden |= {
        m.value
        for m in visible
        if m.kind in (_SETS_TRUTH | {MutationKind.CONTRADICT}) and m.value
    }
    forbidden.discard(truth)

    if truth is None:
        return Expected(kind=ExpectedKind.ABSTAIN, forbidden=tuple(sorted(forbidden)))
    return Expected(
        kind=ExpectedKind.VALUE,
        value=truth,
        aliases=tuple(aliases),
        forbidden=tuple(sorted(forbidden)),
    )


def _sequence_errors(fact_id: str, mutations: Sequence[Mutation]) -> list[str]:
    errors: list[str] = []
    active: str | None = None
    for m in mutations:
        tag = f"{fact_id}/{m.mutation_id}"
        if m.kind is MutationKind.INTRODUCE:
            if active is not None:
                errors.append(f"{tag}: INTRODUCE while fact is active (use UPDATE)")
            active = m.value
        elif m.kind is MutationKind.UPDATE:
            if active is None:
                errors.append(f"{tag}: UPDATE of an inactive fact")
            elif m.value == active:
                errors.append(f"{tag}: UPDATE to the same value (use RESTATE)")
            active = m.value
        elif m.kind is MutationKind.RESTATE:
            if active is None or m.value != active:
                errors.append(f"{tag}: RESTATE must repeat the current value")
        elif m.kind is MutationKind.CONTRADICT:
            if active is None:
                errors.append(f"{tag}: CONTRADICT of an inactive fact")
            elif m.value == active:
                errors.append(f"{tag}: CONTRADICT must conflict with the current value")
        elif m.kind is MutationKind.DELETE:
            if active is None:
                errors.append(f"{tag}: DELETE of an inactive fact")
            active = None
    return errors


def _duplicates(ids: Sequence[str]) -> list[str]:
    return [i for i, n in Counter(ids).items() if n > 1]


# --------------------------------------------------------------------------
# The timeline
# --------------------------------------------------------------------------


class Timeline(_Model):
    schema_version: Literal["0.1"] = "0.1"
    workload_id: str
    generator: str
    """Name and version of the generator that produced this file."""
    seed: int
    events: tuple[Event, ...]
    facts: tuple[Fact, ...]
    mutations: tuple[Mutation, ...]
    checkpoints: tuple[Checkpoint, ...]
    probes: tuple[Probe, ...]

    @model_validator(mode="after")
    def _validate(self) -> "Timeline":
        errors = self.consistency_errors()
        if errors:
            raise ValueError("inconsistent timeline:\n  " + "\n  ".join(errors))
        return self

    # ---- lookups ---------------------------------------------------------

    def mutations_for(self, fact_id: str) -> list[Mutation]:
        """Mutations of one fact, in time order (stable for ties)."""
        return sorted(
            (m for m in self.mutations if m.fact_id == fact_id), key=lambda m: m.at
        )

    def checkpoint(self, name: str) -> Checkpoint:
        for c in self.checkpoints:
            if c.name == name:
                return c
        raise KeyError(name)

    def expected_for(
        self, fact_id: str, asked_at: datetime, as_of: datetime | None = None
    ) -> Expected:
        return expected_answer(self.mutations_for(fact_id), asked_at, as_of)

    # ---- validation ------------------------------------------------------

    def consistency_errors(self) -> list[str]:
        errors: list[str] = []
        events = {e.event_id: e for e in self.events}
        facts = {f.fact_id: f for f in self.facts}
        cps = {c.name: c for c in self.checkpoints}

        for label, ids in [
            ("event", [e.event_id for e in self.events]),
            ("fact", [f.fact_id for f in self.facts]),
            ("mutation", [m.mutation_id for m in self.mutations]),
            ("checkpoint", [c.name for c in self.checkpoints]),
            ("probe", [p.probe_id for p in self.probes]),
        ]:
            for d in _duplicates(ids):
                errors.append(f"duplicate {label} id: {d}")

        keys = [(f.user_id, f.entity, f.attribute) for f in self.facts]
        for d in _duplicates(keys):
            errors.append(f"two facts share the slot {d}")

        stamps = [e.timestamp for e in self.events]
        if stamps != sorted(stamps):
            errors.append("events are not in chronological order")

        # mutations: references, provenance, per-fact sequence
        by_fact: dict[str, list[Mutation]] = defaultdict(list)
        for m in self.mutations:
            fact = facts.get(m.fact_id)
            if fact is None:
                errors.append(f"{m.mutation_id}: unknown fact {m.fact_id}")
                continue
            by_fact[m.fact_id].append(m)
            carriers = [events.get(eid) for eid in m.carrier_event_ids]
            if any(c is None for c in carriers):
                errors.append(f"{m.mutation_id}: unknown carrier event")
                continue
            first = min(carriers, key=lambda e: e.timestamp)
            if m.at != first.timestamp:
                errors.append(f"{m.mutation_id}: 'at' must equal first carrier's timestamp")
            if KIND_TO_SOURCE[first.kind] is not m.source:
                errors.append(
                    f"{m.mutation_id}: source {m.source.value} does not match "
                    f"carrier kind {first.kind.value}"
                )
            if any(c.user_id != fact.user_id for c in carriers):
                errors.append(f"{m.mutation_id}: carrier belongs to another user")
        for fact_id, muts in by_fact.items():
            errors += _sequence_errors(fact_id, sorted(muts, key=lambda m: m.at))

        # checkpoints
        ordered = sorted(self.checkpoints, key=lambda c: c.fraction)
        if [c.at for c in ordered] != sorted(c.at for c in ordered):
            errors.append("checkpoint times must increase with fraction")
        if self.events and any(c.at < self.events[0].timestamp for c in self.checkpoints):
            errors.append("a checkpoint precedes the first event")

        # probes
        for p in self.probes:
            errors += self._probe_errors(p, facts, cps)
        return errors

    def _probe_errors(
        self, p: Probe, facts: dict[str, Fact], cps: dict[str, Checkpoint]
    ) -> list[str]:
        tag = f"probe {p.probe_id}"
        cp = cps.get(p.checkpoint)
        if cp is None:
            return [f"{tag}: unknown checkpoint {p.checkpoint}"]
        errors: list[str] = []
        unknown = [f for f in p.fact_ids if f not in facts]
        if unknown:
            return [f"{tag}: unknown facts {unknown}"]
        if any(facts[f].user_id != p.user_id for f in p.fact_ids):
            errors.append(f"{tag}: refers to another user's fact")
        if p.as_of is not None and p.as_of > cp.at:
            errors.append(f"{tag}: as_of is after the checkpoint")

        t = p.probe_type
        if t is ProbeType.HISTORICAL and p.as_of is None:
            errors.append(f"{tag}: HISTORICAL probe needs as_of")
        if t in (ProbeType.POST_DELETE, ProbeType.ABSTENTION) and (
            p.expected.kind is not ExpectedKind.ABSTAIN
        ):
            errors.append(f"{tag}: {t.value} probe must expect abstention")
        if not p.fact_ids and t is not ProbeType.ABSTENTION:
            errors.append(f"{tag}: only ABSTENTION probes may have no facts")
        if t is ProbeType.MULTI_FACT and len(p.fact_ids) < 2:
            errors.append(f"{tag}: MULTI_FACT probe needs two or more facts")

        required = {
            ProbeType.POST_UPDATE: MutationKind.UPDATE,
            ProbeType.POST_DELETE: MutationKind.DELETE,
            ProbeType.CONTAMINATION: MutationKind.CONTRADICT,
        }.get(t)
        if required is not None and not any(
            m.kind is required and m.at <= cp.at
            for f in p.fact_ids
            for m in self.mutations_for(f)
        ):
            errors.append(f"{tag}: {t.value} probe but no {required.value} before checkpoint")

        # single-fact probes: recompute the expected answer from the mutations
        if len(p.fact_ids) == 1:
            derived = self.expected_for(p.fact_ids[0], cp.at, p.as_of)
            if (derived.kind, derived.value) != (p.expected.kind, p.expected.value):
                errors.append(
                    f"{tag}: expects {p.expected.kind.value}={p.expected.value!r} "
                    f"but ground truth is {derived.kind.value}={derived.value!r}"
                )
            missing = set(derived.forbidden) - set(p.expected.forbidden)
            if missing:
                errors.append(f"{tag}: forbidden list omits {sorted(missing)}")
        return errors

    # ---- I/O ---------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Timeline":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
