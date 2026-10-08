"""Replays a timeline against one memory system and scores every probe.

Protocol (the contract in ``interface/memory_system.py``):

* Events are fed in timeline order. A session's events go to ``ingest`` in one
  call when its last event has arrived. A checkpoint or deletion inside a
  session flushes what has arrived so far; the rest of the session follows in
  a later call.
* Each DELETE mutation becomes a ``DeleteRequest`` delivered right after its
  carrier event (the user's "please forget ...") has been ingested. Its
  ``source_event_ids`` are the events that stated the fact before the
  deletion; ``delete_with_sources=False`` withholds them.
* At each checkpoint every event with ``timestamp <= at`` has been ingested,
  and each probe becomes a ``Query`` with ``asked_at = checkpoint.at``.
* The runner, not the system, measures latency, counts injected tokens with
  ``interface.tokens``, and truncates evidence that exceeds the budget (whole
  items, in the system's order). Scoring sees only the truncated evidence.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict

from interface.memory_system import (
    Capabilities, DeleteRequest, Event, Evidence, MemorySystem, NotSupported,
    Query, StoreStats, Usage,
)
from interface.tokens import TOKENIZER, count_tokens
from runner.scoring import ForbiddenHit, Verdict, explain_forbidden, score
from workloads.timeline import (
    Checkpoint, Expected, MutationKind, Probe, ProbeType, Timeline,
)

T = TypeVar("T")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class OpRecord(_Frozen):
    """One write-path call (``ingest`` or ``delete``)."""

    op: Literal["ingest", "delete"]
    at: datetime
    """Stream time of the call."""
    ref: str
    """Session id for ingest, request id for delete."""
    event_ids: tuple[str, ...] = ()
    latency_s: float = 0.0
    usage: Usage = Usage()
    report: dict[str, Any] = {}
    error: str | None = None


class ProbeResult(_Frozen):
    probe_id: str
    probe_type: ProbeType
    checkpoint: str
    question: str
    asked_at: datetime
    as_of: datetime | None
    expected: Expected
    abstained: bool
    abstain_reason: str | None
    evidence: tuple[Evidence, ...]
    """What would be injected, after budget truncation."""
    evidence_tokens: int
    """Tokens the system returned, before truncation."""
    injected_tokens: int
    budget_exceeded: bool
    verdict: Verdict
    matched: str | None
    forbidden_hits: tuple[ForbiddenHit, ...]
    foreign_hits: tuple[str, ...]
    latency_s: float
    usage: Usage


class RunSettings(_Frozen):
    token_budget: int | None
    delete_with_sources: bool
    tokenizer: str = TOKENIZER


class RunResult(_Frozen):
    workload_id: str
    generator: str
    seed: int
    system: str
    system_version: str
    system_config: dict[str, Any]
    capabilities: Capabilities
    settings: RunSettings
    started_at: datetime
    ops: tuple[OpRecord, ...]
    probes: tuple[ProbeResult, ...]
    checkpoint_stats: dict[str, StoreStats]
    """Whole-store size after each checkpoint, for storage-growth curves."""

    def summary(self) -> dict[str, Any]:
        verdicts = Counter(p.verdict.value for p in self.probes)
        errors = Counter(
            t.value
            for p in self.probes
            for t in {t for h in p.forbidden_hits for t in h.error_types}
        )
        latencies = [p.latency_s for p in self.probes]
        last = list(self.checkpoint_stats.values())[-1] if self.checkpoint_stats else None
        return {
            "system": self.system,
            "probes": len(self.probes),
            "correct": verdicts.get(Verdict.CORRECT.value, 0),
            "verdicts": dict(verdicts),
            "error_types": dict(errors),
            "budget_violations": sum(p.budget_exceeded for p in self.probes),
            "isolation_violations": sum(bool(p.foreign_hits) for p in self.probes),
            "unsupported": sorted({o.op for o in self.ops if o.error}),
            "injected_tokens": sum(p.injected_tokens for p in self.probes),
            "usage": sum_usage([o.usage for o in self.ops] + [p.usage for p in self.probes]).model_dump(),
            "ingest_s": sum(o.latency_s for o in self.ops if o.op == "ingest"),
            "mean_retrieve_ms": 1000 * sum(latencies) / len(latencies) if latencies else 0.0,
            "final_items": last.n_items if last else None,
        }


def sum_usage(usages: Sequence[Usage]) -> Usage:
    models: list[str] = []
    for u in usages:
        models += [m for m in u.model_ids if m not in models]
    return Usage(
        llm_calls=sum(u.llm_calls for u in usages),
        llm_input_tokens=sum(u.llm_input_tokens for u in usages),
        llm_output_tokens=sum(u.llm_output_tokens for u in usages),
        embedding_tokens=sum(u.embedding_tokens for u in usages),
        model_ids=tuple(models),
    )


def _timed(fn: Callable[..., T], *args: Any) -> tuple[T, float]:
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def delete_requests(timeline: Timeline, with_sources: bool = True) -> dict[str, list[DeleteRequest]]:
    """DELETE mutations as requests, keyed by the event after which each is
    delivered (its first carrier)."""
    events = {e.event_id: e for e in timeline.events}
    order = {e.event_id: i for i, e in enumerate(timeline.events)}
    facts = {f.fact_id: f for f in timeline.facts}
    out: dict[str, list[DeleteRequest]] = defaultdict(list)
    for fid in facts:
        stated: list[str] = []
        for m in timeline.mutations_for(fid):
            if m.kind is not MutationKind.DELETE:
                stated += [e for e in m.carrier_event_ids if m.value and e not in stated]
                continue
            trigger = min(m.carrier_event_ids, key=order.__getitem__)
            out[trigger].append(DeleteRequest(
                request_id=m.mutation_id,
                user_id=facts[fid].user_id,
                requested_at=m.at,
                description=events[trigger].content,
                source_event_ids=tuple(sorted(stated, key=order.__getitem__)) if with_sources else (),
            ))
    return out


def _fit(evidence: Sequence[Evidence], budget: int | None) -> tuple[tuple[Evidence, ...], int, int]:
    costs = [count_tokens(ev.content) for ev in evidence]
    total = sum(costs)
    if budget is None or total <= budget:
        return tuple(evidence), total, total
    kept, used = [], 0
    for ev, c in zip(evidence, costs):
        if used + c > budget:
            break
        kept.append(ev)
        used += c
    return tuple(kept), total, used


def run(
    system: MemorySystem,
    timeline: Timeline,
    *,
    token_budget: int | None = 2000,
    delete_with_sources: bool = True,
) -> RunResult:
    system.reset()
    started = datetime.now(timezone.utc)
    events = timeline.events

    last_of = {(e.user_id, e.session_id): i for i, e in enumerate(events)}
    deletes = delete_requests(timeline, delete_with_sources)
    checkpoints = sorted(timeline.checkpoints, key=lambda c: c.at)
    probes_at: dict[str, list[Probe]] = defaultdict(list)
    for p in timeline.probes:
        probes_at[p.checkpoint].append(p)

    # Surface forms for every value (forbidden ones included), and the values
    # each user must never see from other users (I7).
    aliases: dict[str, list[str]] = defaultdict(list)
    for p in timeline.probes:
        if p.expected.value:
            aliases[p.expected.value] += [a for a in p.expected.aliases if a not in aliases[p.expected.value]]
    fact_user = {f.fact_id: f.user_id for f in timeline.facts}
    values_of: dict[str, set[str]] = defaultdict(set)
    for m in timeline.mutations:
        if m.value:
            values_of[fact_user[m.fact_id]].add(m.value)
    foreign = {
        u: sorted(set().union(*(v for o, v in values_of.items() if o != u)) - values_of[u])
        for u in {f.user_id for f in timeline.facts} | {p.user_id for p in timeline.probes}
    }

    ops: list[OpRecord] = []
    results: list[ProbeResult] = []
    cp_stats: dict[str, StoreStats] = {}
    pending: dict[tuple[str, str], list[Event]] = {}

    def flush(key: tuple[str, str]) -> None:
        batch = pending.pop(key)
        report, dt = _timed(system.ingest, batch)
        ops.append(OpRecord(
            op="ingest", at=batch[-1].timestamp, ref=key[1],
            event_ids=tuple(e.event_id for e in batch), latency_s=dt,
            usage=report.usage, report=report.model_dump(exclude={"usage"}),
        ))

    def flush_all() -> None:
        for key in sorted(pending, key=lambda k: pending[k][0].timestamp):
            flush(key)

    def deliver(req: DeleteRequest) -> None:
        try:
            report, dt = _timed(system.delete, req)
        except NotSupported as exc:
            ops.append(OpRecord(op="delete", at=req.requested_at, ref=req.request_id,
                                event_ids=req.source_event_ids, error=f"unsupported: {exc}"))
            return
        ops.append(OpRecord(
            op="delete", at=req.requested_at, ref=req.request_id,
            event_ids=req.source_event_ids, latency_s=dt, usage=report.usage,
            report=report.model_dump(exclude={"usage"}),
        ))

    def ask(cp: Checkpoint) -> None:
        flush_all()
        for p in probes_at[cp.name]:
            query = Query(
                query_id=p.probe_id, user_id=p.user_id, text=p.question,
                asked_at=cp.at, as_of=p.as_of, token_budget=token_budget,
            )
            res, dt = _timed(system.retrieve, query)
            evidence, total, used = _fit(res.evidence, token_budget)
            sc = score(
                p.expected, evidence, res.abstained,
                explain_forbidden(timeline, p.fact_ids, cp.at, p.as_of),
                aliases, foreign[p.user_id],
            )
            results.append(ProbeResult(
                probe_id=p.probe_id, probe_type=p.probe_type, checkpoint=cp.name,
                question=p.question, asked_at=cp.at, as_of=p.as_of, expected=p.expected,
                abstained=res.abstained, abstain_reason=res.abstain_reason,
                evidence=evidence, evidence_tokens=total, injected_tokens=used,
                budget_exceeded=token_budget is not None and total > token_budget,
                verdict=sc.verdict, matched=sc.matched, forbidden_hits=sc.forbidden_hits,
                foreign_hits=sc.foreign_hits, latency_s=dt, usage=res.usage,
            ))
        cp_stats[cp.name] = system.stats()

    ci = 0
    for i, e in enumerate(events):
        while ci < len(checkpoints) and checkpoints[ci].at < e.timestamp:
            ask(checkpoints[ci])
            ci += 1
        key = (e.user_id, e.session_id)
        pending.setdefault(key, []).append(e)
        if last_of[key] == i:
            flush(key)
        for req in deletes.get(e.event_id, ()):
            flush_all()
            deliver(req)
    for cp in checkpoints[ci:]:
        ask(cp)
    flush_all()

    return RunResult(
        workload_id=timeline.workload_id, generator=timeline.generator, seed=timeline.seed,
        system=system.name, system_version=system.version, system_config=system.config(),
        capabilities=system.capabilities,
        settings=RunSettings(token_budget=token_budget, delete_with_sources=delete_with_sources),
        started_at=started, ops=tuple(ops), probes=tuple(results), checkpoint_stats=cp_stats,
    )
