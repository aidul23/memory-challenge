"""The ledger: a bitemporal, trust-ranked, provenance-tracked fact store.

Every value lives in a *slot* (as named by the extractor) as an ``Entry`` with
a validity interval and the events that stated it. The write path applies
four rules per claim, in stream order:

  restate     same value as the active entry: add the event to its provenance,
              no new item (I4).
  supersede   different value from a source at least as trusted as the active
              entry's: close the active entry at the claim's time and open a
              new one (I2). Closed entries stay queryable by ``as_of`` (I3).
  dispute     different value from a *less* trusted source (an old README, the
              agent's own echo): kept for audit, never retrieved (I5).
  introduce   no active entry: open one.

Deletion erases every entry and dispute of the affected slots recorded up to
the request, keeping only a value-free tombstone, and blocks the source events
so a replay cannot bring them back (I1). Retrieval returns the one entry valid
at ``as_of`` (or now) per matched slot, and abstains when there is none (I6).
All state is partitioned by ``user_id`` (I7).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from interface.memory_system import (
    Capabilities, DeleteReport, DeleteRequest, Event, EventKind, Evidence,
    MemoryItem, MemoryItemStatus, MemorySystem, Query, RetrievalResult,
    StoreStats, Usage, WriteReport,
)
from interface.tokens import count_tokens
from systems.ledger.extract import Claim, Extractor, RuleExtractor

#: Higher wins. A source may supersede a value only from equal or higher rank.
#: Tool output is as trusted as the user: it observes the world directly.
TRUST: dict[EventKind, int] = {
    EventKind.USER_MESSAGE: 3,
    EventKind.TOOL_RESULT: 3,
    EventKind.CODE_CHANGE: 3,
    EventKind.DOCUMENT: 2,
    EventKind.ASSISTANT_MESSAGE: 1,
    EventKind.TOOL_CALL: 1,
}


def _total(usages: Sequence[Usage]) -> Usage:
    return Usage(
        llm_calls=sum(u.llm_calls for u in usages),
        llm_input_tokens=sum(u.llm_input_tokens for u in usages),
        llm_output_tokens=sum(u.llm_output_tokens for u in usages),
        embedding_tokens=sum(u.embedding_tokens for u in usages),
        model_ids=tuple(dict.fromkeys(m for u in usages for m in u.model_ids)),
    )


@dataclass
class Entry:
    memory_id: str
    slot: str
    label: str
    value: str
    trust: int
    source_kind: str
    valid_from: datetime
    valid_to: datetime | None = None
    source_event_ids: list[str] = field(default_factory=list)

    @property
    def recorded_at(self) -> datetime:
        return self.valid_from

    def holds_at(self, t: datetime) -> bool:
        return self.valid_from <= t and (self.valid_to is None or t < self.valid_to)


@dataclass
class Tombstone:
    slot: str
    deleted_at: datetime
    request_id: str


@dataclass
class _UserLedger:
    entries: list[Entry] = field(default_factory=list)
    disputes: list[Entry] = field(default_factory=list)
    tombstones: list[Tombstone] = field(default_factory=list)
    blocked: set[str] = field(default_factory=set)
    """Event ids that may never be ingested again for this user."""

    def active(self, slot: str) -> Entry | None:
        return next((e for e in reversed(self.entries) if e.slot == slot and e.valid_to is None), None)


class Ledger(MemorySystem):
    name = "ledger"
    version = "1"
    capabilities = Capabilities(
        provenance=True, temporal_query=True, deletion=True,
        abstention=True, persistence=True, export=True,
    )

    def __init__(self, extractor: Extractor | None = None) -> None:
        self.extractor = extractor or RuleExtractor()
        self.reset()

    def reset(self) -> None:
        self._users: dict[str, _UserLedger] = {}
        self._seen: set[str] = set()
        self._next_id = 0

    def _user(self, user_id: str) -> _UserLedger:
        return self._users.setdefault(user_id, _UserLedger())

    # ---- write path ------------------------------------------------------

    def ingest(self, events: Sequence[Event]) -> WriteReport:
        added = updated = superseded = rejected = 0
        usages: list[Usage] = []
        for ev in events:
            if ev.event_id in self._user(ev.user_id).blocked:
                rejected += 1
                continue
            if ev.event_id in self._seen:
                continue
            self._seen.add(ev.event_id)
            claims, usage = self.extractor.claims(ev)
            usages.append(usage)
            for c in claims:
                outcome = self._apply(self._user(ev.user_id), ev, c)
                added += outcome in ("introduce", "supersede")
                updated += outcome == "restate"
                superseded += outcome == "supersede"
                rejected += outcome == "dispute"
        return WriteReport(
            usage=_total(usages), items_added=added, items_updated=updated,
            items_superseded=superseded, items_rejected=rejected,
        )

    def _apply(self, ul: _UserLedger, ev: Event, c: Claim) -> str:
        trust = TRUST[ev.kind]
        cur = ul.active(c.slot)
        if cur is not None and cur.value == c.value:
            if ev.event_id not in cur.source_event_ids:
                cur.source_event_ids.append(ev.event_id)
            cur.trust = max(cur.trust, trust)
            return "restate"
        new = Entry(
            memory_id=self._new_id(), slot=c.slot, label=c.label, value=c.value,
            trust=trust, source_kind=ev.kind.value, valid_from=ev.timestamp,
            source_event_ids=[ev.event_id],
        )
        if cur is not None and trust < cur.trust:
            ul.disputes.append(new)
            return "dispute"
        if cur is not None:
            cur.valid_to = ev.timestamp
        ul.entries.append(new)
        return "supersede" if cur is not None else "introduce"

    def _new_id(self) -> str:
        self._next_id += 1
        return f"L{self._next_id:06d}"

    def delete(self, request: DeleteRequest) -> DeleteReport:
        ul = self._user(request.user_id)
        sources = set(request.source_event_ids)
        slots = {e.slot for e in ul.entries + ul.disputes if sources & set(e.source_event_ids)}
        usage = Usage()
        if not slots:
            # Description-only request, or sources we never turned into entries.
            found, usage = self.extractor.slots_for(request.description)
            slots = set(found)

        def erased(e: Entry) -> bool:
            return e.slot in slots and e.valid_from <= request.requested_at

        gone = [e for e in ul.entries + ul.disputes if erased(e)]
        ul.entries = [e for e in ul.entries if not erased(e)]
        ul.disputes = [e for e in ul.disputes if not erased(e)]
        ul.blocked |= sources | {i for e in gone for i in e.source_event_ids}
        ul.tombstones += [Tombstone(s, request.requested_at, request.request_id) for s in sorted(slots)]
        return DeleteReport(usage=usage, items_deleted=len(gone))

    # ---- read path -------------------------------------------------------

    def retrieve(self, query: Query) -> RetrievalResult:
        slots, usage = self.extractor.slots_for(query.text)
        if not slots:
            return RetrievalResult(abstained=True, abstain_reason="no known slot matches", usage=usage)
        ul = self._users.get(query.user_id, _UserLedger())
        target = query.as_of or query.asked_at

        evidence: list[Evidence] = []
        used = 0
        for slot in slots:
            hit = next((e for e in ul.entries if e.slot == slot and e.valid_from <= query.asked_at
                        and e.holds_at(target)), None)
            if hit is None:
                continue
            ev = self._evidence(hit)
            cost = count_tokens(ev.content)
            if query.token_budget is not None and used + cost > query.token_budget:
                break
            evidence.append(ev)
            used += cost
        if not evidence:
            deleted = any(t.slot in slots and t.deleted_at <= query.asked_at for t in ul.tombstones)
            reason = "deleted at the user's request" if deleted else f"no value held at {target:%Y-%m-%d}"
            return RetrievalResult(abstained=True, abstain_reason=reason, usage=usage)
        return RetrievalResult(evidence=tuple(evidence), usage=usage)

    @staticmethod
    def _evidence(e: Entry) -> Evidence:
        until = f" until {e.valid_to:%Y-%m-%d}" if e.valid_to else ""
        return Evidence(
            content=f"{e.label}: {e.value} (since {e.valid_from:%Y-%m-%d}{until}; source: {e.source_kind})",
            memory_id=e.memory_id, source_event_ids=tuple(e.source_event_ids),
            valid_from=e.valid_from, valid_to=e.valid_to, recorded_at=e.recorded_at,
        )

    # ---- introspection and persistence ------------------------------------

    def stats(self, user_id: str | None = None) -> StoreStats:
        uls = [ul for u, ul in self._users.items() if user_id in (None, u)]
        entries = [e for ul in uls for e in ul.entries]
        return StoreStats(
            user_id=user_id,
            n_items=len(entries),
            storage_bytes=sum(len(e.value.encode("utf-8")) for e in entries),
            extra={
                "active": sum(e.valid_to is None for e in entries),
                "disputes": sum(len(ul.disputes) for ul in uls),
                "tombstones": sum(len(ul.tombstones) for ul in uls),
            },
        )

    def export_items(self, user_id: str) -> list[MemoryItem]:
        ul = self._users.get(user_id, _UserLedger())
        return [
            MemoryItem(
                memory_id=e.memory_id, user_id=user_id, content=f"{e.label}: {e.value}",
                status=MemoryItemStatus.ACTIVE if e.valid_to is None else MemoryItemStatus.SUPERSEDED,
                source_event_ids=tuple(e.source_event_ids), valid_from=e.valid_from,
                valid_to=e.valid_to, recorded_at=e.recorded_at,
            )
            for e in ul.entries
        ]

    def snapshot(self, path: Path) -> None:
        state = {
            "next_id": self._next_id,
            "seen": sorted(self._seen),
            "users": {u: {**asdict(ul), "blocked": sorted(ul.blocked)} for u, ul in self._users.items()},
        }
        path.write_text(json.dumps(state, default=datetime.isoformat, indent=1), encoding="utf-8")

    def restore(self, path: Path) -> None:
        state = json.loads(path.read_text(encoding="utf-8"))

        def entry(d: dict[str, Any]) -> Entry:
            d["valid_from"] = datetime.fromisoformat(d["valid_from"])
            d["valid_to"] = d["valid_to"] and datetime.fromisoformat(d["valid_to"])
            return Entry(**d)

        self.reset()
        self._next_id = state["next_id"]
        self._seen = set(state["seen"])
        for u, d in state["users"].items():
            self._users[u] = _UserLedger(
                entries=[entry(e) for e in d["entries"]],
                disputes=[entry(e) for e in d["disputes"]],
                tombstones=[Tombstone(t["slot"], datetime.fromisoformat(t["deleted_at"]), t["request_id"])
                            for t in d["tombstones"]],
                blocked=set(d["blocked"]),
            )

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name, "version": self.version,
            "extractor": self.extractor.config() if hasattr(self.extractor, "config")
            else {"name": self.extractor.name, "version": self.extractor.version},
            "trust": {k.value: v for k, v in TRUST.items()},
        }
