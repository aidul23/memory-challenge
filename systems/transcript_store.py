"""Shared storage for the transcript baselines (full replay, transcript RAG).

Every event is kept verbatim, keyed by ``event_id``, so re-ingesting an event
is a no-op (I4). An event is its own provenance, so deletion removes the
request's source events and remembers them, so a later replay cannot bring
them back (I1). Without ``source_event_ids`` these baselines cannot delete
anything: they have no way to act on a natural-language description.
"""

from __future__ import annotations

from typing import Sequence

from interface.memory_system import (
    Capabilities, DeleteReport, DeleteRequest, Event, Evidence, MemorySystem,
    StoreStats, WriteReport,
)


class TranscriptStore(MemorySystem):
    capabilities = Capabilities(provenance=True, deletion=True)

    def __init__(self) -> None:
        self._events: dict[str, Event] = {}
        self._deleted: set[str] = set()

    def ingest(self, events: Sequence[Event]) -> WriteReport:
        added = rejected = 0
        for e in events:
            if e.event_id in self._deleted:
                rejected += 1
            elif e.event_id not in self._events:
                self._events[e.event_id] = e
                added += 1
        return WriteReport(items_added=added, items_rejected=rejected)

    def delete(self, request: DeleteRequest) -> DeleteReport:
        n = 0
        for eid in request.source_event_ids:
            e = self._events.get(eid)
            if e is not None and e.user_id == request.user_id:
                del self._events[eid]
                self._deleted.add(eid)
                n += 1
        return DeleteReport(items_deleted=n)

    def stats(self, user_id: str | None = None) -> StoreStats:
        items = [e for e in self._events.values() if user_id in (None, e.user_id)]
        return StoreStats(
            user_id=user_id,
            n_items=len(items),
            storage_bytes=sum(len(e.content.encode("utf-8")) for e in items),
        )

    def reset(self) -> None:
        self._events.clear()
        self._deleted.clear()

    def _history(self, user_id: str) -> list[Event]:
        return sorted(
            (e for e in self._events.values() if e.user_id == user_id),
            key=lambda e: e.timestamp,
        )

    @staticmethod
    def _evidence(e: Event, score: float | None = None) -> Evidence:
        """Timestamp and event kind stay in the text so a reader can judge
        recency and trust."""
        return Evidence(
            content=f"[{e.timestamp:%Y-%m-%d %H:%M} {e.kind.value}] {e.content}",
            memory_id=e.event_id,
            source_event_ids=(e.event_id,),
            recorded_at=e.timestamp,
            score=score,
        )
