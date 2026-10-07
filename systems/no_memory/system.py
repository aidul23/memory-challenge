"""Lower-bound baseline: stores nothing, never abstains. Also the smallest
example of implementing the interface."""

from __future__ import annotations

from typing import Any, Sequence

from interface.memory_system import (
    Event, MemorySystem, Query, RetrievalResult, StoreStats, WriteReport,
)


class NoMemory(MemorySystem):
    name = "no_memory"
    version = "1"

    def ingest(self, events: Sequence[Event]) -> WriteReport:
        return WriteReport(items_added=0)

    def retrieve(self, query: Query) -> RetrievalResult:
        return RetrievalResult()

    def stats(self, user_id: str | None = None) -> StoreStats:
        return StoreStats(user_id=user_id, n_items=0, storage_bytes=0)

    def reset(self) -> None:
        pass

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}
