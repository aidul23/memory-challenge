"""Recall upper bound: returns the user's entire history, oldest first, and
leaves all reasoning to the reader. It ignores ``as_of`` and the token budget,
so it should be run with ``token_budget=None`` (see ``Query.token_budget``)."""

from __future__ import annotations

from typing import Any

from interface.memory_system import Query, RetrievalResult
from systems.transcript_store import TranscriptStore


class FullReplay(TranscriptStore):
    name = "full_replay"
    version = "1"

    def retrieve(self, query: Query) -> RetrievalResult:
        return RetrievalResult(
            evidence=tuple(self._evidence(e) for e in self._history(query.user_id))
        )

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "order": "chronological"}
