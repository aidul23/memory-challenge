"""The ledger with an episodic fallback: facts where it can be sure, raw
turns otherwise.

The ledger answers whenever it holds a value for a matched slot, so its
guarantees carry over: no stale or contaminated value, no future value for a
historical question, nothing deleted coming back. When it holds nothing (no
slot matches, the slot has no value at that time, or only an unconfirmed
claim exists) the question goes to BM25 over the raw turns instead, which
covers what the ledger does not model: one-off details buried in long chats,
what the assistant itself said, preferences, counts across sessions.

Deletion keeps both stores in step. From the episodes it removes every event
behind an erased ledger entry, every turn that mentions an erased value, and
every turn up to the request that names a deleted slot outright (entity and
attribute, as ``named_slots`` matches them), and every assistant or tool-call
turn up to the request that names an entity the request is about. The last two
cover what the ledger never extracted: with trusted-only extraction, an
assistant's guess ("Running `npm test` for inventory-service like last time")
is no ledger entry, and if the user's own statement was missed the request
matches no slot at all. Over-deleting the assistant's chatter about a service
is the safe direction for a deletion guarantee. What remains possible is a
trusted turn the extractor missed that names neither the value it states nor
the property; the evaluation counts these. A question the ledger abstains
on *because* of a deletion is not sent to the fallback.

``mode="augment"`` instead always appends episodes after the ledger's facts
(the facts come first and are marked as current), trading some evidence
cleanliness for recall on questions that need several turns.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence

from interface.memory_system import (
    Capabilities, DeleteReport, DeleteRequest, Event, EventKind, MemorySystem, Query,
    RetrievalResult, StoreStats, Usage, WriteReport,
)
from interface.tokens import count_tokens
from runner.scoring import mentions
from systems.ledger.extract import Extractor
from systems.ledger.system import Ledger, _total, named_slots
from systems.transcript_rag.system import TranscriptRAG

DELETED = "deleted at the user's request"
UNTRUSTED = (EventKind.ASSISTANT_MESSAGE, EventKind.TOOL_CALL)


def _names(text: str, entity: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(entity)}(?![\w-])", text.lower()) is not None


class LedgerHybrid(MemorySystem):
    name = "ledger_hybrid"
    version = "3"
    capabilities = Capabilities(
        provenance=True, temporal_query=True, deletion=True,
        abstention=True, persistence=True, export=True,
    )

    def __init__(self, extractor: Extractor | None = None, mode: Literal["fallback", "augment"] = "fallback",
                 top_k: int = 5, extract_min_trust: int | None = None) -> None:
        self.ledger = Ledger(extractor, extract_min_trust=extract_min_trust)
        self.episodes = TranscriptRAG(top_k=top_k)
        self.mode = mode

    def reset(self) -> None:
        self.ledger.reset()
        self.episodes.reset()

    def ingest(self, events: Sequence[Event]) -> WriteReport:
        rep = self.ledger.ingest(events)
        ep = self.episodes.ingest([e for e in events if e.event_id not in self.ledger._user(e.user_id).blocked])
        return rep.model_copy(update={"items_added": (rep.items_added or 0) + (ep.items_added or 0)})

    def delete(self, request: DeleteRequest) -> DeleteReport:
        ul = self.ledger._user(request.user_id)
        before = {id(e): e for e in ul.entries + ul.disputes + ul.pending}
        rep = self.ledger.delete(request)
        after = {id(e) for e in ul.entries + ul.disputes + ul.pending}
        erased = [e for k, e in before.items() if k not in after]
        values = {e.value for e in erased}
        slots = sorted({t.slot for t in ul.tombstones if t.request_id == request.request_id})
        known_entities = {k.split("/", 1)[0] for k in self.ledger._known(ul) if "/" in k}
        entities = {s.split("/", 1)[0] for s in slots if "/" in s}
        entities |= {x for x in known_entities if _names(request.description, x)}
        ids = set(request.source_event_ids) | {i for e in erased for i in e.source_event_ids}
        for e in self.episodes._history(request.user_id):
            before_request = e.timestamp <= request.requested_at
            if (any(mentions(e.content, v) for v in values)
                    or (before_request and named_slots(e.content, slots))
                    or (before_request and e.kind in UNTRUSTED and any(_names(e.content, x) for x in entities))):
                ids.add(e.event_id)
        ep = self.episodes.delete(request.model_copy(update={"source_event_ids": tuple(sorted(ids))}))
        ul.blocked |= ids
        return rep.model_copy(update={"items_deleted": (rep.items_deleted or 0) + (ep.items_deleted or 0)})

    def retrieve(self, query: Query) -> RetrievalResult:
        facts = self.ledger.retrieve(query)
        if facts.evidence and self.mode == "fallback":
            return facts
        if facts.abstained and facts.abstain_reason == DELETED:
            return facts
        used = sum(count_tokens(e.content) for e in facts.evidence)
        budget = None if query.token_budget is None else max(query.token_budget - used, 0)
        eps = self.episodes.retrieve(query.model_copy(update={"token_budget": budget}))
        evidence = facts.evidence + eps.evidence
        usage = _total([facts.usage, eps.usage])
        if not evidence:
            return RetrievalResult(abstained=True, abstain_reason=facts.abstain_reason or "nothing relevant",
                                   usage=usage)
        return RetrievalResult(evidence=evidence, usage=usage)

    def stats(self, user_id: str | None = None) -> StoreStats:
        a, b = self.ledger.stats(user_id), self.episodes.stats(user_id)
        return StoreStats(user_id=user_id, n_items=a.n_items + b.n_items,
                          storage_bytes=a.storage_bytes + b.storage_bytes,
                          extra={**a.extra, "episodes": b.n_items})

    def snapshot(self, path: Path) -> None:
        state = {"ledger": self.ledger.state(), "episodes": self.episodes.state()}
        path.write_text(json.dumps(state, default=datetime.isoformat), encoding="utf-8")

    def restore(self, path: Path) -> None:
        state = json.loads(path.read_text(encoding="utf-8"))
        self.ledger.load_state(state["ledger"])
        self.episodes.load_state(state["episodes"])

    def export_items(self, user_id: str):
        return self.ledger.export_items(user_id)

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "mode": self.mode,
                "ledger": self.ledger.config(), "episodes": self.episodes.config()}
