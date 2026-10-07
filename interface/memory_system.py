"""Common interface for every memory system under evaluation.

Every system -- the trivial baselines (no memory, full-history replay, RAG over
transcripts), third-party systems (Mem0, A-MEM, ...) and our ledger --
implements ``MemorySystem``. The runner and the invariant suite talk to systems
only through this interface, which is what makes the R2 comparison
("same model, prompts, tools, and resource budgets") fair.

Design rules
------------
1. Black-box first. The runner and the invariant suite need only ``ingest``,
   ``retrieve``, ``delete`` and ``stats``. White-box hooks (``export_items``,
   ``snapshot``/``restore``) are optional and declared in ``Capabilities``, so a
   missing feature is reported as "unsupported" in the R1 table instead of
   crashing the run.
2. Everything is scoped by ``user_id`` (isolation between users or agents, C3).
3. Systems report the LLM and embedding tokens they consume. The *runner*
   measures wall-clock latency, counts injected tokens with one fixed
   tokenizer, and converts tokens to dollars with one pinned price table.
   Systems never report their own latency or cost, so no system can flatter
   itself (R2).
4. All data types are pydantic models, so every call and result can be logged
   as JSON for auditing and re-scoring (R3).

Runner protocol (the contract systems may rely on)
--------------------------------------------------
* Events arrive in chronological order. The runner calls ``ingest`` once at the
  end of each session with that session's events.
* At a checkpoint, the runner ingests every event with
  ``timestamp <= checkpoint.at`` and then issues the probes as ``Query`` objects.
  A system therefore never sees an event from after ``query.asked_at``.
* Deletion requests are delivered through ``delete`` at their position in the
  stream, *in addition to* appearing as ordinary user messages in the events.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------
# Input: interaction traces
# --------------------------------------------------------------------------


class EventKind(str, Enum):
    """Heterogeneous trace types (C1)."""

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    DOCUMENT = "document"
    CODE_CHANGE = "code_change"


class Event(_Frozen):
    """One item of an interaction trace. ``event_id`` is globally unique and is
    what provenance points back to (R1)."""

    event_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    kind: EventKind
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------


class Usage(_Frozen):
    """Model consumption reported by a system for one call. The runner turns
    this into dollars with a pinned price table."""

    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    embedding_tokens: int = 0
    model_ids: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Write path
# --------------------------------------------------------------------------


class WriteReport(_Frozen):
    """Result of ``ingest``. Counts are ``None`` when a system cannot expose
    them; the runner then falls back to ``stats()`` deltas."""

    usage: Usage = Usage()
    items_added: int | None = None
    items_updated: int | None = None
    items_superseded: int | None = None
    items_rejected: int | None = None


# --------------------------------------------------------------------------
# Read path
# --------------------------------------------------------------------------


class Query(_Frozen):
    query_id: str
    user_id: str
    text: str
    asked_at: datetime
    as_of: datetime | None = None
    """Historical question: "what held at time t" (C2). ``None`` means now."""
    token_budget: int | None = 2000
    """Maximum injected evidence tokens, counted by the runner's tokenizer.
    ``None`` means unbounded (only for full-history replay). Overflow is
    truncated by the runner and logged as a budget violation."""


class Evidence(_Frozen):
    """One retrieved item. Provenance and validity fields are optional because
    baselines may not track them; R1 reports which systems do."""

    content: str
    memory_id: str | None = None
    source_event_ids: tuple[str, ...] = ()
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    recorded_at: datetime | None = None
    score: float | None = None


class RetrievalResult(_Frozen):
    evidence: tuple[Evidence, ...] = ()
    abstained: bool = False
    """True means the system asserts it has no reliable memory for this query
    (R1). A system without abstention support always returns False."""
    abstain_reason: str | None = None
    usage: Usage = Usage()

    @model_validator(mode="after")
    def _abstain_means_no_evidence(self) -> "RetrievalResult":
        if self.abstained and self.evidence:
            raise ValueError("an abstaining result must not carry evidence")
        return self


# --------------------------------------------------------------------------
# Deletion
# --------------------------------------------------------------------------


class DeleteRequest(_Frozen):
    request_id: str
    user_id: str
    requested_at: datetime
    description: str
    """What the user asked to forget, in natural language."""
    source_event_ids: tuple[str, ...] = ()
    """Events that carried the fact. Systems with provenance can delete by
    source; others must work from ``description``. Experiments may withhold
    this field to test description-only deletion."""


class DeleteReport(_Frozen):
    usage: Usage = Usage()
    items_deleted: int | None = None


# --------------------------------------------------------------------------
# Introspection
# --------------------------------------------------------------------------


class StoreStats(_Frozen):
    user_id: str | None = None
    n_items: int | None = None
    storage_bytes: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class MemoryItemStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class MemoryItem(_Frozen):
    """White-box view of one stored item, for systems that support export."""

    memory_id: str
    user_id: str
    content: str
    status: MemoryItemStatus = MemoryItemStatus.ACTIVE
    source_event_ids: tuple[str, ...] = ()
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    recorded_at: datetime | None = None


class Capabilities(_Frozen):
    """What a system claims to support. Claims are *checked* by the invariant
    suite, not trusted; the R1 table reports claimed vs. verified."""

    provenance: bool = False
    temporal_query: bool = False
    deletion: bool = False
    abstention: bool = False
    persistence: bool = False
    export: bool = False


class NotSupported(NotImplementedError):
    """Raised by optional methods a system does not implement."""


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------


class MemorySystem(ABC):
    name: str
    version: str
    capabilities: Capabilities = Capabilities()

    @abstractmethod
    def ingest(self, events: Sequence[Event]) -> WriteReport:
        """Ingest one session's events, in chronological order.

        Expected (and tested by invariant I4): re-ingesting an already seen
        ``event_id`` must not create new items, so a crash-and-replay does not
        duplicate memory.
        """

    @abstractmethod
    def retrieve(self, query: Query) -> RetrievalResult:
        """Return bounded evidence for ``query`` or abstain.

        Must use only memory belonging to ``query.user_id`` (invariant I7).
        Systems without ``temporal_query`` may ignore ``query.as_of``.
        """

    def delete(self, request: DeleteRequest) -> DeleteReport:
        """Remove the requested fact so it can never be retrieved again
        (invariant I1)."""
        raise NotSupported(f"{self.name} does not support deletion")

    @abstractmethod
    def stats(self, user_id: str | None = None) -> StoreStats:
        """Store size, for the storage-growth curves (R2, R3)."""

    def snapshot(self, path: Path) -> None:
        """Persist the full store to ``path`` (restart and recovery, R1)."""
        raise NotSupported(f"{self.name} does not support snapshots")

    def restore(self, path: Path) -> None:
        """Load a store written by ``snapshot``. After restore, retrieval must
        behave as before the snapshot."""
        raise NotSupported(f"{self.name} does not support restore")

    def export_items(self, user_id: str) -> list[MemoryItem]:
        """White-box dump of stored items, for duplicate and staleness audits."""
        raise NotSupported(f"{self.name} does not support export")

    @abstractmethod
    def reset(self) -> None:
        """Erase everything. Called between repeated runs."""

    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Every setting that affects behaviour (model ids and versions,
        thresholds, embedding model, chunk sizes, prompts or prompt hashes).
        Logged with every run for reproducibility (R3)."""
