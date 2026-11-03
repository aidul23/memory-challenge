"""Mem0 (Chhikara et al., 2025) as a third-party baseline, through its OSS
library pinned at ``mem0ai==1.0.11``.

1.0.11 is the last release that implements the paper's algorithm: one LLM
call extracts facts from a session, each fact is embedded and its nearest
stored memories retrieved, and a second LLM call decides per fact to ADD,
UPDATE, DELETE or do nothing. From 2.0 the library only ever adds, which is
not the system the paper describes.

Mem0 runs unmodified, with its own prompts, a local Qdrant store and OpenAI
embeddings (``text-embedding-3-small``). Only its OpenAI clients are wrapped,
so that

* every chat and embedding call goes through an on-disk cache keyed by the
  full request, like the ledger's extractor, so a rerun is free and exact;
* the model, temperature 0 and sampling seed are the ones the ledger uses;
* usage is reported to the runner, which prices it (no self-reported cost);
* "Today's date is ..." in Mem0's prompts, which the library fixes from the
  wall clock at import, becomes the date of the session being ingested. That
  is the date the conversation took place, and it keeps cache keys stable.

Mapping the trace onto Mem0's two roles: assistant messages and tool calls
are ``assistant``; user messages, tool results, documents and code changes
are ``user``, each prefixed with its kind ("[document] ..."), since Mem0's
default extraction prompt reads facts from user messages only. A session is
one ``add`` call.

Mem0 has no event provenance, no validity intervals and no abstention. For
deletion with source events, the adapter deletes every memory that an ``add``
call over one of those events created or updated: session-level provenance,
the finest Mem0 exposes, and nearly exact here, since the generator writes
each mutation in its own session. Given only a description, the adapter does
nothing: Mem0 has no delete-by-description API, so the in-stream "forget ..."
message, which every system also ingests, is all it can act on, through its
own update step. ``as_of`` is ignored.

``profile="coding"`` gives Mem0 the domain knowledge the ledger's extractor
has, through Mem0's documented ``custom_fact_extraction_prompt``. The prompt
is built mechanically and frozen: Mem0's own user-memory extraction prompt
with only its "Types of Information to Remember" list (personal preferences,
health, travel, ...) replaced by the sentence defining what to extract in the
ledger's coding prompt, copied from it at runtime. Mem0's role, user-only
rule, few-shot examples and output format are kept, and nothing else from the
ledger's prompt is added (its trust and provenance instructions describe the
ledger, not Mem0). The default profile runs Mem0 untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Sequence

from interface.memory_system import (
    Capabilities, DeleteReport, DeleteRequest, Event, EventKind, Evidence, MemorySystem, Query,
    RetrievalResult, StoreStats, Usage, WriteReport,
)
from interface.tokens import count_tokens
from systems.ledger.backends import ROOT, SAMPLING_SEED

MEM0_VERSION = "1.0.11"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
DEFAULT_CACHE = ROOT / ".cache" / "mem0"
ASSISTANT = (EventKind.ASSISTANT_MESSAGE, EventKind.TOOL_CALL)
_TODAY = re.compile(r"Today's date is \d{4}-\d{2}-\d{2}")


def _import_mem0():
    # Read by mem0 at import: no telemetry leaves the machine.
    os.environ["MEM0_TELEMETRY"] = "False"
    import mem0

    if mem0.__version__ != MEM0_VERSION:
        raise RuntimeError(f"mem0ai {mem0.__version__} installed; this baseline needs {MEM0_VERSION}")
    from mem0 import Memory

    return Memory


class CachedOpenAI:
    """Stands in for the ``openai.OpenAI`` client inside Mem0's LLM and
    embedder: answers from the cache when it can, otherwise calls ``inner``
    and stores the response. Reports each call's usage (a cached call reports
    the usage of the call that produced it) to ``on_usage``."""

    def __init__(self, inner: Any, cache_dir: Path | None, on_usage: Callable[[Usage], None],
                 today: Callable[[], str | None] = lambda: None) -> None:
        self.inner, self.cache_dir, self.on_usage, self.today = inner, cache_dir, on_usage, today
        self.calls = self.cache_hits = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embed)

    def _chat(self, **params: Any):
        from openai.types.chat import ChatCompletion

        day = self.today()
        if day:
            params["messages"] = [{**m, "content": _TODAY.sub(f"Today's date is {day}", m["content"])}
                                  for m in params["messages"]]
        params |= {"temperature": 0.0, "seed": SAMPLING_SEED}
        data = self._cached("chat", params, lambda: self.inner.chat.completions.create(**params))
        resp = ChatCompletion.model_validate(data)
        u = resp.usage
        self.on_usage(Usage(llm_calls=1, llm_input_tokens=u.prompt_tokens if u else 0,
                            llm_output_tokens=u.completion_tokens if u else 0, model_ids=(resp.model,)))
        return resp

    def _embed(self, **params: Any):
        from openai.types import CreateEmbeddingResponse

        data = self._cached("embed", params, lambda: self.inner.embeddings.create(**params))
        resp = CreateEmbeddingResponse.model_validate(data)
        self.on_usage(Usage(embedding_tokens=resp.usage.total_tokens, model_ids=(resp.model,)))
        return resp

    def _cached(self, kind: str, params: dict[str, Any], call: Callable[[], Any]) -> dict[str, Any]:
        request = {"provider": "openai", "kind": kind, "params": params}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        self.calls += 1
        data = call().model_dump(mode="json")
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
        return data


def coding_extraction_prompt() -> str:
    """Mem0's user-memory extraction prompt with its list of personal fact
    types replaced by the ledger's definition of a coding fact."""
    _import_mem0()
    from mem0.configs.prompts import USER_MEMORY_EXTRACTION_PROMPT as base
    from systems.ledger.llm import EXTRACT_SYSTEM

    domain = re.search(r"Extract every assertion.*?and similar\.", EXTRACT_SYSTEM, re.S)
    section = re.search(r"Types of Information to Remember:\n.*?\n\n(?=Here are some few shot examples:)", base, re.S)
    if not (domain and section):
        raise RuntimeError("prompt layout changed; the coding profile cannot be built")
    sentence = " ".join(domain.group(0).split())
    return base.replace(section.group(0), f"Types of Information to Remember:\n\n{sentence}\n\n")


def _model_from_env() -> str:
    from systems.ledger.backends import backend_from_env

    backend = backend_from_env()
    if backend.provider != "openai":
        raise RuntimeError("the Mem0 baseline runs on OpenAI models: set LLM_PROVIDER=openai in .env")
    return backend.model


class Mem0System(MemorySystem):
    name = "mem0"
    version = "1"
    capabilities = Capabilities(deletion=True)

    def __init__(self, model: str | None = None, top_k: int = 10, cache_dir: Path | None = DEFAULT_CACHE,
                 client: Any = None, profile: Literal["default", "coding"] = "default") -> None:
        """``client`` replaces the OpenAI client for both chat and embeddings
        (tests); by default Mem0 builds its own from ``OPENAI_API_KEY``."""
        self.model, self.top_k, self.cache_dir, self._client = model, top_k, cache_dir, client
        self.profile = profile
        self.prompt = coding_extraction_prompt() if profile == "coding" else None
        self._mem = None
        self._dir: str | None = None
        self.reset()

    # ---- setup ---------------------------------------------------------------

    def _build(self) -> None:
        Memory = _import_mem0()
        self.model = self.model or _model_from_env()
        self._dir = tempfile.mkdtemp(prefix="mem0-")
        mem = Memory.from_config({
            "llm": {"provider": "openai", "config": {"model": self.model, "temperature": 0.0}},
            "embedder": {"provider": "openai", "config": {"model": EMBEDDING_MODEL}},
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": "mem0", "path": os.path.join(self._dir, "qdrant"),
                "embedding_model_dims": EMBEDDING_DIMS}},
            "history_db_path": os.path.join(self._dir, "history.db"),
            **({"custom_fact_extraction_prompt": self.prompt} if self.prompt else {}),
        })
        record = lambda u: self._usage.append(u)
        self._llm = CachedOpenAI(self._client or mem.llm.client, self.cache_dir, record, lambda: self._today)
        self._emb = CachedOpenAI(self._client or mem.embedding_model.client, self.cache_dir, record)
        mem.llm.client, mem.embedding_model.client = self._llm, self._emb
        self._mem = mem

    def _close(self) -> None:
        if self._mem is not None:
            self._mem.close()
            self._mem = None
        if self._dir:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

    def reset(self) -> None:
        self._close()
        self._seen: set[str] = set()
        self._blocked: set[str] = set()
        self._prov: dict[str, set[str]] = defaultdict(set)
        """memory id -> events of every ``add`` call that created or updated it"""
        self._session_at: dict[str, datetime] = {}
        self._users: set[str] = set()
        self._usage: list[Usage] = []
        self._today: str | None = None
        self._build()

    def __del__(self) -> None:
        try:
            self._close()
        except Exception:
            pass  # interpreter shutdown

    def _take_usage(self) -> Usage:
        from runner.replay import sum_usage

        used, self._usage = self._usage, []
        return sum_usage(used)

    # ---- write path ----------------------------------------------------------

    def ingest(self, events: Sequence[Event]) -> WriteReport:
        by_user: dict[str, list[Event]] = defaultdict(list)
        rejected = 0
        for e in events:
            if e.event_id in self._blocked:
                rejected += 1
            elif e.event_id not in self._seen:
                by_user[e.user_id].append(e)
        added = updated = 0
        for user_id, evs in by_user.items():
            self._users.add(user_id)
            self._seen |= {e.event_id for e in evs}
            at = max(e.timestamp for e in evs)
            self._today = f"{at:%Y-%m-%d}"
            messages = [{"role": "assistant" if e.kind in ASSISTANT else "user",
                         "content": f"[{e.kind.value}] {e.content}"} for e in evs]
            out = self._mem.add(messages, user_id=user_id, metadata={"session_at": at.isoformat()})
            for r in out.get("results", []):
                if r["event"] in ("ADD", "UPDATE"):
                    self._prov[r["id"]] |= {e.event_id for e in evs}
                    self._session_at[r["id"]] = at
                    added += r["event"] == "ADD"
                    updated += r["event"] == "UPDATE"
                elif r["event"] == "DELETE":
                    self._prov.pop(r["id"], None)
                    self._session_at.pop(r["id"], None)
            self._today = None
        return WriteReport(usage=self._take_usage(), items_added=added, items_updated=updated,
                           items_rejected=rejected)

    def delete(self, request: DeleteRequest) -> DeleteReport:
        sources = set(request.source_event_ids)
        doomed = [mid for mid, evs in self._prov.items() if evs & sources]
        for mid in doomed:
            self._mem.delete(mid)
            del self._prov[mid]
        self._blocked |= sources
        return DeleteReport(usage=self._take_usage(), items_deleted=len(doomed))

    # ---- read path -----------------------------------------------------------

    def retrieve(self, query: Query) -> RetrievalResult:
        hits = self._mem.search(query.text, user_id=query.user_id, limit=self.top_k)["results"]
        evidence: list[Evidence] = []
        used = 0
        for h in hits:
            at = self._session_at.get(h["id"])
            ev = Evidence(
                content=(f"[{at:%Y-%m-%d}] " if at else "") + h["memory"],
                memory_id=h["id"],
                source_event_ids=tuple(sorted(self._prov.get(h["id"], ()))),
                recorded_at=at,
                score=round(h["score"], 4) if h.get("score") is not None else None,
            )
            cost = count_tokens(ev.content)
            if query.token_budget is not None and used + cost > query.token_budget:
                break
            evidence.append(ev)
            used += cost
        return RetrievalResult(evidence=tuple(evidence), usage=self._take_usage())

    # ---- introspection -------------------------------------------------------

    def _memories(self, user_id: str) -> list[dict[str, Any]]:
        return self._mem.get_all(user_id=user_id, limit=1_000_000)["results"]

    def stats(self, user_id: str | None = None) -> StoreStats:
        mems = [m for u in ([user_id] if user_id else sorted(self._users)) for m in self._memories(u)]
        return StoreStats(user_id=user_id, n_items=len(mems),
                          storage_bytes=sum(len(m["memory"].encode("utf-8")) for m in mems),
                          extra={"vector_bytes": len(mems) * EMBEDDING_DIMS * 4})

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "mem0ai": MEM0_VERSION, "llm": self.model,
                "temperature": 0.0, "seed": SAMPLING_SEED, "embedder": EMBEDDING_MODEL,
                "vector_store": "qdrant-local", "top_k": self.top_k, "profile": self.profile,
                "prompt_sha": hashlib.sha256(self.prompt.encode()).hexdigest()[:12] if self.prompt else None,
                "calls": self._llm.calls + self._emb.calls,
                "cache_hits": self._llm.cache_hits + self._emb.cache_hits}
