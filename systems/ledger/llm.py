"""LLM extractor for the ledger, on any supported model provider.

Two structured-output calls, both given the user's known slots so keys and
value spellings stay stable across sessions:

* ``claims``: one call per ingest batch (a session). Returns every factual
  assertion about the user's projects, from any source -- the ledger, not the
  model, decides what to trust.
* ``slots_for``: one call per question or deletion description, choosing from
  the known slot keys (an ``enum`` in the schema, so the model cannot invent
  one). With no known slots it abstains without calling the model.

The provider (Anthropic, OpenAI, or an OpenAI-compatible endpoint) is a
``Backend`` from ``backends.py``, chosen from ``.env`` by default. Responses
are cached on disk, keyed by a hash of provider, model and full request, so a
rerun of the same workload is free and reproducible (R3). A cached result
reports the usage of the call that produced it, so costs stay comparable.
Output is re-checked here, since not every OpenAI-compatible endpoint
enforces the schema.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from interface.memory_system import Event, Usage
from systems.ledger.backends import ROOT, Backend, backend_from_env
from systems.ledger.extract import Claim

DEFAULT_CACHE = ROOT / ".cache" / "llm"

EXTRACT_SYSTEM = """\
You extract durable facts from a coding agent's interaction trace, for a memory
store. Each event is one message, tool call, tool result, document or code
change from one user's work on their software projects.

Extract every assertion of a concrete value for a lasting property of a
project: its configuration, infrastructure, dependencies, workflow, commands,
hosts, regions, owners and similar. Extract from every source, including
documents, tool output and the assistant's own messages, even when they
conflict with what is already known: a separate component decides what to
trust. Skip questions, requests to forget something, plans, chit-chat, and
mentions of a property without a value.

For each claim give:
- event_id: the id of the event that makes it.
- entity: the service, repository or component it is about, spelled as in the
  known slots when it is the same one (e.g. "payments-service").
- attribute: a short snake_case name for the property (e.g. "database",
  "deploy_branch"). Reuse a known slot's attribute when it is the same property.
- value: the value, without surrounding prose or quotes. If it is the same as a
  known current value written differently (e.g. "Postgres 14" vs
  "PostgreSQL 14"), use the known spelling exactly.
"""

SLOTS_SYSTEM = """\
You route a question, or a request to forget something, to the memory slots it
is about. Choose only from the listed slot keys. Return every slot the text
asks about, and an empty list when none of them matches: a slot about a
different service or a different property does not match.
"""

CLAIM_FIELDS = ("event_id", "entity", "attribute", "value")

CLAIMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "entity": {"type": "string"},
                    "attribute": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["event_id", "entity", "attribute", "value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


def slot_key(entity: str, attribute: str) -> str:
    norm = lambda s: re.sub(r"\s+", "-", s.strip().lower())
    return f"{norm(entity)}/{norm(attribute).replace('-', '_')}"


def _known_block(known: Mapping[str, str]) -> str:
    if not known:
        return "Known slots: none yet."
    return "Known slots (key: description):\n" + "\n".join(f"- {k}: {v}" for k, v in sorted(known.items()))


class LLMExtractor:
    name = "llm"
    version = "2"

    def __init__(self, backend: Backend | None = None, cache_dir: Path | None = DEFAULT_CACHE) -> None:
        self._backend, self.cache_dir = backend, cache_dir
        self.calls = self.cache_hits = self.refusals = 0

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = backend_from_env()
        return self._backend

    # ---- Extractor API ------------------------------------------------------

    def claims(self, events: Sequence[Event], known: Mapping[str, str] = {}) -> tuple[list[Claim], Usage]:
        if not events:
            return [], Usage()
        lines = [f"[{e.event_id}] {e.timestamp:%Y-%m-%d %H:%M} {e.kind.value}: {e.content}" for e in events]
        prompt = f"{_known_block(known)}\n\nEvents:\n" + "\n".join(lines)
        data, usage = self._call(EXTRACT_SYSTEM, prompt, CLAIMS_SCHEMA, "claims")
        ids = {e.event_id for e in events}
        out = []
        for c in data.get("claims", []) if isinstance(data.get("claims"), list) else []:
            if not (isinstance(c, dict) and all(isinstance(c.get(k), str) for k in CLAIM_FIELDS)):
                continue
            if c["event_id"] in ids and c["value"].strip() and c["entity"].strip() and c["attribute"].strip():
                key = slot_key(c["entity"], c["attribute"])
                label = f"{c['attribute'].strip().replace('_', ' ')} of {c['entity'].strip()}"
                out.append(Claim(event_id=c["event_id"], slot=key, value=c["value"].strip(), label=label))
        return out, usage

    def slots_for(self, text: str, known: Mapping[str, str] = {}) -> tuple[list[str], Usage]:
        if not known:
            return [], Usage()
        schema = {
            "type": "object",
            "properties": {"slots": {"type": "array", "items": {"type": "string", "enum": sorted(known)}}},
            "required": ["slots"],
            "additionalProperties": False,
        }
        data, usage = self._call(SLOTS_SYSTEM, f"{_known_block(known)}\n\nText: {text}", schema, "slots")
        slots = data.get("slots") if isinstance(data.get("slots"), list) else []
        return [s for s in dict.fromkeys(slots) if isinstance(s, str) and s in known], usage

    def config(self) -> dict[str, object]:
        digest = hashlib.sha256((EXTRACT_SYSTEM + SLOTS_SYSTEM).encode()).hexdigest()[:12]
        return {"name": self.name, "version": self.version, **self.backend.config(),
                "prompt_sha": digest, "calls": self.calls, "cache_hits": self.cache_hits,
                "refusals": self.refusals}

    # ---- caching --------------------------------------------------------------

    def _call(self, system: str, prompt: str, schema: dict[str, Any], name: str) -> tuple[dict[str, Any], Usage]:
        request = {"backend": self.backend.config(), "system": system, "prompt": prompt, "schema": schema}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            self.cache_hits += 1
            cached = json.loads(path.read_text(encoding="utf-8"))
            return cached["data"], Usage(**cached["usage"])

        self.calls += 1
        result = self.backend.complete_json(system, prompt, schema, name)
        if result.data is not None and not isinstance(result.data, dict):
            result = type(result)({}, result.usage)
        if result.data is None:
            # Not cached: a refusal is not a stable answer.
            self.refusals += 1
            return {}, result.usage
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"data": result.data, "usage": result.usage.model_dump()}), encoding="utf-8")
        return result.data, result.usage
