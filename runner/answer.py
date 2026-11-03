"""The shared answerer: one pinned model, one prompt, for every system.

Evidence-level scoring asks whether the right value reached the context. The
answerer asks whether it helps the task: the same model, prompt and settings
turn each system's (budget-truncated) evidence into an answer, which is
scored against the same ground truth. It runs for every probe, including when
a system returned nothing or abstained, since an agent would still be asked
the question; it is told the system's abstention reason, if any.

The answer is structured (``answer``, ``abstain``), so abstention is read
from a field, not guessed from wording, and scoring stays deterministic: no
LLM judge. Responses are cached like the extractor's.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

from interface.memory_system import Evidence, Usage
from systems.ledger.backends import ROOT, Backend, backend_from_env
from systems.ledger.llm import cached_json

DEFAULT_CACHE = ROOT / ".cache" / "answer"

ANSWER_SYSTEM = """\
You are a coding agent's assistant. Answer the user's question about their
software projects using only the memory provided below the question; you
have no other knowledge of their projects. The memory may be empty,
incomplete, or contain outdated or conflicting statements: when statements
conflict, prefer what the user or a tool most recently stated over guesses,
and for a question about a past date, answer what held on that date.

Give the value only (e.g. "PostgreSQL 16", "staging-7.internal"), with no
explanation. If the memory does not establish the answer, set abstain to
true and leave answer empty; do not guess.
"""

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "abstain": {"type": "boolean"}},
    "required": ["answer", "abstain"],
    "additionalProperties": False,
}


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    abstained: bool


def answer_prompt(question: str, asked_at: datetime, evidence: Sequence[Evidence],
                  abstain_reason: str | None = None) -> str:
    if evidence:
        memory = "\n".join(f"- {e.content}" for e in evidence)
    else:
        memory = "(nothing)" + (f" The memory system reported: {abstain_reason}." if abstain_reason else "")
    return f"Today is {asked_at:%Y-%m-%d}.\n\nQuestion: {question}\n\nMemory:\n{memory}"


class Answerer:
    version = "1"

    def __init__(self, backend: Backend | None = None, cache_dir: Path | None = DEFAULT_CACHE) -> None:
        self._backend, self.cache_dir = backend, cache_dir
        self.calls = self.cache_hits = self.refusals = 0

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = backend_from_env()
        return self._backend

    def answer(self, question: str, asked_at: datetime, evidence: Sequence[Evidence],
               abstain_reason: str | None = None) -> tuple[Answer, Usage]:
        prompt = answer_prompt(question, asked_at, evidence, abstain_reason)
        data, usage, outcome = cached_json(self.backend, self.cache_dir, ANSWER_SYSTEM, prompt,
                                           ANSWER_SCHEMA, "answer")
        self.calls += outcome != "hit"
        self.cache_hits += outcome == "hit"
        self.refusals += outcome == "refused"
        text = data.get("answer") if isinstance(data.get("answer"), str) else ""
        abstained = data.get("abstain") is True or not text.strip()
        return Answer(text="" if abstained else text.strip(), abstained=abstained), usage

    def config(self) -> dict[str, object]:
        return {"version": self.version, **self.backend.config(),
                "prompt_sha": hashlib.sha256(ANSWER_SYSTEM.encode()).hexdigest()[:12],
                "calls": self.calls, "cache_hits": self.cache_hits, "refusals": self.refusals}
