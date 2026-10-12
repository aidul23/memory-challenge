"""Turning events into claims, and questions into slots.

The ledger core never reads free text itself: an ``Extractor`` maps a batch of
events to ``Claim``s about named slots, and maps a question or a deletion
description to the slots it is about. Both calls receive the user's ``known``
slots (key -> description, including the current value) so an extractor can
reuse slot keys and value spellings. Swapping the extractor (``RuleExtractor``
here, ``LLMExtractor`` in ``llm.py``) leaves the ledger's temporal, trust and
deletion logic untouched.

``RuleExtractor`` is a deterministic, model-free stand-in. Each ``SlotRule``
pairs a value pattern (for writes) with cue patterns (for questions and
deletions). Its slot keys are attribute names only, so it assumes a single
project per user; an LLM extractor should emit ``entity/attribute`` keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence

from interface.memory_system import Event, Usage


@dataclass(frozen=True)
class Claim:
    """One value asserted for a slot by one event."""

    event_id: str
    slot: str
    value: str
    """Canonical form, used to tell a restatement from a change."""
    label: str
    """Human-readable slot name, shown in evidence."""


class Extractor(Protocol):
    name: str
    version: str

    def claims(self, events: Sequence[Event], known: Mapping[str, str]) -> tuple[list[Claim], Usage]:
        """Claims made by ``events`` (one user, stream order)."""
        ...

    def slots_for(self, text: str, known: Mapping[str, str]) -> tuple[list[str], Usage]:
        """Which of the ``known`` slots a question or deletion description
        refers to."""
        ...

    def config(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class SlotRule:
    slot: str
    label: str
    value: re.Pattern[str]
    """First capture group (or a ``canon`` of all groups) is the value."""
    cues: re.Pattern[str]
    """Matches questions and deletion descriptions about this slot."""
    canon: Callable[[re.Match[str]], str] = field(default=lambda m: m.group(1))


_DB_NAMES = {"postgresql": "PostgreSQL", "postgres": "PostgreSQL", "pg": "PostgreSQL",
             "mysql": "MySQL", "mariadb": "MariaDB", "sqlite": "SQLite", "mongodb": "MongoDB"}

_I = re.IGNORECASE

DEFAULT_RULES: tuple[SlotRule, ...] = (
    SlotRule(
        slot="database",
        label="database",
        value=re.compile(r"\b(" + "|".join(_DB_NAMES) + r")\s*(\d+(?:\.\d+)?)\b", _I),
        cues=re.compile(r"\b(database|db|postgres\w*|mysql)\b", _I),
        canon=lambda m: f"{_DB_NAMES[m.group(1).lower()]} {m.group(2)}",
    ),
    SlotRule(
        slot="test_command",
        label="test command",
        value=re.compile(r"\btests?\b[^`]*`([^`]+)`", _I),
        cues=re.compile(r"\b(tests?|test suite)\b", _I),
    ),
    SlotRule(
        slot="deploy_branch",
        label="deploy branch",
        value=re.compile(r"\bdeploy\w*\b[^.`]*?\bfrom\s+(?:the\s+)?`?([A-Za-z][\w./-]*)`?", _I),
        cues=re.compile(r"\b(branch|deploy\w*)\b", _I),
    ),
    SlotRule(
        slot="staging_hostname",
        label="staging hostname",
        value=re.compile(r"\bstaging\b.*?\b([a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:internal|local|com|net|io|org))\b", _I),
        cues=re.compile(r"\b(staging|hostname|host)\b", _I),
    ),
)


class RuleExtractor:
    name = "rules"
    version = "1"

    def __init__(self, rules: Sequence[SlotRule] = DEFAULT_RULES) -> None:
        self.rules = tuple(rules)

    def claims(self, events: Sequence[Event], known: Mapping[str, str] = {}) -> tuple[list[Claim], Usage]:
        out = []
        for e in events:
            for r in self.rules:
                m = r.value.search(e.content)
                if m:
                    out.append(Claim(event_id=e.event_id, slot=r.slot, value=r.canon(m), label=r.label))
        return out, Usage()

    def slots_for(self, text: str, known: Mapping[str, str] = {}) -> tuple[list[str], Usage]:
        return [r.slot for r in self.rules if r.cues.search(text)], Usage()

    def config(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version,
                "slots": [r.slot for r in self.rules]}
