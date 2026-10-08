"""Turning events into claims, and questions into slots.

The ledger core never reads free text itself: an ``Extractor`` maps each event
to zero or more ``Claim``s about named slots, and maps a question or a deletion
description to the slots it is about. Swapping the extractor (rules today, an
LLM later) leaves the ledger's temporal, trust and deletion logic untouched.

``RuleExtractor`` is a deterministic, model-free stand-in. Each ``SlotRule``
pairs a value pattern (for writes) with cue patterns (for questions and
deletions). Its slot keys are attribute names only, so it assumes a single
project per user; an LLM extractor should emit ``entity/attribute`` keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from interface.memory_system import Event, Usage


@dataclass(frozen=True)
class Claim:
    """One value asserted for a slot by one event."""

    slot: str
    value: str
    """Canonical form, used to tell a restatement from a change."""
    label: str
    """Human-readable slot name, shown in evidence."""


class Extractor(Protocol):
    name: str
    version: str

    def claims(self, event: Event) -> tuple[list[Claim], Usage]: ...

    def slots_for(self, text: str) -> tuple[list[str], Usage]:
        """Slots a question or deletion description refers to."""
        ...


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

    def claims(self, event: Event) -> tuple[list[Claim], Usage]:
        out = []
        for r in self.rules:
            m = r.value.search(event.content)
            if m:
                out.append(Claim(slot=r.slot, value=r.canon(m), label=r.label))
        return out, Usage()

    def slots_for(self, text: str) -> tuple[list[str], Usage]:
        return [r.slot for r in self.rules if r.cues.search(text)], Usage()

    def config(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version,
                "slots": [r.slot for r in self.rules]}
