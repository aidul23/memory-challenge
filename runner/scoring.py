"""Evidence-level scoring against planted ground truth.

A probe is scored on the evidence the system would inject (after the runner's
budget truncation), not on a generated answer: a value counts as surfaced when
any evidence item mentions it. Every forbidden value found is labelled with
*why* it is forbidden, derived from the fact's mutations:

  stale         superseded by a later UPDATE (I2)
  future        set after ``as_of`` in a historical question (I3)
  contaminated  asserted by a less trusted CONTRADICT (I5)
  resurrected   stated before a DELETE (I1)
  forbidden     listed by the probe but not explained by its facts'
                mutations (hand-written multi-fact expectations)
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from interface.memory_system import Evidence
from workloads.timeline import Expected, ExpectedKind, MutationKind, Timeline


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Verdict(str, Enum):
    CORRECT = "correct"                    # value surfaced cleanly, or abstained when it should
    POLLUTED = "polluted"                  # value surfaced next to a forbidden one
    WRONG = "wrong"                        # only forbidden values surfaced
    MISSED = "missed"                      # neither value nor forbidden surfaced
    FALSE_ABSTENTION = "false_abstention"  # abstained although the fact is known
    LEAKED = "leaked"                      # should abstain; a forbidden value surfaced
    NO_ABSTENTION = "no_abstention"        # should abstain; did not, nothing forbidden surfaced


class ErrorType(str, Enum):
    STALE = "stale"
    FUTURE = "future"
    CONTAMINATED = "contaminated"
    RESURRECTED = "resurrected"
    FORBIDDEN = "forbidden"


class ForbiddenHit(_Frozen):
    value: str
    error_types: tuple[ErrorType, ...]
    source_event_ids: tuple[str, ...]
    """Provenance of the evidence items that mentioned the value."""


class Score(_Frozen):
    verdict: Verdict
    matched: str | None = None
    forbidden_hits: tuple[ForbiddenHit, ...] = ()
    foreign_hits: tuple[str, ...] = ()
    """Values of other users' facts that surfaced (I7)."""


def mentions(text: str, phrase: str) -> bool:
    """Case-insensitive whole-phrase match, so ``main`` does not match
    ``maintain`` and whitespace differences are ignored."""
    words = phrase.split()
    if not words:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


def explain_forbidden(
    timeline: Timeline,
    fact_ids: Sequence[str],
    asked_at: datetime,
    as_of: datetime | None = None,
) -> dict[str, set[ErrorType]]:
    """Map every value the facts ever held to the reasons it could be wrong at
    ``asked_at`` / ``as_of``. Mirrors ``workloads.timeline.expected_answer``."""
    target = as_of or asked_at
    reasons: dict[str, set[ErrorType]] = defaultdict(set)
    for fid in fact_ids:
        known = [m for m in timeline.mutations_for(fid) if m.at <= asked_at]
        cut = max((i for i, m in enumerate(known) if m.kind is MutationKind.DELETE), default=-1)
        for m in known[: cut + 1]:
            if m.value:
                reasons[m.value].add(ErrorType.RESURRECTED)
        for m in known[cut + 1 :]:
            if m.kind is MutationKind.CONTRADICT:
                reasons[m.value].add(ErrorType.CONTAMINATED)
            elif m.kind in (MutationKind.INTRODUCE, MutationKind.UPDATE):
                reasons[m.value].add(ErrorType.FUTURE if m.at > target else ErrorType.STALE)
    return reasons


def _citing(evidence: Sequence[Evidence], forms: Sequence[str]) -> tuple[str, ...]:
    ids: list[str] = []
    for ev in evidence:
        if any(mentions(ev.content, f) for f in forms):
            for i in ev.source_event_ids or ((ev.memory_id,) if ev.memory_id else ("?",)):
                if i not in ids:
                    ids.append(i)
    return tuple(ids)


def score(
    expected: Expected,
    evidence: Sequence[Evidence],
    abstained: bool,
    reasons: Mapping[str, set[ErrorType]],
    aliases: Mapping[str, Sequence[str]] = {},
    foreign_values: Sequence[str] = (),
) -> Score:
    """``aliases`` gives surface forms for *forbidden* values too, which the
    probe's own ``Expected`` does not carry."""
    text = [ev.content for ev in evidence]

    matched = None
    if expected.value is not None:
        for form in (expected.value, *expected.aliases, *aliases.get(expected.value, ())):
            if any(mentions(t, form) for t in text):
                matched = form
                break

    hits = []
    for value in expected.forbidden:
        ids = _citing(evidence, (value, *aliases.get(value, ())))
        if ids:
            types = tuple(sorted(reasons.get(value) or {ErrorType.FORBIDDEN}, key=lambda t: t.value))
            hits.append(ForbiddenHit(value=value, error_types=types, source_event_ids=ids))

    foreign = tuple(v for v in foreign_values if any(mentions(t, v) for t in text))

    if expected.kind is ExpectedKind.ABSTAIN:
        verdict = (
            Verdict.CORRECT if abstained
            else Verdict.LEAKED if hits
            else Verdict.NO_ABSTENTION
        )
    elif abstained:
        verdict = Verdict.FALSE_ABSTENTION
    elif matched:
        verdict = Verdict.POLLUTED if hits else Verdict.CORRECT
    else:
        verdict = Verdict.WRONG if hits else Verdict.MISSED
    return Score(verdict=verdict, matched=matched, forbidden_hits=tuple(hits), foreign_hits=foreign)
