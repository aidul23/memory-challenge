"""LongMemEval (Wu et al., ICLR 2025) as event streams for any memory system.

Each instance is one user's chat history (about 48 sessions, ~115k tokens in
LongMemEval_S) and one question asked after it. Every turn becomes an
``Event`` (user or assistant message) at its session's date, sessions in date
order, and the user id is the question id so instances never share memory.

Download the data (git-ignored) with::

    mkdir -p data/longmemeval && curl -L -o data/longmemeval/longmemeval_s_cleaned.json \\
      https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

The full set is 500 questions; ``stratified_sample`` draws a fixed subset with
every question type (and its abstention questions) represented, and the ids
are saved so a run can be repeated exactly.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from interface.memory_system import Event, EventKind

DEFAULT_DATA = Path("data/longmemeval/longmemeval_s_cleaned.json")
SUBSET_DIR = Path("workloads/longmemeval")

KINDS = {"user": EventKind.USER_MESSAGE, "assistant": EventKind.ASSISTANT_MESSAGE}


def parse_date(s: str) -> datetime:
    """``2023/05/20 (Sat) 02:21`` -> an aware UTC datetime."""
    return datetime.strptime(s, "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Instance:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: datetime
    events: tuple[Event, ...]
    sessions: int

    @property
    def abstention(self) -> bool:
        return self.question_id.endswith("_abs")

    @property
    def category(self) -> str:
        """The question type, with abstention questions as their own category."""
        return "abstention" if self.abstention else self.question_type


def to_instance(raw: dict[str, Any]) -> Instance:
    qid = raw["question_id"]
    sessions = sorted(zip(raw["haystack_dates"], raw["haystack_session_ids"], raw["haystack_sessions"]),
                      key=lambda s: parse_date(s[0]))
    events = []
    for date, sid, turns in sessions:
        start = parse_date(date)
        for j, turn in enumerate(turns):
            if turn.get("role") not in KINDS or not str(turn.get("content", "")).strip():
                continue
            events.append(Event(
                event_id=f"{sid}#{j}", user_id=qid, session_id=sid,
                timestamp=start + timedelta(seconds=j), kind=KINDS[turn["role"]],
                content=turn["content"],
            ))
    return Instance(
        question_id=qid, question_type=raw["question_type"], question=raw["question"],
        answer=str(raw["answer"]), question_date=parse_date(raw["question_date"]),
        events=tuple(events), sessions=len(sessions),
    )


def load(path: Path = DEFAULT_DATA, ids: Sequence[str] | None = None) -> list[Instance]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if ids is not None:
        by_id = {r["question_id"]: r for r in raw}
        return [to_instance(by_id[i]) for i in ids]
    return [to_instance(r) for r in raw]


def category_of(raw: dict[str, Any]) -> str:
    return "abstention" if raw["question_id"].endswith("_abs") else raw["question_type"]


def stratified_sample(path: Path, per_category: int, seed: int = 0) -> list[str]:
    """``per_category`` question ids from each category (fewer if it has
    fewer), drawn with a fixed seed, in dataset order."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = defaultdict(list)
    for r in raw:
        groups[category_of(r)].append(r["question_id"])
    rng = random.Random(seed)
    chosen = {i for ids in groups.values() for i in rng.sample(ids, min(per_category, len(ids)))}
    return [r["question_id"] for r in raw if r["question_id"] in chosen]


def save_subset(ids: Sequence[str], name: str) -> Path:
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBSET_DIR / f"{name}.json"
    path.write_text(json.dumps({"source": DEFAULT_DATA.name, "question_ids": list(ids)}, indent=1),
                    encoding="utf-8")
    return path


def load_subset(name: str) -> list[str]:
    return json.loads((SUBSET_DIR / f"{name}.json").read_text(encoding="utf-8"))["question_ids"]
