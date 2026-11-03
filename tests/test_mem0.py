"""The Mem0 adapter, offline: a fake OpenAI client plays Mem0's two LLM steps
and a bag-of-words embedder, so the tests check the plumbing (roles, dates,
provenance, deletion, caching, usage), not Mem0's quality."""

import ast
import hashlib
import json
import math
import re
from datetime import datetime, timezone

import pytest
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

from interface.memory_system import DeleteRequest, Event, EventKind, Query
from runner.prices import cost_usd
from systems.mem0_oss.system import EMBEDDING_DIMS, Mem0System


def ts(day: int) -> datetime:
    return datetime(2026, 1, day, 12, tzinfo=timezone.utc)


def ev(eid: str, day: int, content: str, kind=EventKind.USER_MESSAGE) -> Event:
    return Event(event_id=eid, user_id="u1", session_id=f"s{day}", timestamp=ts(day), kind=kind, content=content)


def ask(s, text: str):
    return s.retrieve(Query(query_id="q", user_id="u1", text=text, asked_at=ts(28)))


class FakeOpenAI:
    """Extraction returns each user-role line as a fact; the update step adds
    a fact, or updates memory 0 when the fact says something "moved"."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.chat = type("C", (), {"completions": type("X", (), {"create": staticmethod(self._chat)})})
        self.embeddings = type("E", (), {"create": staticmethod(self._embed)})

    def _chat(self, **params):
        text = "\n".join(m["content"] for m in params["messages"])
        self.prompts.append(text)
        if params["messages"][0]["role"] == "system":
            facts = re.findall(r"^user: \[\w+\] (.*)$", text, re.M)
            out = {"facts": facts}
        else:
            # Mem0 writes both lists into the prompt as Python reprs.
            facts = ast.literal_eval(re.search(r"triple backticks\..*?```\s*(\[.*?\])\s*```", text, re.S).group(1))
            old = "'id': '0'" in text
            out = {"memory": [{"id": "0", "text": f, "event": "UPDATE"} if old and "Moved" in f
                              else {"id": str(i + 100), "text": f, "event": "ADD"} for i, f in enumerate(facts)]}
        return ChatCompletion.model_validate({
            "id": "c", "object": "chat.completion", "created": 0, "model": "gpt-4.1-mini-2025-04-14",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(out)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})

    def _embed(self, **params):
        vec = [0.0] * EMBEDDING_DIMS
        for w in re.findall(r"\w+", params["input"][0].lower()):
            vec[int(hashlib.md5(w.encode()).hexdigest(), 16) % EMBEDDING_DIMS] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return CreateEmbeddingResponse.model_validate({
            "object": "list", "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [v / norm for v in vec]}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5}})


class NoNetwork(FakeOpenAI):
    def _chat(self, **params):
        raise AssertionError("cache miss")

    _embed = _chat


@pytest.fixture
def system(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    client = FakeOpenAI()
    s = Mem0System(model="gpt-4.1-mini", cache_dir=tmp_path, client=client)
    s.client = client
    yield s
    s.reset()


def test_stores_facts_with_the_session_date_and_reports_usage(system):
    rep = system.ingest([ev("a", 1, "We're on PostgreSQL 14."),
                         ev("b", 1, "I'll assume Jenkins.", kind=EventKind.ASSISTANT_MESSAGE)])
    assert rep.items_added == 1
    assert set(rep.usage.model_ids) == {"gpt-4.1-mini-2025-04-14", "text-embedding-3-small"}
    assert cost_usd(rep.usage) > 0
    # Mem0's wall-clock date is replaced by the session's.
    assert "Today's date is 2026-01-01" in system.client.prompts[0]
    res = ask(system, "Which PostgreSQL version?")
    assert [e.content for e in res.evidence] == ["[2026-01-01] We're on PostgreSQL 14."]
    assert res.evidence[0].source_event_ids == ("a", "b")


def test_update_replaces_the_old_value(system):
    system.ingest([ev("a", 1, "We're on PostgreSQL 14.")])
    rep = system.ingest([ev("b", 5, "Moved to PostgreSQL 16.")])
    assert rep.items_updated == 1
    res = ask(system, "Which PostgreSQL version?")
    assert [e.content for e in res.evidence] == ["[2026-01-05] Moved to PostgreSQL 16."]
    assert system.stats("u1").n_items == 1


def test_deletion_by_source_and_by_description(system):
    system.ingest([ev("a", 1, "Staging is staging-7.internal.")])
    system.ingest([ev("c", 2, "CI runs on Fridays.")])
    req = dict(request_id="d", user_id="u1", requested_at=ts(3), description="Forget the staging host.")
    assert system.delete(DeleteRequest(**req)).items_deleted == 0  # no delete-by-description in Mem0
    assert system.delete(DeleteRequest(**req, source_event_ids=("a",))).items_deleted == 1
    assert "staging-7" not in str(ask(system, "staging host").evidence)
    assert system.ingest([ev("a", 1, "Staging is staging-7.internal.")]).items_rejected == 1
    assert system.stats("u1").n_items == 1


def test_reingest_adds_nothing_and_budget_is_respected(system):
    system.ingest([ev("a", 1, "We're on PostgreSQL 14.")])
    calls = len(system.client.prompts)
    system.ingest([ev("a", 1, "We're on PostgreSQL 14.")])
    assert len(system.client.prompts) == calls
    res = system.retrieve(Query(query_id="q", user_id="u1", text="PostgreSQL", asked_at=ts(28), token_budget=2))
    assert res.evidence == ()


def test_cached_replay_needs_no_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    events = [ev("a", 1, "We're on PostgreSQL 14.")]
    first = Mem0System(model="gpt-4.1-mini", cache_dir=tmp_path, client=FakeOpenAI())
    first.ingest(events)
    want = [e.content for e in ask(first, "PostgreSQL version").evidence]
    again = Mem0System(model="gpt-4.1-mini", cache_dir=tmp_path, client=NoNetwork())
    rep = again.ingest(events)
    assert [e.content for e in ask(again, "PostgreSQL version").evidence] == want
    assert rep.usage.llm_calls == 2 and again.config()["calls"] == 0


def test_coding_profile_changes_only_the_fact_types():
    from mem0.configs.prompts import USER_MEMORY_EXTRACTION_PROMPT as base
    from systems.mem0_oss.system import coding_extraction_prompt

    prompt = coding_extraction_prompt()
    assert "lasting property of a project: its configuration, infrastructure" in prompt
    assert "Personal Preferences" not in prompt and "Health and Wellness" not in prompt
    head, tail = base.split("Types of Information to Remember:")[0], base.split("Here are some few shot examples:")[1]
    assert prompt.startswith(head) and prompt.endswith(tail)  # role, user-only rule, examples, format kept
