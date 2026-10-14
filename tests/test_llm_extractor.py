"""LLM extractor tests against scripted fake clients for both providers:
request shape, output filtering, caching, refusals, provider selection from
the environment, and the ledger passing known slots through. No network."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from interface.memory_system import Event, EventKind, Query
from systems.ledger.backends import AnthropicBackend, OpenAIBackend, backend_from_env
from systems.ledger.llm import LLMExtractor, slot_key
from systems.ledger.system import Ledger


class FakeAnthropic:
    """Answers each request with ``reply(system, prompt) -> dict | "refusal"``."""

    def __init__(self, reply):
        self.reply = reply
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        out = self.reply(request["system"], request["messages"][0]["content"])
        refused = out == "refusal"
        return SimpleNamespace(
            model=request["model"], _request_id="req_test",
            stop_reason="refusal" if refused else "end_turn",
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
            content=[] if refused else [SimpleNamespace(type="text", text=json.dumps(out))],
        )


class FakeOpenAI:
    def __init__(self, reply):
        self.reply = reply
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        out = self.reply(request["messages"][0]["content"], request["messages"][1]["content"])
        refused = out == "refusal"
        message = SimpleNamespace(content=None if refused else json.dumps(out),
                                  refusal="I can't help with that." if refused else None)
        return SimpleNamespace(
            id="chatcmpl_test", model=request["model"],
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=90, completion_tokens=10),
        )


def anthropic_x(reply, tmp_path):
    client = FakeAnthropic(reply)
    return LLMExtractor(AnthropicBackend(client=client), cache_dir=tmp_path), client


def openai_x(reply, tmp_path):
    client = FakeOpenAI(reply)
    return LLMExtractor(OpenAIBackend("gpt-test", client=client), cache_dir=tmp_path), client


PROVIDERS = [anthropic_x, openai_x]


def ev(eid: str, day: int, content: str, kind=EventKind.USER_MESSAGE) -> Event:
    return Event(event_id=eid, user_id="u1", session_id=f"s{day}",
                 timestamp=datetime(2026, 1, day, 12, tzinfo=timezone.utc), kind=kind, content=content)


def test_slot_key_normalises():
    assert slot_key(" Payments-Service ", "Deploy Branch") == "payments-service/deploy_branch"


@pytest.mark.parametrize("make", PROVIDERS)
def test_claims_filtering(make, tmp_path):
    x, _ = make(lambda system, prompt: {"claims": [
        {"event_id": "a", "entity": "payments-service", "attribute": "database", "value": " PostgreSQL 14 "},
        {"event_id": "zzz", "entity": "x", "attribute": "y", "value": "invented event"},
        {"event_id": "a", "entity": "x", "attribute": "y", "value": "  "},
        {"event_id": "a", "entity": "x"},  # malformed: an endpoint that ignores the schema
    ]}, tmp_path)
    claims, usage = x.claims([ev("a", 1, "payments-service runs on PostgreSQL 14.")], {"k/v": "something"})
    assert [(c.event_id, c.slot, c.value) for c in claims] == [("a", "payments-service/database", "PostgreSQL 14")]
    assert usage.llm_calls == 1 and usage.llm_input_tokens > 0


def test_anthropic_request_shape(tmp_path):
    x, client = anthropic_x(lambda s, p: {"claims": []}, tmp_path)
    x.claims([ev("a", 1, "hello")], {"k/v": "something"})
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"] == {"format": {"type": "json_schema", "schema": req["output_config"]["format"]["schema"]},
                                    "effort": "low"}
    assert req["fallbacks"] == "default"
    assert "[a]" in req["messages"][0]["content"] and "- k/v: something" in req["messages"][0]["content"]


def test_openai_request_shape(tmp_path):
    x, client = openai_x(lambda s, p: {"claims": []}, tmp_path)
    x.claims([ev("a", 1, "hello")])
    req = client.requests[0]
    assert req["model"] == "gpt-test"
    assert [m["role"] for m in req["messages"]] == ["system", "user"]
    assert req["response_format"]["type"] == "json_schema"
    assert req["response_format"]["json_schema"]["strict"] is True
    assert "reasoning_effort" not in req
    OpenAIBackend("o-test", effort="low", client=client).complete_json("s", "p", {"type": "object"}, "n")
    assert client.requests[-1]["reasoning_effort"] == "low"


@pytest.mark.parametrize("make", PROVIDERS)
def test_cache_replays_without_calling(make, tmp_path):
    x, client = make(lambda s, p: {"claims": []}, tmp_path)
    events = [ev("a", 1, "hello")]
    x.claims(events)
    again = LLMExtractor(x.backend, cache_dir=tmp_path)
    _, usage = again.claims(events)
    assert len(client.requests) == 1
    assert again.cache_hits == 1 and usage.llm_input_tokens > 0  # original cost is reported


def test_cache_is_per_provider_and_model(tmp_path):
    a, _ = anthropic_x(lambda s, p: {"claims": []}, tmp_path)
    o, oc = openai_x(lambda s, p: {"claims": []}, tmp_path)
    a.claims([ev("a", 1, "hello")])
    o.claims([ev("a", 1, "hello")])
    assert len(oc.requests) == 1


@pytest.mark.parametrize("make", PROVIDERS)
def test_slots_for_uses_enum_and_skips_call_when_nothing_known(make, tmp_path):
    x, client = make(lambda s, p: {"slots": ["a/db", "a/db", "made/up"]}, tmp_path)
    assert x.slots_for("anything?", {})[0] == []
    assert client.requests == []
    slots, _ = x.slots_for("Which db does a use?", {"a/db": "db of a", "b/db": "db of b"})
    assert slots == ["a/db"]


@pytest.mark.parametrize("make", PROVIDERS)
def test_event_ids_are_enumerated_and_bad_ones_counted(make, tmp_path):
    # Seen with gpt-4.1-mini: "e0014" for "e00014" silently lost a whole batch.
    x, client = make(lambda s, p: {"claims": [
        {"event_id": "e0014", "entity": "a", "attribute": "db", "value": "x"},
        {"event_id": "e00014", "entity": "a", "attribute": "db", "value": "y"},
    ]}, tmp_path)
    claims, _ = x.claims([ev("e00014", 1, "a uses y"), ev("e00015", 1, "hi")])
    assert [c.value for c in claims] == ["y"]
    assert x.dropped == 1 and x.config()["dropped_claims"] == 1
    assert '"enum": ["e00014", "e00015"]' in json.dumps(client.requests[0])


@pytest.mark.parametrize("make", PROVIDERS)
def test_refusal_yields_nothing_and_is_not_cached(make, tmp_path):
    x, client = make(lambda s, p: "refusal", tmp_path)
    assert x.claims([ev("a", 1, "hi")])[0] == []
    x.claims([ev("a", 1, "hi")])
    assert x.refusals == 2 and len(client.requests) == 2


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A clean LLM environment that never reads the developer's real .env."""
    monkeypatch.setattr("systems.ledger.backends.ROOT", tmp_path)
    for k in ("LLM_PROVIDER", "LLM_MODEL", "LLM_EFFORT", "LLM_TEMPERATURE", "ANTHROPIC_API_KEY",
              "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.setenv(k, "")
    return monkeypatch


def test_backend_from_env_picks_provider_by_key(env):
    env.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(RuntimeError, match="LLM_MODEL"):
        backend_from_env()
    env.setenv("LLM_MODEL", "gpt-test")
    b = backend_from_env()
    assert (b.provider, b.model, b.effort) == ("openai", "gpt-test", None)

    env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    env.setenv("LLM_MODEL", "")
    b = backend_from_env()
    assert (b.provider, b.model, b.effort) == ("anthropic", "claude-opus-5-5", "low")

    env.setenv("LLM_PROVIDER", "openai")
    env.setenv("LLM_MODEL", "gpt-test")
    assert backend_from_env().provider == "openai"


def test_blank_env_values_are_unset(env):
    env.setenv("OPENAI_API_KEY", "sk-test")
    env.setenv("LLM_MODEL", "gpt-test")
    b = backend_from_env()  # OPENAI_BASE_URL is "" in the fixture
    assert "OPENAI_BASE_URL" not in __import__("os").environ
    assert str(b.client.base_url).startswith("https://api.openai.com")


def test_openai_sampling_is_pinned_unless_reasoning(env, tmp_path):
    x, client = openai_x(lambda s, p: {"claims": []}, tmp_path)
    x.claims([ev("a", 1, "hello")])
    assert (client.requests[0]["temperature"], client.requests[0]["seed"]) == (0.0, 0)
    assert x.backend.config()["temperature"] == 0.0  # so it is part of the cache key

    client = FakeOpenAI(lambda s, p: {"claims": []})
    LLMExtractor(OpenAIBackend("o-test", effort="low", client=client, temperature=None),
                 cache_dir=tmp_path).claims([ev("a", 1, "hello")])
    assert "temperature" not in client.requests[0] and "seed" not in client.requests[0]

    env.setenv("OPENAI_API_KEY", "sk-test")
    env.setenv("LLM_MODEL", "gpt-test")
    assert backend_from_env().temperature == 0.0
    env.setenv("LLM_EFFORT", "low")
    assert backend_from_env().temperature is None
    env.setenv("LLM_TEMPERATURE", "0.7")
    assert backend_from_env().temperature == 0.7
    env.setenv("LLM_TEMPERATURE", "none")
    assert backend_from_env().temperature is None
    env.setenv("LLM_TEMPERATURE", "cold")
    with pytest.raises(RuntimeError, match="LLM_TEMPERATURE"):
        backend_from_env()


def test_backend_from_env_errors(env):
    with pytest.raises(RuntimeError, match="no LLM configured"):
        backend_from_env()
    env.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(RuntimeError, match="unknown LLM_PROVIDER"):
        backend_from_env()


@pytest.mark.parametrize("make", PROVIDERS)
def test_ledger_with_llm_extractor(make, tmp_path):
    prompts: list[str] = []

    def reply(system, prompt):
        prompts.append(prompt)
        if "Events:" not in prompt:
            return {"slots": ["payments-service/database"] if "database" in prompt.split("Text:")[1] else []}
        if "[b]" in prompt:
            return {"claims": [{"event_id": "b", "entity": "payments-service", "attribute": "database",
                                "value": "PostgreSQL 16"}]}
        return {"claims": [{"event_id": "a", "entity": "Payments-Service", "attribute": "database",
                            "value": "PostgreSQL 14"}]}

    x, _ = make(reply, tmp_path)
    s = Ledger(x)
    s.ingest([ev("a", 1, "payments-service runs on PostgreSQL 14.")])
    s.ingest([ev("b", 5, "migrate.log: payments-service migrated to PostgreSQL 16", kind=EventKind.TOOL_RESULT)])
    # The second batch saw the slot and its current value.
    assert "payments-service/database: database of Payments-Service (current: PostgreSQL 14)" in prompts[1]

    def ask(text, as_of=None):
        t = lambda d: datetime(2026, 1, d, 12, tzinfo=timezone.utc)
        return s.retrieve(Query(query_id="q", user_id="u1", text=text, asked_at=t(9), as_of=as_of and t(as_of)))

    assert "PostgreSQL 16" in ask("Which database does payments-service use?").evidence[0].content
    assert "PostgreSQL 14" in ask("Which database does payments-service use?", as_of=3).evidence[0].content
    assert ask("Which queue does payments-service use?").abstained
    assert s.config()["extractor"]["provider"] == x.backend.provider
