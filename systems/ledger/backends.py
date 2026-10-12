"""Model providers for the LLM extractor.

A backend does one thing: send a system prompt and a user prompt, and return
JSON that matches a schema (provider-native structured outputs, so parsing
never guesses). Two backends cover most model families:

* ``AnthropicBackend``: Claude, via the Anthropic SDK.
* ``OpenAIBackend``: OpenAI models via Chat Completions, and any
  OpenAI-compatible endpoint (Gemini, Mistral, Groq, Together, Ollama, vLLM,
  ...) by setting ``OPENAI_BASE_URL``. Strict ``json_schema`` support varies
  by endpoint; the extractor validates what comes back either way.

``backend_from_env`` reads the repository's ``.env``:

  LLM_PROVIDER   anthropic | openai. Default: whichever API key is set.
  LLM_MODEL      model id. Default for anthropic: claude-opus-5-5; required
                 for openai, since available models differ by account.
  LLM_EFFORT     reasoning effort. Anthropic: low..max (default low).
                 OpenAI: sent as reasoning_effort only when set (reasoning
                 models only).
  ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENAI_BASE_URL
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from interface.memory_system import Usage

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
MAX_OUTPUT_TOKENS = 16000
ENV_KEYS = ("LLM_PROVIDER", "LLM_MODEL", "LLM_EFFORT", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
            "OPENAI_API_KEY", "OPENAI_BASE_URL")


@dataclass(frozen=True)
class JSONResult:
    data: dict[str, Any] | None
    """``None`` when the model refused."""
    usage: Usage


class Backend(Protocol):
    provider: str
    model: str

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any], name: str) -> JSONResult: ...

    def config(self) -> dict[str, object]: ...


class AnthropicBackend:
    provider = "anthropic"

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, effort: str | None = "low", client: Any = None) -> None:
        self.model, self.effort = model, effort
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any], name: str) -> JSONResult:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config=output_config,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # re-run on a fallback model if a safety classifier declines
        )
        usage = Usage(llm_calls=1, llm_input_tokens=response.usage.input_tokens,
                      llm_output_tokens=response.usage.output_tokens, model_ids=(response.model,))
        if response.stop_reason == "refusal":
            return JSONResult(None, usage)
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"{self.model}: output truncated (request {response._request_id})")
        text = next(b.text for b in response.content if b.type == "text")
        return JSONResult(json.loads(text), usage)

    def config(self) -> dict[str, object]:
        return {"provider": self.provider, "model": self.model, "effort": self.effort}


class OpenAIBackend:
    provider = "openai"

    def __init__(self, model: str, effort: str | None = None, client: Any = None) -> None:
        self.model, self.effort = model, effort
        if client is None:
            import openai

            client = openai.OpenAI()  # reads OPENAI_API_KEY and OPENAI_BASE_URL
        self.client = client

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any], name: str) -> JSONResult:
        extra: dict[str, Any] = {"reasoning_effort": self.effort} if self.effort else {}
        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": name, "schema": schema, "strict": True}},
            **extra,
        )
        u = response.usage
        usage = Usage(llm_calls=1, llm_input_tokens=u.prompt_tokens if u else 0,
                      llm_output_tokens=u.completion_tokens if u else 0, model_ids=(response.model,))
        choice = response.choices[0]
        if choice.message.refusal or choice.finish_reason == "content_filter":
            return JSONResult(None, usage)
        if choice.finish_reason == "length":
            raise RuntimeError(f"{self.model}: output truncated (response {response.id})")
        return JSONResult(json.loads(choice.message.content), usage)

    def config(self) -> dict[str, object]:
        base = os.environ.get("OPENAI_BASE_URL") or None
        return {"provider": self.provider, "model": self.model, "effort": self.effort, "base_url": base}


def backend_from_env() -> Backend:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    # A blank line in .env means "unset". The SDKs read some of these
    # themselves and would take "" literally (an empty OPENAI_BASE_URL breaks
    # every request), so drop blanks from the environment.
    for k in ENV_KEYS:
        if k in os.environ and not os.environ[k].strip():
            del os.environ[k]
    env = lambda k: os.environ.get(k) or None

    provider = env("LLM_PROVIDER")
    if provider is None:
        provider = "anthropic" if env("ANTHROPIC_API_KEY") else "openai" if env("OPENAI_API_KEY") else None
    if provider is None:
        raise RuntimeError("no LLM configured: set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")

    model, effort = env("LLM_MODEL"), env("LLM_EFFORT")
    if provider == "anthropic":
        return AnthropicBackend(model or DEFAULT_ANTHROPIC_MODEL, effort or "low")
    if provider == "openai":
        if model is None:
            raise RuntimeError("set LLM_MODEL in .env to an OpenAI (or compatible) model id")
        return OpenAIBackend(model, effort)
    raise RuntimeError(f"unknown LLM_PROVIDER {provider!r} (expected 'anthropic' or 'openai')")
