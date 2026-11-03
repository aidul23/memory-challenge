"""Pinned prices, so every run reports cost in dollars computed by the runner.

Prices are USD per million tokens, copied from the provider's model page on
the date given. A model is looked up by the longest listed id its reported id
starts with (``gpt-4.1-mini-2025-04-14`` -> ``gpt-4.1-mini``). Input tokens
are all charged at the uncached rate, so costs are an upper bound when the
provider's automatic prompt caching applied.
"""

from __future__ import annotations

from interface.memory_system import Usage

PRICES_AS_OF = "2026-10-19"

#: model id -> (input, output) USD per 1M tokens.
PRICES: dict[str, tuple[float, float]] = {
    # https://developers.openai.com/api/docs/models/gpt-4.1-mini
    "gpt-4.1-mini": (0.40, 1.60),
    # https://developers.openai.com/api/docs/models/gpt-4o (LongMemEval's judge;
    # the exact snapshot, since other gpt-4o snapshots are priced differently)
    "gpt-4o-2024-08-06": (2.50, 10.00),
    # https://developers.openai.com/api/docs/models/gpt-4.1 (the upgrade target in
    # the model-change experiment; exact snapshot, so it never matches gpt-4.1-mini)
    "gpt-4.1-2025-04-14": (2.00, 8.00),
}

#: Embedding model id -> USD per 1M input tokens.
EMBEDDING_PRICES: dict[str, float] = {
    # https://developers.openai.com/api/docs/models/text-embedding-3-small
    # (checked 2026-11-03; Mem0's embedder)
    "text-embedding-3-small": 0.02,
}


def price_of(model_id: str) -> tuple[float, float]:
    matches = [m for m in PRICES if model_id.startswith(m)]
    if not matches:
        raise KeyError(f"no pinned price for model {model_id!r}; add it to runner/prices.py")
    return PRICES[max(matches, key=len)]


def cost_usd(usage: Usage) -> float:
    """Dollar cost of ``usage``. Zero when no model was called; an error when
    the LLM or the embedding tokens span several models, since they cannot be
    split by model."""
    embedders = [m for m in usage.model_ids if m in EMBEDDING_PRICES]
    llms = [m for m in usage.model_ids if m not in EMBEDDING_PRICES]
    cost = 0.0
    if usage.embedding_tokens:
        if len(embedders) != 1:
            raise ValueError(f"cannot price embedding usage across models {usage.model_ids!r}")
        cost += usage.embedding_tokens * EMBEDDING_PRICES[embedders[0]] / 1e6
    if usage.llm_input_tokens or usage.llm_output_tokens:
        if len(llms) != 1:
            raise ValueError(f"cannot price usage across models {usage.model_ids!r}")
        inp, out = price_of(llms[0])
        cost += (usage.llm_input_tokens * inp + usage.llm_output_tokens * out) / 1e6
    return cost
