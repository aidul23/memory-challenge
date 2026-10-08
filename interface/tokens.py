"""The one tokenizer used to count injected evidence tokens (R2).

The runner measures every system's evidence with this function, and systems
may import it to fit their output into ``Query.token_budget``. It is a
dependency-free stand-in (words and punctuation marks); to switch to a model
tokenizer, change only this module and bump ``TOKENIZER`` so old runs stay
distinguishable.
"""

from __future__ import annotations

import re

TOKENIZER = "regex-words/1"

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))
