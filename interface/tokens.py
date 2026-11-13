"""The one tokenizer used to count injected evidence tokens (R2).

The runner measures every system's evidence with this function, and systems
import it to fit their output into ``Query.token_budget``. It is the answering
model's own tokenizer: ``o200k_base``, the encoding of ``gpt-4.1-mini``.

Runs logged before this change (``settings.tokenizer == "regex-words/1"``)
counted words and punctuation marks instead. In none of them did a budget cut
any evidence (``runner/recount_tokens.py`` checks this), so their evidence
is what this tokenizer would have selected; their token counts are recomputed
from the logged evidence by ``runner/recount_tokens.py``.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

TOKENIZER = "o200k_base/1"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))
