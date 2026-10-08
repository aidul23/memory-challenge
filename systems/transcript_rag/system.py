"""RAG over raw transcripts: BM25 over individual events, top-k within the
token budget. Lexical rather than embedding-based so the baseline needs no
model and is deterministic; an embedding variant can share the same store.
The index is rebuilt per query, which is fine at baseline scale."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from interface.memory_system import Evidence, Query, RetrievalResult
from interface.tokens import count_tokens
from systems.transcript_store import TranscriptStore

_WORD = re.compile(r"\w+")
STOPWORDS = frozenset(
    "a an and are as at be by can could did do does for from has have how i in is "
    "it its me my of on or our please s should t that the this to us was we were "
    "what when where which who why will with would you your".split()
)


def _terms(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


class TranscriptRAG(TranscriptStore):
    name = "transcript_rag"
    version = "1"

    def __init__(self, top_k: int = 5, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__()
        self.top_k, self.k1, self.b = top_k, k1, b

    def retrieve(self, query: Query) -> RetrievalResult:
        docs = self._history(query.user_id)
        q = set(_terms(query.text))
        if not docs or not q:
            return RetrievalResult()

        tfs = [Counter(_terms(e.content)) for e in docs]
        lens = [sum(tf.values()) for tf in tfs]
        avg_len = sum(lens) / len(lens) or 1.0
        n = len(docs)
        idf = {}
        for t in q:
            df = sum(1 for tf in tfs if t in tf)
            idf[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))

        scored = []
        for e, tf, dl in zip(docs, tfs, lens):
            s = sum(
                idf[t] * tf[t] * (self.k1 + 1)
                / (tf[t] + self.k1 * (1 - self.b + self.b * dl / avg_len))
                for t in q
                if tf[t]
            )
            if s > 0:
                scored.append((s, e))
        scored.sort(key=lambda se: (-se[0], se[1].timestamp))

        evidence: list[Evidence] = []
        used = 0
        for s, e in scored[: self.top_k]:
            ev = self._evidence(e, round(s, 4))
            cost = count_tokens(ev.content)
            if query.token_budget is not None and used + cost > query.token_budget:
                break
            evidence.append(ev)
            used += cost
        return RetrievalResult(evidence=tuple(evidence))

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name, "version": self.version, "retriever": "bm25",
            "unit": "event", "top_k": self.top_k, "k1": self.k1, "b": self.b,
            "stopwords": len(STOPWORDS),
        }
