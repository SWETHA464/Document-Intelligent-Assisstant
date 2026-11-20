"""Okapi BM25 keyword ranking (small in-house implementation, no extra dependency)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25Index:
    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(tokenized_docs)
        self._tf = [Counter(doc) for doc in tokenized_docs]
        self._len = np.array([len(doc) for doc in tokenized_docs], dtype=np.float32)
        self.avg_len = float(self._len.mean()) if self.n_docs else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        # BM25+ style IDF that is never negative
        self._idf = {t: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """BM25 score of every document for the query."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs or self.avg_len == 0:
            return out
        for term in set(query_tokens):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self._tf):
                freq = tf.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self._len[i] / self.avg_len)
                out[i] += idf * freq * (self.k1 + 1) / denom
        return out
