"""Sentence/section embedding for the drift signals.

Provides a small ``SentenceEncoder`` interface with two implementations:

* ``MiniLMEncoder`` -- the real CPU encoder (sentence-transformers
  ``all-MiniLM-L6-v2``, ~80 MB), lazily imported so this module loads without
  torch installed.
* ``FakeEncoder`` -- a deterministic, dependency-light (numpy only) bag-of-words
  hash encoder used by the test suite so CI needs no network/GPU. Identical text
  maps to identical vectors and texts sharing tokens are closer, which is enough
  to exercise the clustering and directional-drift logic deterministically.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SentenceEncoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) array of (preferably unit-norm) embeddings."""
        ...


class FakeEncoder:
    """Deterministic bag-of-words hashing encoder (numpy only, no model)."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    @staticmethod
    def _bucket(token: str, dim: int) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=float)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                out[i, self._bucket(tok, self.dim)] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class MiniLMEncoder:
    """Real CPU encoder; sentence-transformers is imported lazily."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=float)  # MiniLM-L6 dim
        model = self._ensure()
        return model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
