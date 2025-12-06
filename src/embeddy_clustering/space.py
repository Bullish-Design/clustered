from __future__ import annotations

from typing import Iterable

import numpy as np
from pydantic import BaseModel, Field

from embeddify import EmbeddingResult


class EmbeddingSpace(BaseModel):
    """Convenience wrapper around an :class:`EmbeddingResult`.

    This type exposes embeddings as a stacked NumPy array and provides
    a few helpers for computing similarity matrices and nearest
    neighbors. It is designed to be a lightweight building block for
    downstream plugins such as clustering or dimensionality reduction.
    """

    result: EmbeddingResult = Field(description="Underlying embedding batch")
    metric: str = Field(default="cosine", description="Similarity metric to interpret the vectors with")

    model_config = {"arbitrary_types_allowed": True}

    @property
    def words(self) -> list[str]:
        """Return any associated text for each embedding.

        If the underlying :class:`Embedding` instances carry no text
        metadata, this will be an empty list.
        """
        return [e.text or "" for e in self.result.embeddings]

    @property
    def model_name(self) -> str:
        return self.result.model_name

    @property
    def dimensions(self) -> int:
        return self.result.dimensions

    def as_numpy(self) -> np.ndarray:
        """Return the embeddings as a stacked ``(n, d)`` array."""
        return self.result.as_numpy()

    def similarity_matrix(self) -> np.ndarray:
        """Compute a full pairwise similarity matrix.

        The implementation uses the metric implied by ``self.metric``.
        Currently only cosine similarity is supported.
        """
        vectors = self.as_numpy()
        if vectors.size == 0:
            return np.empty((0, 0), dtype=float)

        if self.metric == "cosine":
            # normalize to unit length
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            normalized = vectors / norms
            return normalized @ normalized.T

        raise ValueError(f"Unsupported metric for similarity matrix: {self.metric!r}")

    def neighbors(self, index: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Return the ``top_k`` most similar neighbors for a given index.

        The return value is a list of ``(neighbor_index, similarity)``
        pairs, sorted by similarity descending, and excluding the item
        itself.
        """
        sims = self.similarity_matrix()
        if sims.size == 0:
            return []

        if index < 0 or index >= sims.shape[0]:
            raise IndexError(f"Index {index} is out of range for {sims.shape[0]} embeddings")

        row = sims[index]
        # exclude self
        indices = np.arange(len(row))
        mask = indices != index
        indices = indices[mask]
        values = row[mask]

        order = np.argsort(values)[::-1][:top_k]
        return [(int(indices[i]), float(values[i])) for i in order]