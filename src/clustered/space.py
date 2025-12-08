from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class EmbeddingSpace(BaseModel):
    """Convenience wrapper around a batch of embeddings.

    This type exposes embeddings as a stacked NumPy array and provides
    helpers for computing similarity matrices and nearest neighbours.
    It only relies on a minimal ``.embeddings`` / ``.vector`` / ``.text``
    API from the underlying result object and does *not* require a
    specific Embeddify type at runtime.
    """

    result: Any = Field(..., description="Underlying embedding batch")
    metric: str = Field(
        default="cosine",
        description="Similarity metric used for similarity_matrix() and neighbors()",
    )

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Basic views
    # ------------------------------------------------------------------

    @property
    def words(self) -> list[str]:
        """Return the surface text for each embedding."""
        words: list[str] = []
        embeddings = getattr(self.result, "embeddings", []) or []
        for e in embeddings:
            text = getattr(e, "text", None)
            if text is None:
                text = str(e)
            words.append(text)
        return words

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embeddings."""
        try:
            dims = int(getattr(self.result, "dimensions"))
            if dims > 0:
                return dims
        except Exception:  # pragma: no cover - defensive
            pass

        embeddings = getattr(self.result, "embeddings", []) or []
        if not embeddings:
            return 0

        vec = np.asarray(getattr(embeddings[0], "vector"), dtype=float)
        if vec.ndim == 0:
            return 0
        return int(vec.shape[-1])

    # ------------------------------------------------------------------
    # Numeric helpers
    # ------------------------------------------------------------------

    def as_numpy(self) -> np.ndarray:
        """Return embeddings as a 2D ``(n_samples, dimensions)`` array."""
        embeddings = getattr(self.result, "embeddings", []) or []
        if not embeddings:
            return np.zeros((0, 0), dtype=float)

        vectors = [np.asarray(getattr(e, "vector"), dtype=float) for e in embeddings]
        return np.stack(vectors, axis=0)

    def similarity_matrix(self) -> np.ndarray:
        """Compute the pairwise cosine similarity matrix."""
        x = self.as_numpy()
        if x.size == 0:
            return np.zeros((0, 0), dtype=float)

        if self.metric != "cosine":
            raise ValueError(f"Unsupported similarity metric: {self.metric!r}")

        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        x_norm = x / norms
        sims = x_norm @ x_norm.T
        return sims.astype(float)

    def neighbors(self, index: int, top_k: int = 10) -> list[tuple[int, float]]:
        """Return the top-k nearest neighbours for ``index``."""
        if top_k <= 0:
            return []

        sims = self.similarity_matrix()
        if sims.size == 0:
            return []

        if index < 0 or index >= sims.shape[0]:
            raise IndexError(f"Index {index} is out of range for {sims.shape[0]} embeddings")

        row = sims[index]
        indices = np.arange(len(row))
        mask = indices != index
        indices = indices[mask]
        values = row[mask]

        if indices.size == 0:
            return []

        order = np.argsort(values)[::-1][:top_k]
        return [(int(indices[i]), float(values[i])) for i in order]
