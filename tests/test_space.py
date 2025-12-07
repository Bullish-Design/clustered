from __future__ import annotations

import numpy as np
import pytest

from embeddy_clustering import EmbeddingSpace

from pydantic import BaseModel, Field, ConfigDict


class DummyEmbedding(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    vector: np.ndarray = Field(...)
    text: str = Field(...)


class DummyEmbeddingResult(BaseModel):
    embeddings: list[DummyEmbedding] = Field(default_factory=list)
    model_name: str = "test-model"
    dimensions: int = 0


def make_space(vectors, texts):
    embs = [
        DummyEmbedding(vector=np.asarray(vec, dtype=float), text=text)
        for vec, text in zip(vectors, texts)
    ]
    result = DummyEmbeddingResult(embeddings=embs, dimensions=len(vectors[0]) if vectors else 0)
    return EmbeddingSpace(result=result)


def test_words_dimensions_and_as_numpy():
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    texts = ["foo", "bar"]
    space = make_space(vectors, texts)

    assert space.words == texts
    assert space.dimensions == 2

    arr = space.as_numpy()
    assert arr.shape == (2, 2)
    assert arr.dtype == float


def test_similarity_matrix_and_neighbors():
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ]
    texts = ["x", "y", "z"]
    space = make_space(vectors, texts)

    sims = space.similarity_matrix()
    assert sims.shape == (3, 3)
    for i in range(3):
        assert sims[i, i] == pytest.approx(1.0, rel=1e-6)

    neighbors_0 = space.neighbors(0, top_k=2)
    assert neighbors_0[0][0] == 2

    single = make_space([[1.0, 0.0]], ["solo"])
    assert single.neighbors(0) == []

    with pytest.raises(IndexError):
        single.neighbors(1)
