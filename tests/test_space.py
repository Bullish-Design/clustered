from __future__ import annotations

import numpy as np
import pytest

from embeddify import Embedding, EmbeddingResult
from embeddy_clustering import EmbeddingSpace


def make_space(vectors: list[list[float]], texts: list[str]) -> EmbeddingSpace:
    embs = [
        Embedding(
            vector=np.asarray(vec, dtype=float),
            model_name="test-model",
            normalized=False,
            text=text,
        )
        for vec, text in zip(vectors, texts)
    ]
    result = EmbeddingResult(
        embeddings=embs,
        model_name="test-model",
        dimensions=len(vectors[0]),
    )
    return EmbeddingSpace(result=result)


def test_words_dimensions_and_as_numpy():
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    texts = ["foo", "bar"]
    space = make_space(vectors, texts)

    assert space.words == texts
    assert space.dimensions == 2

    as_np = space.as_numpy()
    assert isinstance(as_np, np.ndarray)
    assert as_np.shape == (2, 2)
    np.testing.assert_allclose(as_np, np.asarray(vectors, dtype=float))


def test_similarity_matrix_and_neighbors_cosine():
    vectors = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ]
    texts = ["x", "y", "z"]
    space = make_space(vectors, texts)

    sims = space.similarity_matrix()
    assert sims.shape == (3, 3)

    # cosine similarity: self-similarity is 1
    np.testing.assert_allclose(np.diag(sims), np.ones(3))

    # identical vectors (0 and 2) should have highest similarity
    neighbors_0 = space.neighbors(0, top_k=2)
    assert neighbors_0[0][0] == 2
    assert neighbors_0[0][1] == pytest.approx(1.0, rel=1e-6)

    # when only one embedding is present, neighbors should be empty
    single_space = make_space([[1.0, 0.0]], ["solo"])
    assert single_space.neighbors(0, top_k=5) == []

    # out-of-range index should raise IndexError
    with pytest.raises(IndexError):
        single_space.neighbors(1)
