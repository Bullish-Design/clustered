from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class DummyEmbedding(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    vector: np.ndarray = Field(...)
    text: str = Field(...)


class DummyEmbeddingResult(BaseModel):
    embeddings: list[DummyEmbedding] = Field(default_factory=list)
    model_name: str = "test-model"
    dimensions: int = 0

import numpy as np
import pytest

from embeddy_clustering import ClusterConfig, Clusterer, EmbeddingSpace


def make_space(vectors, texts):
    embs = [
        DummyEmbedding(vector=np.asarray(vec, dtype=float), text=text)
        for vec, text in zip(vectors, texts)
    ]
    result = DummyEmbeddingResult(embeddings=embs, dimensions=len(vectors[0]) if vectors else 0)
    return EmbeddingSpace(result=result)


def test_cluster_empty_space_returns_empty_result():
    space = make_space([], [])
    cfg = ClusterConfig(method="kmeans")
    clusterer = Clusterer(config=cfg)

    result = clusterer.cluster(space)

    assert result.clusters == []
    assert result.n_clusters == 0
    assert result.assignments == {}
    assert result.metrics.silhouette_score is None
    assert result.metrics.mean_internal_similarity is None


def test_clusterer_with_custom_labels_and_centroids():
    class DummyClusterer(Clusterer):
        def _cluster_kmeans(self, vectors, n_clusters):  # type: ignore[override]
            return np.asarray([0, 0, 1, 1])

    vectors = [
        [0.0, 1.0],
        [0.0, 2.0],
        [10.0, 10.0],
        [10.0, 11.0],
    ]
    texts = ["a", "b", "c", "d"]
    space = make_space(vectors, texts)

    cfg = ClusterConfig(method="kmeans", n_clusters=2)
    clusterer = DummyClusterer(config=cfg)

    result = clusterer.cluster(space)

    assert result.n_clusters == 2
    clusters_by_id = {c.id: c for c in result.clusters}

    assert clusters_by_id[0].words == ["a", "b"]
    assert clusters_by_id[1].words == ["c", "d"]

    assert clusters_by_id[0].centroid in {"a", "b"}
    assert clusters_by_id[1].centroid in {"c", "d"}

    assert result.assignments == {"a": [0], "b": [0], "c": [1], "d": [1]}


def test_metrics_are_computed_reasonably():
    class DummyClusterer(Clusterer):
        def _cluster_kmeans(self, vectors, n_clusters):  # type: ignore[override]
            return np.asarray([0, 0, 1, 1])

    vectors = [
        [0.0, 1.0],
        [0.0, 2.0],
        [10.0, 10.0],
        [10.0, 11.0],
    ]
    texts = ["a", "b", "c", "d"]
    space = make_space(vectors, texts)

    cfg = ClusterConfig(method="kmeans", n_clusters=2)
    clusterer = DummyClusterer(config=cfg)
    result = clusterer.cluster(space)

    metrics = result.metrics
    assert metrics.silhouette_score is None or metrics.silhouette_score > 0.5
    assert metrics.mean_internal_similarity is not None
    assert 0.0 <= metrics.mean_internal_similarity <= 1.0


def test_clusterer_with_inflection_deduplication():
    class DummyLemmatizer:
        def lemma(self, word: str) -> str:
            return word.rstrip("s")

    class DummyClusterer(Clusterer):
        def _cluster_kmeans(self, vectors, n_clusters):  # type: ignore[override]
            return np.arange(len(vectors), dtype=int)

    vectors = [
        [0.0, 1.0],
        [0.0, 2.0],
        [10.0, 10.0],
        [10.0, 11.0],
    ]
    texts = ["cats", "cat", "dogs", "dog"]
    space = make_space(vectors, texts)

    cfg = ClusterConfig(method="kmeans", deduplicate_inflections=True)
    clusterer = DummyClusterer(config=cfg)
    lemma = DummyLemmatizer()

    result = clusterer.cluster(space, lemmatizer=lemma)

    assert set(result.assignments.keys()) <= {"cats", "cat", "dogs", "dog"}
    assert result.inflection_groups is not None
    by_lemma = {g.lemma: g for g in result.inflection_groups}
    assert set(by_lemma["cat"].variants) == {"cats", "cat"}
    assert set(by_lemma["dog"].variants) == {"dogs", "dog"}
