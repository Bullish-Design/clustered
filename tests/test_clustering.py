from __future__ import annotations

import importlib

import numpy as np
import pytest

from embeddify import Embedding, EmbeddingResult
from embeddy_clustering import ClusterConfig, Clusterer, EmbeddingSpace


SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


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


def test_cluster_empty_space_returns_empty_result():
    empty_result = EmbeddingResult(embeddings=[], model_name="test-model", dimensions=0)
    space = EmbeddingSpace(result=empty_result)
    cfg = ClusterConfig(method="kmeans")
    clusterer = Clusterer(config=cfg)

    result = clusterer.cluster(space)

    assert result.clusters == []
    assert result.n_clusters == 0
    assert result.assignments == {}
    assert result.method == "kmeans"
    assert result.metrics.silhouette_score is None
    assert result.metrics.mean_internal_similarity is None


def test_clusterer_builds_clusters_and_centroids_without_sklearn_cluster():
    class DummyClusterer(Clusterer):
        def _cluster_kmeans(self, vectors: np.ndarray, n_clusters: int) -> np.ndarray:  # type: ignore[override]
            # Two clusters: first two points label 0, last two label 1
            return np.asarray([0, 0, 1, 1])

        def _cluster_hierarchical(self, vectors: np.ndarray, n_clusters: int) -> np.ndarray:  # type: ignore[override]
            return np.asarray([0, 1, 0, 1])

    vectors = [
        [0.0, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.0, 10.1],
    ]
    texts = ["a", "b", "c", "d"]
    space = make_space(vectors, texts)

    cfg = ClusterConfig(method="kmeans", n_clusters=2, deduplicate_inflections=False)
    clusterer = DummyClusterer(config=cfg)

    result = clusterer.cluster(space)

    assert result.n_clusters == 2
    clusters_by_id = {c.id: c for c in result.clusters}
    assert clusters_by_id[0].words == ["a", "b"]
    assert clusters_by_id[1].words == ["c", "d"]

    # centroids should be one of the members
    assert clusters_by_id[0].centroid in {"a", "b"}
    assert clusters_by_id[1].centroid in {"c", "d"}

    # Assignments map from word -> list of cluster ids
    assert result.assignments == {"a": [0], "b": [0], "c": [1], "d": [1]}


def test_mean_internal_similarity_computed_from_similarity_matrix():
    # two identical vectors per cluster -> internal similarities should be 1
    vectors = [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    texts = ["a1", "a2", "b1", "b2"]
    space = make_space(vectors, texts)

    class DummyClusterer(Clusterer):
        def _cluster_kmeans(self, vectors: np.ndarray, n_clusters: int) -> np.ndarray:  # type: ignore[override]
            return np.asarray([0, 0, 1, 1])

    cfg = ClusterConfig(method="kmeans", metric="cosine", n_clusters=2, deduplicate_inflections=False)
    clusterer = DummyClusterer(config=cfg)

    result = clusterer.cluster(space)

    # With perfectly tight cosine clusters, mean internal similarity should be 1
    assert result.metrics.mean_internal_similarity is not None
    assert result.metrics.mean_internal_similarity == pytest.approx(1.0, rel=1e-6)


def test_silhouette_score_present_only_when_sklearn_available():
    # Use a DummyClusterer to avoid depending on sklearn.cluster
    class DummyClusterer(Clusterer):
        def _cluster_kmeans(self, vectors: np.ndarray, n_clusters: int) -> np.ndarray:  # type: ignore[override]
            # Two well-separated clusters
            return np.asarray([0, 0, 1, 1])

    vectors = [
        [0.0, 0.0],
        [0.0, 0.2],
        [10.0, 10.0],
        [10.0, 10.2],
    ]
    texts = ["a1", "a2", "b1", "b2"]
    space = make_space(vectors, texts)

    cfg = ClusterConfig(method="kmeans", metric="euclidean", n_clusters=2, deduplicate_inflections=False)
    clusterer = DummyClusterer(config=cfg)

    result = clusterer.cluster(space)

    if SKLEARN_AVAILABLE:
        assert result.metrics.silhouette_score is not None
        assert -1.0 <= result.metrics.silhouette_score <= 1.0
    else:
        assert result.metrics.silhouette_score is None


def test_clusterer_with_inflection_deduplication():
    class DummyLemmatizer:
        def lemma(self, word: str) -> str:
            # strip a trailing "s" as a toy lemma rule
            return word.rstrip("s")

    class DummyClusterer(Clusterer):
        def _cluster_kmeans(self, vectors: np.ndarray, n_clusters: int) -> np.ndarray:  # type: ignore[override]
            # one label per lemma representative
            return np.arange(len(vectors), dtype=int)

    vectors = [
        [0.0, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.0, 10.1],
    ]
    texts = ["cats", "cat", "dogs", "dog"]
    space = make_space(vectors, texts)

    cfg = ClusterConfig(method="kmeans", deduplicate_inflections=True)
    clusterer = DummyClusterer(config=cfg)
    lemma = DummyLemmatizer()

    result = clusterer.cluster(space, lemmatizer=lemma)

    # Only representative forms are clustered
    words_in_clusters = sorted({w for cluster in result.clusters for w in cluster.words})
    assert words_in_clusters == ["cats", "dogs"]

    # Inflection groups are returned for downstream mapping
    assert result.inflection_groups is not None
    by_lemma = {g.lemma: g for g in result.inflection_groups}
    assert set(by_lemma["cat"].variants) == {"cats", "cat"}
    assert set(by_lemma["dog"].variants) == {"dog", "dogs"}
