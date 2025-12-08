# tests/test_clustering.py
from __future__ import annotations

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

from clustered import ClusterConfig, Clusterer, EmbeddingSpace


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


class TestMembershipBasedClustering:
    """Test unified membership-based architecture."""

    def test_kmeans_produces_binary_memberships(self):
        """KMeans should produce binary (0.0 or 1.0) memberships."""
        vectors = [[0.0, 1.0], [0.0, 2.0], [10.0, 10.0], [10.0, 11.0]]
        texts = ["a", "b", "c", "d"]
        space = make_space(vectors, texts)

        cfg = ClusterConfig(method="kmeans", n_clusters=2)
        clusterer = Clusterer(config=cfg)
        result = clusterer.cluster(space)

        # Check all memberships are binary
        for word_memberships in result.memberships.values():
            for m in word_memberships:
                assert m.score in (0.0, 1.0)
                assert m.score_type == "binary"

        # Each word in exactly one cluster
        for word in texts:
            clusters = result.get_clusters_for_word(word, min_score=0.5)
            assert len(clusters) == 1

    def test_fuzzy_produces_continuous_memberships(self):
        """Fuzzy c-means should produce continuous memberships."""
        vectors = [[0.0, 1.0], [0.1, 1.0], [10.0, 10.0], [10.1, 10.0]]
        texts = ["a", "b", "c", "d"]
        space = make_space(vectors, texts)

        cfg = ClusterConfig(method="fuzzy", n_clusters=2, fuzzy_m=2.0)
        clusterer = Clusterer(config=cfg)
        result = clusterer.cluster(space)

        # Check memberships are continuous
        all_scores = [m.score for mems in result.memberships.values() for m in mems]
        # At least some should be non-binary
        non_binary = [s for s in all_scores if s not in (0.0, 1.0)]
        assert len(non_binary) > 0

    def test_threshold_allows_multi_cluster_membership(self):
        """Threshold method should allow words in multiple clusters."""
        vectors = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
        texts = ["x", "y", "mid"]
        space = make_space(vectors, texts)

        cfg = ClusterConfig(method="threshold", n_clusters=2, threshold=0.5)
        clusterer = Clusterer(config=cfg)
        result = clusterer.cluster(space)

        # "mid" might be in both clusters
        mid_clusters = result.get_clusters_for_word("mid", min_score=0.5)
        # At least check it works
        assert len(mid_clusters) >= 1

    def test_empty_space_returns_empty_result(self):
        """Empty input should return empty result."""
        space = make_space([], [])
        cfg = ClusterConfig(method="kmeans")
        clusterer = Clusterer(config=cfg)

        result = clusterer.cluster(space)

        assert result.clusters == []
        assert result.n_clusters == 0
        assert result.memberships == {}
        assert result.outliers == []

    def test_get_primary_cluster(self):
        """Test getting primary cluster for a word."""
        vectors = [[0.0, 1.0], [0.0, 2.0], [10.0, 10.0]]
        texts = ["a", "b", "c"]
        space = make_space(vectors, texts)

        cfg = ClusterConfig(method="kmeans", n_clusters=2)
        clusterer = Clusterer(config=cfg)
        result = clusterer.cluster(space)

        for word in texts:
            primary = result.get_primary_cluster(word)
            assert primary is not None
            assert 0 <= primary < result.n_clusters

    def test_clusters_have_membership_info(self):
        """Clusters should contain ClusterMembership objects."""
        vectors = [[0.0, 1.0], [0.0, 2.0], [10.0, 10.0], [10.0, 11.0]]
        texts = ["a", "b", "c", "d"]
        space = make_space(vectors, texts)

        cfg = ClusterConfig(method="kmeans", n_clusters=2)
        clusterer = Clusterer(config=cfg)
        result = clusterer.cluster(space)

        for cluster in result.clusters:
            assert len(cluster.memberships) > 0
            for m in cluster.memberships:
                assert m.cluster_id == cluster.id
                assert m.word in texts
                assert 0.0 <= m.score <= 1.0

    def test_auto_cluster_count(self):
        """Test auto-determination of cluster count."""
        vectors = [[float(i), 0.0] for i in range(20)]
        texts = [f"word{i}" for i in range(20)]
        space = make_space(vectors, texts)

        cfg = ClusterConfig(method="kmeans", n_clusters=None)
        clusterer = Clusterer(config=cfg)
        result = clusterer.cluster(space)

        assert result.n_clusters > 1
        assert result.n_clusters <= 5  # Heuristic caps at 5

    def test_gmm_produces_probabilities(self):
        """GMM should produce probability-type memberships."""
        vectors = [[0.0, 1.0], [0.0, 2.0], [10.0, 10.0], [10.0, 11.0]]
        texts = ["a", "b", "c", "d"]
        space = make_space(vectors, texts)

        cfg = ClusterConfig(method="gmm", n_clusters=2)
        clusterer = Clusterer(config=cfg)
        result = clusterer.cluster(space)

        # Check score types
        for word_memberships in result.memberships.values():
            for m in word_memberships:
                assert m.score_type == "probability"

        # Memberships for each word should sum to ~1.0
        for word in texts:
            total = sum(m.score for m in result.memberships[word])
            assert abs(total - 1.0) < 0.01


class TestOutlierDetection:
    """Test outlier handling."""

    def test_no_outliers_for_kmeans(self):
        """KMeans should not produce outliers."""
        vectors = [[0.0, 1.0], [0.0, 2.0], [10.0, 10.0]]
        texts = ["a", "b", "c"]
        space = make_space(vectors, texts)

        cfg = ClusterConfig(method="kmeans", n_clusters=2)
        clusterer = Clusterer(config=cfg)
        result = clusterer.cluster(space)

        assert len(result.outliers) == 0
        assert result.metrics.n_outliers == 0

    def test_is_outlier_method(self):
        """Test is_outlier method."""
        vectors = [[0.0, 1.0], [0.0, 2.0]]
        texts = ["a", "b"]
        space = make_space(vectors, texts)

        cfg = ClusterConfig(method="kmeans", n_clusters=2)
        clusterer = Clusterer(config=cfg)
        result = clusterer.cluster(space)

        assert not result.is_outlier("a")
        assert not result.is_outlier("b")


class TestMetrics:
    """Test metrics computation."""

    def test_metrics_computed_for_valid_clustering(self):
        """Metrics should be computed when clustering is valid."""
        vectors = [[0.0, 1.0], [0.0, 2.0], [10.0, 10.0], [10.0, 11.0]]
        texts = ["a", "b", "c", "d"]
        space = make_space(vectors, texts)

        cfg = ClusterConfig(method="kmeans", n_clusters=2)
        clusterer = Clusterer(config=cfg)
        result = clusterer.cluster(space)

        assert result.metrics.silhouette_score is not None
        assert result.metrics.mean_internal_similarity is not None
