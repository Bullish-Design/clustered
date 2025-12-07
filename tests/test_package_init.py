from __future__ import annotations

import embeddy_clustering


def test_public_api_reexports():
    from embeddy_clustering import (
        EmbeddingSpace,
        ClusterConfig,
        Clusterer,
        ClusteringResult,
        WordCluster,
        ClusterMetrics,
        Lemmatizer,
        SpacyLemmatizer,
        InflectionGroup,
    )

    # Simple smoke-test that the symbols are importable
    assert EmbeddingSpace is not None
    assert ClusterConfig is not None
    assert Clusterer is not None
    assert ClusteringResult is not None
    assert WordCluster is not None
    assert ClusterMetrics is not None
    assert Lemmatizer is not None
    assert SpacyLemmatizer is not None
    assert InflectionGroup is not None


def test_all_contains_core_symbols():
    public = set(getattr(embeddy_clustering, "__all__", []))
    expected = {
        "EmbeddingSpace",
        "ClusterConfig",
        "Clusterer",
        "ClusteringResult",
        "WordCluster",
        "ClusterMetrics",
        "Lemmatizer",
        "SpacyLemmatizer",
        "InflectionGroup",
    }

    missing = expected - public
    assert not missing, f"Missing names from embeddy_clustering.__all__: {sorted(missing)}"
