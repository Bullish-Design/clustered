from __future__ import annotations

import clustered


def test_public_api_reexports():
    from clustered import (
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
    public = set(getattr(clustered, "__all__", []))
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
    assert not missing, f"Missing names from clustered.__all__: {sorted(missing)}"
