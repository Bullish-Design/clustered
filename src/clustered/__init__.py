"""Clustering and morphology utilities for Embeddify embeddings.

This package builds on :mod:`embeddify` and provides a small, cohesive
abstraction layer for working with embeddings as NumPy arrays,
performing clustering, and handling inflection/morphology.
"""

from .space import EmbeddingSpace
from .clustering import ClusterConfig, Clusterer, ClusteringResult, WordCluster, ClusterMetrics
from .morphology import Lemmatizer, SpacyLemmatizer, InflectionGroup

__all__ = [
    "EmbeddingSpace",
    "ClusterConfig",
    "Clusterer",
    "ClusteringResult",
    "WordCluster",
    "ClusterMetrics",
    "Lemmatizer",
    "SpacyLemmatizer",
    "InflectionGroup",
]