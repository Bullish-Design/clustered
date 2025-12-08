"""Clustering and morphology utilities for Embeddy embeddings.

This package builds on `embeddy` and provides a small, cohesive
abstraction layer for working with embeddings as NumPy arrays,
performing clustering, and handling inflection/morphology.
"""

# src/clustered/__init__.py
from clustered.clustering import ClusterConfig, Clusterer
from clustered.models import (
    ClusteringResult,
    ClusterMembership,
    ClusterMetrics,
    WordCluster,
)
from clustered.morphology import InflectionGroup, Lemmatizer, SpacyLemmatizer
from clustered.space import EmbeddingSpace

__all__ = [
    "EmbeddingSpace",
    "ClusterConfig",
    "Clusterer",
    "ClusteringResult",
    "ClusterMembership",
    "ClusterMetrics",
    "WordCluster",
    "Lemmatizer",
    "SpacyLemmatizer",
    "InflectionGroup",
]

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
    """
