from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .morphology import InflectionGroup, Lemmatizer, group_inflections
from .space import EmbeddingSpace


class ClusterConfig(BaseModel):
    """Configuration for clustering operations."""

    method: Literal["kmeans", "hierarchical"] = Field(
        default="kmeans", description="Clustering algorithm to use"
    )
    metric: Literal["cosine", "euclidean"] = Field(
        default="cosine", description="Distance metric used for clustering"
    )
    n_clusters: int | None = Field(
        default=None,
        description="If set, use this number of clusters. If None, an internal heuristic is used.",
    )
    min_cluster_size: int = Field(
        default=2, description="Minimum cluster size to keep in the final result"
    )
    max_cluster_size: int | None = Field(
        default=None, description="Maximum cluster size; larger clusters are split or truncated"
    )
    random_state: int = Field(default=42, description="Random seed for deterministic clustering")
    deduplicate_inflections: bool = Field(
        default=True, description="Whether to collapse inflected forms via a lemmatizer"
    )


class ClusterMetrics(BaseModel):
    """Quality metrics for a clustering run."""

    silhouette_score: float | None = Field(default=None)
    mean_internal_similarity: float | None = Field(default=None)


class WordCluster(BaseModel):
    """Cluster of words identified by the clustering algorithm."""

    id: int
    words: list[str]
    centroid: str | None = None


class ClusteringResult(BaseModel):
    """Result of a clustering operation."""

    clusters: list[WordCluster]
    metrics: ClusterMetrics
    method: str
    n_clusters: int
    assignments: dict[str, list[int]]
    inflection_groups: list[InflectionGroup] | None = None


class Clusterer(BaseModel):
    """High-level clustering façade for EmbeddingSpace instances."""

    config: ClusterConfig

    model_config = {"arbitrary_types_allowed": True}

    def cluster(
        self,
        space: EmbeddingSpace,
        lemmatizer: Lemmatizer | None = None,
    ) -> ClusteringResult:
        """Cluster the given embedding space according to this config."""
        words = space.words
        vectors = space.as_numpy()

        if self.config.deduplicate_inflections and lemmatizer is not None:
            inflection_groups = group_inflections(words, lemmatizer)
            # map from representative -> indices of variants
            rep_to_indices: dict[str, list[int]] = {}
            for idx, word in enumerate(words):
                for group in inflection_groups:
                    if word in group.variants:
                        rep_to_indices.setdefault(group.representative, []).append(idx)
                        break
            chosen_indices = sorted({indices[0] for indices in rep_to_indices.values()})
            words_for_clustering = [words[i] for i in chosen_indices]
            vectors_for_clustering = vectors[chosen_indices]
        else:
            inflection_groups = None
            words_for_clustering = words
            vectors_for_clustering = vectors

        n_samples = len(words_for_clustering)
        if n_samples == 0:
            empty = ClusteringResult(
                clusters=[],
                metrics=ClusterMetrics(),
                method=self.config.method,
                n_clusters=0,
                assignments={},
                inflection_groups=inflection_groups,
            )
            return empty

        n_clusters = self.config.n_clusters or max(2, min(5, n_samples // 4))

        if self.config.method == "kmeans":
            labels = self._cluster_kmeans(vectors_for_clustering, n_clusters)
        elif self.config.method == "hierarchical":
            labels = self._cluster_hierarchical(vectors_for_clustering, n_clusters)
        else:
            raise ValueError(f"Unsupported clustering method: {self.config.method!r}")

        clusters: dict[int, list[str]] = {}
        for word, label in zip(words_for_clustering, labels):
            clusters.setdefault(int(label), []).append(word)

        word_to_labels: dict[str, list[int]] = {}
        for label, cluster_words in clusters.items():
            for word in cluster_words:
                word_to_labels.setdefault(word, []).append(label)

        word_clusters: list[WordCluster] = [
            WordCluster(id=cid, words=sorted(ws)) for cid, ws in sorted(clusters.items())
        ]

        metrics = ClusterMetrics()
        return ClusteringResult(
            clusters=word_clusters,
            metrics=metrics,
            method=self.config.method,
            n_clusters=len(word_clusters),
            assignments=word_to_labels,
            inflection_groups=inflection_groups,
        )

    # --- internal helpers -------------------------------------------------

    def _cluster_kmeans(self, vectors: "np.ndarray", n_clusters: int) -> "np.ndarray":
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.random_state,
            n_init="auto",
        )
        return kmeans.fit_predict(vectors)

    def _cluster_hierarchical(self, vectors: "np.ndarray", n_clusters: int) -> "np.ndarray":
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(n_clusters=n_clusters)
        return model.fit_predict(vectors)