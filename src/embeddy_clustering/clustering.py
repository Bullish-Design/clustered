from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .morphology import InflectionGroup, Lemmatizer, group_inflections
from .space import EmbeddingSpace


class ClusterConfig(BaseModel):
    """Configuration for clustering operations."""

    method: Literal["kmeans", "hierarchical"] = Field(
        default="kmeans",
        description="Clustering backend to use.",
    )
    metric: str = Field(
        default="cosine",
        description="Similarity metric (currently informational only).",
    )
    n_clusters: int | None = Field(
        default=None,
        description="Desired number of clusters. If None, an internal heuristic is used.",
    )
    min_cluster_size: int = Field(
        default=1,
        description="Reserved for future use; not currently enforced.",
    )
    max_cluster_size: int | None = Field(
        default=None,
        description="Reserved for future use; not currently enforced.",
    )
    random_state: int | None = Field(
        default=None,
        description="Random seed passed to sklearn where applicable.",
    )
    deduplicate_inflections: bool = Field(
        default=False,
        description=(
            "If True and a lemmatizer is provided, cluster representative "
            "forms for inflection groups rather than raw tokens."
        ),
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
    """High-level clustering façade for :class:`EmbeddingSpace`."""

    config: ClusterConfig

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(
        self,
        space: EmbeddingSpace,
        lemmatizer: Lemmatizer | None = None,
    ) -> ClusteringResult:
        """Cluster the given embedding space according to this config."""
        words = space.words
        vectors_all = space.as_numpy()

        # --- choose the points we actually cluster over ----------------
        if self.config.deduplicate_inflections and lemmatizer is not None:
            inflection_groups = group_inflections(words, lemmatizer)

            rep_to_index: dict[str, int] = {}
            for group in inflection_groups:
                for idx, word in enumerate(words):
                    if word in group.variants:
                        rep_to_index.setdefault(group.representative, idx)
                        break

            chosen_indices = sorted(rep_to_index.values())
            words_for_clustering = [words[i] for i in chosen_indices]
            vectors_for_clustering = vectors_all[chosen_indices]
        else:
            inflection_groups = None
            words_for_clustering = words
            vectors_for_clustering = vectors_all

        n_samples = len(words_for_clustering)
        if n_samples == 0:
            return ClusteringResult(
                clusters=[],
                metrics=ClusterMetrics(),
                method=self.config.method,
                n_clusters=0,
                assignments={},
                inflection_groups=inflection_groups,
            )

        # --- choose cluster count --------------------------------------
        if self.config.n_clusters is not None:
            n_clusters = max(1, min(self.config.n_clusters, n_samples))
        else:
            heuristic = max(2, min(5, n_samples // 4))
            n_clusters = max(1, min(heuristic, n_samples))

        # --- run clustering backend ------------------------------------
        if self.config.method == "kmeans":
            labels = self._cluster_kmeans(vectors_for_clustering, n_clusters)
        elif self.config.method == "hierarchical":
            labels = self._cluster_hierarchical(vectors_for_clustering, n_clusters)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported clustering method: {self.config.method!r}")

        labels = np.asarray(labels, dtype=int)
        unique_labels = np.unique(labels)

        # --- build clusters & assignments ------------------------------
        clusters: list[WordCluster] = []
        assignments: dict[str, list[int]] = {}

        for cid in sorted(int(l) for l in unique_labels):
            member_idx = np.where(labels == cid)[0]
            if member_idx.size == 0:
                continue

            cluster_words = [words_for_clustering[i] for i in member_idx]
            cluster_vectors = vectors_for_clustering[member_idx]

            centroid_word = self._pick_centroid_word(cluster_words, cluster_vectors)
            clusters.append(
                WordCluster(
                    id=cid,
                    words=sorted(set(cluster_words)),
                    centroid=centroid_word,
                )
            )
            for w in cluster_words:
                assignments.setdefault(w, []).append(cid)

        metrics = self._compute_metrics(vectors_for_clustering, labels)

        return ClusteringResult(
            clusters=clusters,
            metrics=metrics,
            method=self.config.method,
            n_clusters=len(clusters),
            assignments=assignments,
            inflection_groups=inflection_groups,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cluster_kmeans(self, vectors: np.ndarray, n_clusters: int) -> np.ndarray:
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.random_state,
            n_init=10,
        )
        return kmeans.fit_predict(vectors)

    def _cluster_hierarchical(self, vectors: np.ndarray, n_clusters: int) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(n_clusters=n_clusters)
        return model.fit_predict(vectors)

    def _pick_centroid_word(self, words: list[str], vectors: np.ndarray) -> str | None:
        if len(words) == 0 or vectors.size == 0:
            return None

        centroid = vectors.mean(axis=0)
        norms = np.linalg.norm(vectors, axis=1)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm == 0.0:
            return sorted(words)[0]

        norms[norms == 0.0] = 1.0
        sims = (vectors @ centroid) / (norms * centroid_norm)
        best_idx = int(np.argmax(sims))
        return words[best_idx]

    def _compute_metrics(self, vectors: np.ndarray, labels: np.ndarray) -> ClusterMetrics:
        metrics = ClusterMetrics()

        unique_labels = np.unique(labels)
        if vectors.shape[0] < 2 or unique_labels.size < 2:
            return metrics

        try:  # pragma: no cover
            from sklearn.metrics import silhouette_score

            metrics.silhouette_score = float(
                silhouette_score(vectors, labels, metric="cosine")
            )
        except Exception:  # pragma: no cover
            metrics.silhouette_score = None

        sims: list[float] = []
        for label in unique_labels:
            idx = np.where(labels == label)[0]
            if idx.size <= 1:
                continue

            sub = vectors[idx]
            norms = np.linalg.norm(sub, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            sub_norm = sub / norms
            sim_matrix = sub_norm @ sub_norm.T

            n = sim_matrix.shape[0]
            tri = sim_matrix[np.triu_indices(n, k=1)]
            sims.extend(tri.tolist())

        if sims:
            metrics.mean_internal_similarity = float(np.mean(sims))

        return metrics
