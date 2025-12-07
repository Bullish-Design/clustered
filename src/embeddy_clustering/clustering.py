from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .morphology import InflectionGroup, Lemmatizer, group_inflections
from .space import EmbeddingSpace


class ClusterConfig(BaseModel):
    """Configuration for clustering operations.

    Parameters
    ----------
    method:
        Clustering strategy. Currently ``"kmeans"`` and ``"hierarchical"``
        are supported.
    metric:
        Distance metric used when computing quality metrics such as the
        silhouette score. This does *not* currently change the behaviour of
        the underlying scikit-learn estimators, which use their own defaults.
    n_clusters:
        Desired number of clusters. If ``None``, a simple heuristic based on
        the number of samples is used.
    min_cluster_size:
        Present for future use – not currently enforced. Small clusters are
        still returned in the result.
    max_cluster_size:
        Present for future use – not currently enforced.
    random_state:
        Optional random seed forwarded to algorithms that support it.
    deduplicate_inflections:
        If ``True`` and a :class:`~embeddy_clustering.morphology.Lemmatizer`
        is provided, inflected forms are grouped by lemma and only a single
        representative form per lemma is clustered.
    """

    method: Literal["kmeans", "hierarchical"] = Field(
        default="kmeans", description="Clustering algorithm to use"
    )
    metric: Literal["cosine", "euclidean"] = Field(
        default="cosine", description="Distance metric used for clustering metrics"
    )
    n_clusters: int | None = Field(
        default=None,
        description="If set, use this number of clusters. If None, an internal heuristic is used.",
    )
    min_cluster_size: int = Field(
        default=2, description="Reserved: minimum cluster size to keep in the final result"
    )
    max_cluster_size: int | None = Field(
        default=None, description="Reserved: maximum cluster size to keep in the final result"
    )
    random_state: int | None = Field(default=None, description="Random seed for clustering algorithms")
    deduplicate_inflections: bool = Field(
        default=True,
        description=(
            "If True and a lemmatizer is provided, cluster over lemma representatives "
            "instead of raw surface forms."
        ),
    )


@dataclass
class ClusterMetrics:
    """Quality metrics for a clustering run.

    All metrics are optional and left as ``None`` if they cannot be
    computed (for example when the input is too small or dependencies
    are missing).
    """

    silhouette_score: float | None = None
    """Global silhouette score in the range [-1, 1] if available."""

    mean_internal_similarity: float | None = None
    """Mean pairwise similarity *within* clusters (cosine-based, 0–1)."""



class WordCluster(BaseModel):
    """Cluster of words identified by the clustering algorithm."""

    id: int
    words: list[str]
    centroid: str | None = None
    """Optional representative word for this cluster."""


class ClusteringResult(BaseModel):
    """Result of a clustering operation."""

    clusters: list[WordCluster]
    metrics: ClusterMetrics
    method: str
    n_clusters: int
    assignments: dict[str, list[int]]
    inflection_groups: list[InflectionGroup] | None = None


class Clusterer(BaseModel):
    """High-level clustering façade for :class:`EmbeddingSpace` instances."""

    config: ClusterConfig

    model_config = {"arbitrary_types_allowed": True}

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def cluster(
        self,
        space: EmbeddingSpace,
        lemmatizer: Lemmatizer | None = None,
    ) -> ClusteringResult:
        """Cluster the given embedding space according to this config.

        When ``deduplicate_inflections`` is enabled and a ``lemmatizer`` is
        provided, clustering is performed on lemma *representatives* only,
        and the lemma mapping is returned as ``inflection_groups``.
        """
        words = space.words
        vectors = space.as_numpy()

        # --- optional lemma-based deduplication --------------------------- #
        if self.config.deduplicate_inflections and lemmatizer is not None:
            inflection_groups = group_inflections(words, lemmatizer)

            # map from representative -> indices of variants in the original space
            rep_to_indices: dict[str, list[int]] = {}
            for idx, word in enumerate(words):
                for group in inflection_groups:
                    if word in group.variants:
                        rep_to_indices.setdefault(group.representative, []).append(idx)
                        break

            # we only keep a single index per representative for clustering
            chosen_indices = sorted({indices[0] for indices in rep_to_indices.values()})
            words_for_clustering = [words[i] for i in chosen_indices]
            vectors_for_clustering = vectors[chosen_indices]
            indices_for_metrics = chosen_indices
        else:
            inflection_groups = None
            words_for_clustering = words
            vectors_for_clustering = vectors
            indices_for_metrics = None  # use all points

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

        # --- choose number of clusters ----------------------------------- #
        n_clusters = self.config.n_clusters or max(2, min(5, n_samples // 4))

        # --- obtain labels from underlying algorithm --------------------- #
        if self.config.method == "kmeans":
            labels = self._cluster_kmeans(vectors_for_clustering, n_clusters)
        elif self.config.method == "hierarchical":
            labels = self._cluster_hierarchical(vectors_for_clustering, n_clusters)
        else:  # pragma: no cover - validated by typing
            raise ValueError(f"Unsupported clustering method: {self.config.method!r}")

        labels = np.asarray(labels, dtype=int)
        if labels.shape[0] != n_samples:  # pragma: no cover - defensive
            raise ValueError("Clustering backend returned a label array of unexpected shape")

        # --- build clusters and assignments ------------------------------ #
        clusters: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(i)

        # map from word -> list of cluster ids
        word_to_labels: dict[str, list[int]] = {}
        for cid, member_indices in clusters.items():
            for i in member_indices:
                word = words_for_clustering[i]
                word_to_labels.setdefault(word, []).append(cid)

        # choose centroids per cluster in terms of the original words
        word_clusters: list[WordCluster] = []
        for cid, member_indices in sorted(clusters.items()):
            member_words = [words_for_clustering[i] for i in member_indices]
            centroid_word = self._choose_centroid_word(vectors_for_clustering, words_for_clustering, member_indices)
            word_clusters.append(
                WordCluster(id=cid, words=sorted(member_words), centroid=centroid_word)
            )

        # --- quality metrics --------------------------------------------- #
        metrics = self._compute_metrics(
            vectors_for_clustering=vectors_for_clustering,
            labels=labels,
            space=space,
            indices_for_metrics=indices_for_metrics,
        )

        return ClusteringResult(
            clusters=word_clusters,
            metrics=metrics,
            method=self.config.method,
            n_clusters=len(word_clusters),
            assignments=word_to_labels,
            inflection_groups=inflection_groups,
        )

    # --------------------------------------------------------------------- #
    # Metrics & helpers
    # --------------------------------------------------------------------- #

    def _choose_centroid_word(
        self,
        vectors_for_clustering: np.ndarray,
        words_for_clustering: list[str],
        member_indices: list[int],
    ) -> str | None:
        """Choose a representative word for a cluster.

        For cosine metric, this selects the member whose vector has the highest
        cosine similarity to the mean cluster vector. For Euclidean metric,
        the member closest (in L2) to the mean is selected.
        """
        if not member_indices:
            return None

        member_vectors = vectors_for_clustering[member_indices]
        if member_vectors.size == 0:
            return None

        centroid_vec = member_vectors.mean(axis=0, keepdims=True)

        # fall back gracefully if something is degenerate
        if np.allclose(centroid_vec, 0):
            return words_for_clustering[member_indices[0]]

        if self.config.metric == "cosine":
            # cosine similarity between each member and the centroid
            norms_members = np.linalg.norm(member_vectors, axis=1, keepdims=True)
            norms_members[norms_members == 0.0] = 1.0
            norms_centroid = np.linalg.norm(centroid_vec, axis=1, keepdims=True)
            norms_centroid[norms_centroid == 0.0] = 1.0

            normalized_members = member_vectors / norms_members
            normalized_centroid = centroid_vec / norms_centroid

            sims = (normalized_members @ normalized_centroid.T).ravel()
            best_local_index = int(np.argmax(sims))
        else:
            # Euclidean distance to the centroid
            diffs = member_vectors - centroid_vec
            dists = np.linalg.norm(diffs, axis=1)
            best_local_index = int(np.argmin(dists))

        best_global_index = member_indices[best_local_index]
        return words_for_clustering[best_global_index]

    def _compute_metrics(
        self,
        vectors_for_clustering: np.ndarray,
        labels: np.ndarray,
        space: EmbeddingSpace,
        indices_for_metrics: list[int] | None,
    ) -> ClusterMetrics:
        """Compute high-level quality metrics for a clustering run.

        The function is intentionally robust:
        it returns a :class:`ClusterMetrics` object with metrics set to
        ``None`` when they cannot be reliably computed.
        """
        metrics = ClusterMetrics()

        if vectors_for_clustering.size == 0:
            return metrics

        unique_labels = np.unique(labels)
        if unique_labels.size <= 1:
            # silhouette and internal similarity are not meaningful
            return metrics

        n_samples = vectors_for_clustering.shape[0]
        if n_samples <= 2:
            return metrics

        # --- silhouette score (if sklearn is available) ------------------ #
        try:  # pragma: no cover - import logic
            from sklearn.metrics import silhouette_score  # type: ignore[import]
        except Exception:  # pragma: no cover - gracefully degrade
            silhouette_score = None  # type: ignore[assignment]

        if silhouette_score is not None:
            metric = "cosine" if self.config.metric == "cosine" else "euclidean"
            try:
                metrics.silhouette_score = float(
                    silhouette_score(vectors_for_clustering, labels, metric=metric)
                )
            except Exception:  # pragma: no cover - robustness
                metrics.silhouette_score = None

        # --- mean internal similarity (cosine based) --------------------- #
        # Only defined when the EmbeddingSpace can provide cosine similarities.
        if space.metric == "cosine":
            full_sims = space.similarity_matrix()
            if indices_for_metrics is not None:
                # take the subset corresponding to the points actually clustered
                sims = full_sims[np.ix_(indices_for_metrics, indices_for_metrics)]
            else:
                sims = full_sims

            # sims is square with shape (n_samples, n_samples) matching labels
            if sims.shape[0] == labels.shape[0]:
                total = 0.0
                count = 0
                for lab in unique_labels:
                    idxs = np.where(labels == lab)[0]
                    if idxs.size < 2:
                        continue
                    sub = sims[np.ix_(idxs, idxs)]
                    triu = np.triu_indices_from(sub, k=1)
                    vals = sub[triu]
                    total += float(vals.sum())
                    count += int(vals.size)
                if count > 0:
                    metrics.mean_internal_similarity = total / count

        return metrics

    # --------------------------------------------------------------------- #
    # Clustering backends
    # --------------------------------------------------------------------- #

    def _cluster_kmeans(self, vectors: "np.ndarray", n_clusters: int) -> "np.ndarray":
        from sklearn.cluster import KMeans  # type: ignore[import]

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.random_state,
            n_init="auto",
        )
        return kmeans.fit_predict(vectors)

    def _cluster_hierarchical(self, vectors: "np.ndarray", n_clusters: int) -> "np.ndarray":
        from sklearn.cluster import AgglomerativeClustering  # type: ignore[import]

        model = AgglomerativeClustering(n_clusters=n_clusters)
        return model.fit_predict(vectors)
