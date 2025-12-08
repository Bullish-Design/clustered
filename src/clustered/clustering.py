# src/clustered/clustering.py
from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict

from clustered.models import ClusteringResult, ClusterMembership, ClusterMetrics, WordCluster
from clustered.morphology import InflectionGroup, Lemmatizer, group_inflections
from clustered.space import EmbeddingSpace


class ClusterConfig(BaseModel):
    """Configuration for clustering operations."""

    method: Literal["kmeans", "hierarchical", "fuzzy", "threshold", "gmm", "hdbscan", "dbscan"] = Field(
        default="kmeans",
        description="Clustering algorithm to use",
    )
    n_clusters: int | None = Field(
        default=None,
        description="Number of clusters (None = auto-determine for HDBSCAN/DBSCAN)",
    )
    random_state: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    deduplicate_inflections: bool = Field(
        default=False,
        description="Group inflectional variants before clustering",
    )

    # Method-specific parameters
    fuzzy_m: float = Field(
        default=2.0,
        ge=1.0,
        description="Fuzziness parameter for fuzzy c-means (higher = fuzzier)",
    )
    threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum membership score for threshold-based clustering",
    )
    dbscan_eps: float = Field(
        default=0.5,
        gt=0.0,
        description="DBSCAN epsilon (maximum distance for neighborhood)",
    )
    dbscan_min_samples: int = Field(
        default=2,
        ge=1,
        description="DBSCAN minimum samples per cluster",
    )
    hdbscan_min_cluster_size: int = Field(
        default=3,
        ge=2,
        description="HDBSCAN minimum cluster size",
    )
    gmm_covariance_type: Literal["full", "tied", "diag", "spherical"] = Field(
        default="full",
        description="GMM covariance type",
    )

    @field_validator("n_clusters")
    @classmethod
    def validate_n_clusters(cls, value: int | None, info) -> int | None:
        """Validate n_clusters based on method."""
        if value is not None and value < 1:
            raise ValueError("n_clusters must be at least 1")
        return value


class Clusterer(BaseModel):
    """Unified clusterer using membership-based architecture."""

    config: ClusterConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def cluster(
        self,
        space: EmbeddingSpace,
        lemmatizer: Lemmatizer | None = None,
    ) -> ClusteringResult:
        """Cluster the embedding space.

        Args:
            space: EmbeddingSpace containing word embeddings
            lemmatizer: Optional lemmatizer for inflection deduplication

        Returns:
            ClusteringResult with membership information
        """
        words = space.words
        vectors_all = space.as_numpy()

        # Handle inflection deduplication
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
                memberships={},
                outliers=[],
                method=self.config.method,
                n_clusters=0,
                metrics=ClusterMetrics(),
            )

        # Determine cluster count
        if self.config.n_clusters is not None:
            n_clusters = max(1, min(self.config.n_clusters, n_samples))
        else:
            if self.config.method in {"hdbscan", "dbscan"}:
                n_clusters = None  # Auto-determined by algorithm
            else:
                heuristic = max(2, min(5, n_samples // 4))
                n_clusters = max(1, min(heuristic, n_samples))

        # Compute membership matrix: (n_words, n_clusters or variable)
        memberships_matrix, actual_n_clusters, outlier_mask = self._compute_memberships(
            vectors_for_clustering,
            n_clusters,
        )

        # Build result
        result = self._build_result(
            words_for_clustering,
            vectors_for_clustering,
            memberships_matrix,
            actual_n_clusters,
            outlier_mask,
            inflection_groups,
        )

        return result

    def _compute_memberships(
        self,
        vectors: np.ndarray,
        n_clusters: int | None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Compute membership matrix for the given method.

        Returns:
            (memberships_matrix, actual_n_clusters, outlier_mask)
            - memberships_matrix: (n_words, n_clusters) with scores in [0, 1]
            - actual_n_clusters: number of clusters produced
            - outlier_mask: boolean array marking outliers
        """
        match self.config.method:
            case "kmeans":
                return self._kmeans_memberships(vectors, n_clusters)
            case "hierarchical":
                return self._hierarchical_memberships(vectors, n_clusters)
            case "fuzzy":
                return self._fuzzy_memberships(vectors, n_clusters)
            case "threshold":
                return self._threshold_memberships(vectors, n_clusters)
            case "gmm":
                return self._gmm_memberships(vectors, n_clusters)
            case "hdbscan":
                return self._hdbscan_memberships(vectors)
            case "dbscan":
                return self._dbscan_memberships(vectors)
            case _:
                raise ValueError(f"Unsupported method: {self.config.method}")

    def _kmeans_memberships(
        self,
        vectors: np.ndarray,
        n_clusters: int,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Hard clustering with KMeans: binary memberships."""
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(vectors)

        # Convert to binary membership matrix
        memberships = np.zeros((len(vectors), n_clusters))
        memberships[np.arange(len(vectors)), labels] = 1.0

        outliers = np.zeros(len(vectors), dtype=bool)
        return memberships, n_clusters, outliers

    def _hierarchical_memberships(
        self,
        vectors: np.ndarray,
        n_clusters: int,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Hard clustering with hierarchical: binary memberships."""
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(vectors)

        # Convert to binary membership matrix
        memberships = np.zeros((len(vectors), n_clusters))
        memberships[np.arange(len(vectors)), labels] = 1.0

        outliers = np.zeros(len(vectors), dtype=bool)
        return memberships, n_clusters, outliers

    def _fuzzy_memberships(
        self,
        vectors: np.ndarray,
        n_clusters: int,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Fuzzy c-means: continuous memberships."""
        import skfuzzy as fuzz

        # Transpose for skfuzzy (expects features x samples)
        vectors_t = vectors.T

        # Run fuzzy c-means
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            vectors_t,
            c=n_clusters,
            m=self.config.fuzzy_m,
            error=0.005,
            maxiter=1000,
            init=None,
            seed=self.config.random_state,
        )

        # Transpose back to (samples, clusters)
        memberships = u.T

        outliers = np.zeros(len(vectors), dtype=bool)
        return memberships, n_clusters, outliers

    def _threshold_memberships(
        self,
        vectors: np.ndarray,
        n_clusters: int,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Threshold-based: binary but allows multiple clusters per word."""
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import cosine_similarity

        # Get centroids via KMeans
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.random_state,
            n_init=10,
        )
        kmeans.fit(vectors)
        centers = kmeans.cluster_centers_

        # Compute similarities to centroids
        sims = cosine_similarity(vectors, centers)

        # Binary membership above threshold
        memberships = (sims >= self.config.threshold).astype(float)

        # Ensure every word in at least one cluster
        no_assignment = memberships.sum(axis=1) == 0
        if no_assignment.any():
            best_clusters = sims[no_assignment].argmax(axis=1)
            memberships[no_assignment, best_clusters] = 1.0

        outliers = np.zeros(len(vectors), dtype=bool)
        return memberships, n_clusters, outliers

    def _gmm_memberships(
        self,
        vectors: np.ndarray,
        n_clusters: int,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Gaussian Mixture Model: probabilistic memberships."""
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=self.config.gmm_covariance_type,
            random_state=self.config.random_state,
        )
        gmm.fit(vectors)

        # Predict probabilities (already normalized)
        memberships = gmm.predict_proba(vectors)

        outliers = np.zeros(len(vectors), dtype=bool)
        return memberships, n_clusters, outliers

    def _hdbscan_memberships(
        self,
        vectors: np.ndarray,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """HDBSCAN: auto-determines clusters, marks outliers."""
        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "HDBSCAN requires 'hdbscan' package. Install with: pip install hdbscan"
            )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            metric="euclidean",
        )
        clusterer.fit(vectors)

        labels = clusterer.labels_
        # HDBSCAN uses -1 for outliers
        outlier_mask = labels == -1
        unique_labels = np.unique(labels[~outlier_mask])
        n_clusters = len(unique_labels)

        if n_clusters == 0:
            # All outliers
            memberships = np.zeros((len(vectors), 1))
            return memberships, 0, outlier_mask

        # Build membership matrix
        memberships = np.zeros((len(vectors), n_clusters))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        # Use probabilities if available
        if hasattr(clusterer, "probabilities_"):
            for i, (label, prob) in enumerate(zip(labels, clusterer.probabilities_)):
                if label != -1:
                    memberships[i, label_to_idx[label]] = prob
        else:
            # Binary memberships
            for i, label in enumerate(labels):
                if label != -1:
                    memberships[i, label_to_idx[label]] = 1.0

        return memberships, n_clusters, outlier_mask

    def _dbscan_memberships(
        self,
        vectors: np.ndarray,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """DBSCAN: density-based, marks outliers."""
        from sklearn.cluster import DBSCAN

        dbscan = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
            metric="euclidean",
        )
        labels = dbscan.fit_predict(vectors)

        # DBSCAN uses -1 for outliers
        outlier_mask = labels == -1
        unique_labels = np.unique(labels[~outlier_mask])
        n_clusters = len(unique_labels)

        if n_clusters == 0:
            memberships = np.zeros((len(vectors), 1))
            return memberships, 0, outlier_mask

        # Binary memberships
        memberships = np.zeros((len(vectors), n_clusters))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        for i, label in enumerate(labels):
            if label != -1:
                memberships[i, label_to_idx[label]] = 1.0

        return memberships, n_clusters, outlier_mask

    def _build_result(
        self,
        words: list[str],
        vectors: np.ndarray,
        memberships_matrix: np.ndarray,
        n_clusters: int,
        outlier_mask: np.ndarray,
        inflection_groups: list[InflectionGroup] | None,
    ) -> ClusteringResult:
        """Build ClusteringResult from membership matrix."""
        n_words = len(words)

        # Build memberships map and cluster data
        memberships_map: dict[str, list[ClusterMembership]] = {}
        clusters_data: dict[int, list[tuple[str, int, float]]] = {
            i: [] for i in range(n_clusters)
        }
        outliers: list[str] = []

        score_type = self._get_score_type()

        for word_idx, word in enumerate(words):
            if outlier_mask[word_idx]:
                outliers.append(word)
                memberships_map[word] = []
                continue

            word_memberships = []
            for cluster_id in range(n_clusters):
                score = float(memberships_matrix[word_idx, cluster_id])
                if score > 0:
                    membership = ClusterMembership(
                        word=word,
                        cluster_id=cluster_id,
                        score=score,
                        score_type=score_type,
                    )
                    word_memberships.append(membership)
                    clusters_data[cluster_id].append((word, word_idx, score))

            memberships_map[word] = sorted(word_memberships, key=lambda m: -m.score)

        # Build clusters
        clusters = []
        for cluster_id in range(n_clusters):
            members_data = clusters_data[cluster_id]
            if not members_data:
                continue

            cluster_memberships = [
                ClusterMembership(
                    word=word,
                    cluster_id=cluster_id,
                    score=score,
                    score_type=score_type,
                )
                for word, _, score in members_data
            ]

            # Pick centroid from strong members
            strong_indices = [idx for _, idx, score in members_data if score >= 0.5]
            if strong_indices:
                centroid_word = self._pick_centroid_word(
                    [words[i] for i in strong_indices],
                    vectors[strong_indices],
                )
            else:
                # Fall back to any member
                indices = [idx for _, idx, _ in members_data]
                centroid_word = self._pick_centroid_word(
                    [words[i] for i in indices],
                    vectors[indices],
                )

            clusters.append(
                WordCluster(
                    id=cluster_id,
                    memberships=sorted(cluster_memberships, key=lambda m: -m.score),
                    centroid=centroid_word,
                )
            )

        # Compute metrics
        metrics = self._compute_metrics(vectors, memberships_matrix, outlier_mask)
        metrics.n_outliers = len(outliers)

        return ClusteringResult(
            clusters=clusters,
            memberships=memberships_map,
            outliers=outliers,
            method=self.config.method,
            n_clusters=len(clusters),
            metrics=metrics,
            inflection_groups=inflection_groups,
        )

    def _get_score_type(self) -> str:
        """Determine score type based on method."""
        if self.config.method in {"kmeans", "hierarchical", "threshold", "dbscan", "hdbscan"}:
            return "binary"
        elif self.config.method == "gmm":
            return "probability"
        elif self.config.method == "fuzzy":
            return "similarity"
        return "similarity"

    def _pick_centroid_word(self, words: list[str], vectors: np.ndarray) -> str | None:
        """Pick word closest to centroid of given vectors."""
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

    def _compute_metrics(
        self,
        vectors: np.ndarray,
        memberships_matrix: np.ndarray,
        outlier_mask: np.ndarray,
    ) -> ClusterMetrics:
        """Compute quality metrics."""
        metrics = ClusterMetrics()

        # Filter out outliers
        non_outliers = ~outlier_mask
        if non_outliers.sum() < 2:
            return metrics

        vectors_filtered = vectors[non_outliers]
        memberships_filtered = memberships_matrix[non_outliers]

        # Hard labels for silhouette (use strongest cluster)
        if memberships_filtered.shape[1] > 0:
            hard_labels = memberships_filtered.argmax(axis=1)
            unique_labels = np.unique(hard_labels)

            if len(unique_labels) > 1:
                try:
                    from sklearn.metrics import davies_bouldin_score, silhouette_score

                    metrics.silhouette_score = float(
                        silhouette_score(vectors_filtered, hard_labels, metric="cosine")
                    )
                    metrics.davies_bouldin_score = float(
                        davies_bouldin_score(vectors_filtered, hard_labels)
                    )
                except Exception:
                    pass

        # Mean internal similarity
        sims: list[float] = []
        for label in unique_labels:
            idx = np.where(hard_labels == label)[0]
            if idx.size <= 1:
                continue

            sub = vectors_filtered[idx]
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
