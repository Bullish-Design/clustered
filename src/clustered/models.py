# src/clustered/models.py
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClusterMembership(BaseModel):
    """Represents a word's membership in a cluster."""

    word: str = Field(description="The word")
    cluster_id: int = Field(description="Cluster identifier")
    score: float = Field(ge=0.0, le=1.0, description="Membership strength/probability")
    score_type: Literal["probability", "similarity", "distance", "binary"] = Field(
        default="similarity",
        description="Interpretation of the score value",
    )

    @property
    def is_strong(self) -> bool:
        """True if membership score is above 0.5."""
        return self.score >= 0.5

    @property
    def is_primary(self) -> bool:
        """Alias for is_strong (highest membership for a word)."""
        return self.is_strong


class WordCluster(BaseModel):
    """A cluster with membership-aware members."""

    id: int = Field(description="Cluster identifier")
    memberships: list[ClusterMembership] = Field(
        default_factory=list,
        description="All members with their scores",
    )
    centroid: str | None = Field(
        default=None,
        description="Representative word closest to cluster center",
    )

    @property
    def words(self) -> list[str]:
        """All words with any membership in this cluster."""
        return [m.word for m in self.memberships]

    @property
    def core_words(self) -> list[str]:
        """Primary members only (score >= 0.5)."""
        return [m.word for m in self.memberships if m.is_strong]

    @property
    def size(self) -> int:
        """Number of members."""
        return len(self.memberships)


class ClusterMetrics(BaseModel):
    """Quality metrics for clustering."""

    silhouette_score: float | None = Field(
        default=None,
        description="Silhouette coefficient (for hard clustering)",
    )
    mean_internal_similarity: float | None = Field(
        default=None,
        description="Average similarity within clusters",
    )
    davies_bouldin_score: float | None = Field(
        default=None,
        description="Davies-Bouldin index (lower is better)",
    )
    n_outliers: int = Field(
        default=0,
        description="Number of outlier words",
    )


class ClusteringResult(BaseModel):
    """Result of a clustering operation with membership scores."""

    clusters: list[WordCluster] = Field(
        default_factory=list,
        description="Clusters with membership information",
    )
    memberships: dict[str, list[ClusterMembership]] = Field(
        default_factory=dict,
        description="Map from word to all cluster memberships (sorted by score)",
    )
    outliers: list[str] = Field(
        default_factory=list,
        description="Words that don't fit any cluster",
    )
    method: str = Field(description="Clustering method used")
    n_clusters: int = Field(description="Number of clusters produced")
    metrics: ClusterMetrics = Field(description="Quality metrics")
    inflection_groups: list[InflectionGroup] | None = Field(
        default=None,
        description="Inflection groupings if deduplication was used",
    )

    def get_primary_cluster(self, word: str) -> int | None:
        """Get the strongest cluster assignment for a word.

        Returns:
            Cluster ID with highest membership, or None if word is outlier
        """
        word_memberships = self.memberships.get(word, [])
        if not word_memberships:
            return None
        return max(word_memberships, key=lambda m: m.score).cluster_id

    def get_clusters_for_word(self, word: str, min_score: float = 0.0) -> list[int]:
        """Get all cluster IDs for a word above the score threshold.

        Args:
            word: Word to query
            min_score: Minimum membership score to include

        Returns:
            List of cluster IDs
        """
        return [
            m.cluster_id
            for m in self.memberships.get(word, [])
            if m.score >= min_score
        ]

    def is_outlier(self, word: str) -> bool:
        """Check if a word is an outlier."""
        return word in self.outliers
