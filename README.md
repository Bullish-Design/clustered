# Clustered (embeddy-clustering)

Clustering and morphology utilities built on top of **Embeddify**.

`embeddy-clustering` wraps batches of embeddings from the `embeddify` library
into an `EmbeddingSpace`, and provides:

- Simple helpers for working with embeddings as NumPy arrays.
- Clustering via a small façade around scikit-learn (KMeans / Agglomerative).
- Optional lemma / inflection grouping via pluggable lemmatizers.
- Basic clustering quality metrics (silhouette score, intra-cluster similarity).
- Optional cluster centroids expressed as representative words.

The goal is to give Embeddify a small, composable “clustering toolkit” without
pulling in a heavy framework.

---

## Installation

This library assumes you already have **Embeddify** available.

Required dependencies (also listed in `pyproject.toml`):

- `embeddify`
- `numpy`
- `pydantic>=2`
- `scikit-learn`
- `spacy` (optional, only needed for `SpacyLemmatizer`)

Install them using your preferred Python packaging workflow, then install this
package (`embeddy-clustering`) into the same environment.

---

## Quickstart

### 1. Obtain embeddings with Embeddify

```python
from embeddify import Embedder, EmbedderConfig

config = EmbedderConfig(
    model_path="/models/all-MiniLM-L6-v2",
    device="cpu",
    normalize_embeddings=True,
)

embedder = Embedder(config=config)

words = ["cats", "cat", "dogs", "dog", "running", "runs"]
result = embedder.encode(words)  # EmbeddingResult
```

### 2. Wrap in `EmbeddingSpace` and run clustering

```python
from embeddy_clustering import (
    EmbeddingSpace,
    ClusterConfig,
    Clusterer,
)

space = EmbeddingSpace(result=result)

config = ClusterConfig(
    method="kmeans",          # or "hierarchical"
    n_clusters=2,             # or leave as None to use a heuristic
    metric="cosine",          # used for metrics such as silhouette score
    deduplicate_inflections=False,
)

clusterer = Clusterer(config=config)
clustering = clusterer.cluster(space)

for cluster in clustering.clusters:
    print(f"Cluster {cluster.id} (centroid={cluster.centroid!r}): {cluster.words}")

print("silhouette:", clustering.metrics.silhouette_score)
print("mean internal similarity:", clustering.metrics.mean_internal_similarity)
```

### 3. Clustering with inflection / lemma deduplication

```python
from embeddy_clustering import SpacyLemmatizer

space = EmbeddingSpace(result=result)

config = ClusterConfig(
    method="kmeans",
    n_clusters=2,
    deduplicate_inflections=True,
)

lemmatizer = SpacyLemmatizer(model_name="en_core_web_sm")
clusterer = Clusterer(config=config)

clustering = clusterer.cluster(space, lemmatizer=lemmatizer)

print("Clusters (by representative form):")
for cluster in clustering.clusters:
    print(f"Cluster {cluster.id}: {cluster.words}")

print("Inflection groups:")
for group in clustering.inflection_groups or []:
    print(group.lemma, "->", group.variants)
```

When `deduplicate_inflections=True` and a `lemmatizer` is supplied:

- Surface forms are grouped into `InflectionGroup` objects by lemma.
- Only one representative form per lemma is used in the clustering run.
- `ClusteringResult.inflection_groups` exposes the full mapping back to variants.
- `ClusteringResult.clusters` and `assignments` are expressed in terms of the
  representative forms.

---

## API Overview

### EmbeddingSpace

```python
from embeddify import EmbeddingResult
from embeddy_clustering import EmbeddingSpace

space = EmbeddingSpace(result=embedding_result)
```

Fields:

- `result: EmbeddingResult` – underlying Embeddify result.
- `metric: Literal["cosine"]` – similarity interpretation (currently only cosine).

Main helpers:

- `space.words -> list[str]`  
  Returns the `text` attribute of each embedding.

- `space.dimensions -> int`  
  Dimensionality of each embedding vector.

- `space.as_numpy() -> np.ndarray`  
  Returns a stacked `(n_samples, dimensions)` float array.

- `space.similarity_matrix() -> np.ndarray`  
  Full pairwise cosine similarity matrix, shape `(n_samples, n_samples)`.

- `space.neighbors(index: int, top_k: int = 10) -> list[tuple[int, float]]`  
  Returns a list of `(neighbor_index, similarity)` pairs, sorted by descending
  similarity, excluding the element itself.

### Morphology / Inflection utilities

Located in `embeddy_clustering.morphology` and re-exported at the package root.

- **Lemmatizer (Protocol)**  
  Any object with `lemma(self, word: str) -> str` can act as a lemmatizer.

- **InflectionGroup**

  ```python
  from embeddy_clustering import InflectionGroup

  group = InflectionGroup(
      lemma="cat",
      representative="cat",
      variants=["Cats", "cat"],
  )
  ```

  Fields:

  - `lemma: str` – canonical lemma.
  - `representative: str` – representative surface form used in clustering.
  - `variants: list[str]` – all observed surface forms for this lemma.

- **group_inflections(words: list[str], lemmatizer: Lemmatizer) -> list[InflectionGroup]**  
  Groups words by lemma and chooses the representative form as the shortest
  variant (breaking ties lexicographically).

- **SpacyLemmatizer**

  ```python
  from embeddy_clustering import SpacyLemmatizer

  lemmatizer = SpacyLemmatizer(model_name="en_core_web_sm")
  lemma = lemmatizer.lemma("cats")  # -> "cat"
  ```

  - Lazily loads the spaCy model on first call to `.lemma(...)`.
  - Constructing `SpacyLemmatizer` does **not** require spaCy to be installed yet.

### Clustering

Located in `embeddy_clustering.clustering` and re-exported at the package root.

#### ClusterConfig

```python
from embeddy_clustering import ClusterConfig

config = ClusterConfig(
    method="kmeans",           # or "hierarchical"
    metric="cosine",           # used when computing quality metrics
    n_clusters=None,           # if None, a simple heuristic is used
    min_cluster_size=2,        # currently reserved for future use
    max_cluster_size=None,     # reserved
    random_state=42,
    deduplicate_inflections=True,
)
```

Notes:

- If `n_clusters` is `None`, the number of clusters is chosen as:

  ```python
  n_clusters = max(2, min(5, n_samples // 4))
  ```

- `min_cluster_size` / `max_cluster_size` are present as hooks for more advanced
  post-processing strategies and are not enforced by the current implementation.

#### WordCluster

```python
from embeddy_clustering import WordCluster

cluster = WordCluster(id=0, words=["cats", "cat"], centroid="cat")
```

Fields:

- `id: int` – cluster id (0-based).
- `words: list[str]` – the words contained in the cluster.
- `centroid: str | None` – representative word for the cluster.

For cosine metric, the centroid is the word whose vector is closest (by cosine
similarity) to the mean vector of all cluster members. For Euclidean metric, it
is the closest in L2 distance.

#### ClusterMetrics

```python
from embeddy_clustering import ClusterMetrics

metrics = ClusterMetrics(
    silhouette_score=0.42,
    mean_internal_similarity=0.87,
)
```

Fields:

- `silhouette_score: float | None` – global silhouette score in [-1, 1],
  computed using scikit-learn if available and if more than one cluster is
  present.

- `mean_internal_similarity: float | None` – average pairwise cosine similarity
  within clusters, aggregated across all clusters (only available when the
  underlying `EmbeddingSpace` uses cosine similarity).

If metrics cannot be computed (too little data, or missing dependencies),
they are left as `None`.

#### ClusteringResult

```python
from embeddy_clustering import ClusteringResult

result = ClusteringResult(
    clusters=[...],
    metrics=ClusterMetrics(),
    method="kmeans",
    n_clusters=2,
    assignments={"cat": [0]},
    inflection_groups=[...],
)
```

Fields:

- `clusters: list[WordCluster]` – clusters sorted by cluster id.
- `metrics: ClusterMetrics` – best-effort quality metrics.
- `method: str` – clustering method used (`"kmeans"` or `"hierarchical"`).
- `n_clusters: int` – number of clusters in the result.
- `assignments: dict[str, list[int]]` – mapping from word to cluster ids.
- `inflection_groups: list[InflectionGroup] | None` – inflection / lemma
  mapping information when `deduplicate_inflections=True`, otherwise `None`.

#### Clusterer

```python
from embeddy_clustering import Clusterer, ClusterConfig, EmbeddingSpace

space = EmbeddingSpace(result=embedding_result)
cfg = ClusterConfig(method="kmeans", n_clusters=3)
clusterer = Clusterer(config=cfg)

result = clusterer.cluster(space)
```

Behaviour:

- For an empty `EmbeddingSpace`, returns a `ClusteringResult` with no clusters
  and all metrics set to `None`.
- With `deduplicate_inflections=True` and a lemmatizer, clustering is performed
  on lemma representatives while still exposing full inflection groups.
- Silhouette score is computed when scikit-learn is available and the data has
  at least two clusters.
- Mean internal similarity is computed from the cosine similarity matrix of the
  `EmbeddingSpace`.

---

## Development & Testing

A small pytest suite is included under `tests/` and covers:

- Public API exports from `embeddy_clustering`.
- `EmbeddingSpace` helpers and similarity / neighbor behaviour.
- Morphology utilities (`group_inflections`, `InflectionGroup`, `SpacyLemmatizer`).
- Clustering behaviour, including:
  - Handling of empty spaces.
  - Construction of clusters and assignments.
  - Integration of inflection deduplication.
  - Population of quality metrics where applicable.

To run the tests, install the optional `dev` dependencies and execute `pytest`
from the project root.
