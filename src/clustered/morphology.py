from __future__ import annotations

from typing import Protocol, Any, runtime_checkable

from pydantic import BaseModel, Field, PrivateAttr


@runtime_checkable
class Lemmatizer(Protocol):
    """Simple protocol for lemmatization backends."""

    def lemma(self, word: str) -> str:  #  structural only
        ...


class InflectionGroup(BaseModel):
    """Represents a group of inflected forms for a single lemma."""

    lemma: str = Field(description="Canonical lemma")
    representative: str = Field(description="Representative surface form used for clustering")
    variants: list[str] = Field(default_factory=list, description="All surface forms observed for this lemma")


class SpacyLemmatizer(BaseModel):
    """spaCy-based lemmatizer implementation.

    The spaCy model is loaded lazily on first use to keep startup
    overhead small.
    """

    model_name: str = Field(default="en_core_web_sm", description="Name of the spaCy language model to load")
    _nlp: Any | None = PrivateAttr(default=None)

    def _ensure_loaded(self) -> Any:
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.model_name)
        return self._nlp

    def lemma(self, word: str) -> str:
        nlp = self._ensure_loaded()
        doc = nlp(word)
        if not doc:
            return word
        return doc[0].lemma_


def group_inflections(words: list[str], lemmatizer: Lemmatizer) -> list[InflectionGroup]:
    """Group surface forms into lemma-based inflection sets."""
    by_lemma: dict[str, list[str]] = {}
    for word in words:
        lemma = lemmatizer.lemma(word)
        by_lemma.setdefault(lemma, []).append(word)

    groups: list[InflectionGroup] = []
    for lemma, variants in by_lemma.items():
        # choose the "simplest" representative: shortest, then lexicographically
        representative = sorted(variants, key=lambda w: (len(w), w))[0]
        groups.append(
            InflectionGroup(
                lemma=lemma,
                representative=representative,
                variants=sorted(set(variants)),
            )
        )
    return groups
