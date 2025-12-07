from __future__ import annotations

from embeddy_clustering.morphology import (
    InflectionGroup,
    Lemmatizer,
    SpacyLemmatizer,
    group_inflections,
)


class DummyLemmatizer:
    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping

    def lemma(self, word: str) -> str:
        return self._mapping.get(word, word)


def test_group_inflections_basic():
    words = ["Cats", "cat", "dogs", "dog", "run"]
    mapping = {
        "Cats": "cat",
        "cat": "cat",
        "dogs": "dog",
        "dog": "dog",
        "run": "run",
    }
    groups = group_inflections(words, DummyLemmatizer(mapping))

    by_lemma = {g.lemma: g for g in groups}
    assert set(by_lemma) == {"cat", "dog", "run"}

    cat_group = by_lemma["cat"]
    assert isinstance(cat_group, InflectionGroup)
    assert cat_group.representative == "cat"
    assert sorted(cat_group.variants) == ["Cats", "cat"]


def test_spacy_lemmatizer_can_be_constructed_without_model():
    lemmatizer = SpacyLemmatizer()
    assert lemmatizer.model_name == "en_core_web_sm"
