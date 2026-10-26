"""Pure-function tests for the kinship label engine (no database needed)."""
from types import SimpleNamespace

import pytest

from app.services.kinship import (
    _format_ancestor_term,
    _format_aunt_uncle_term,
    _format_descendant_term,
    _format_niece_nephew_term,
    _gender_from_meta,
    _gendered_variant,
)


@pytest.mark.parametrize("steps,expected", [
    (1, "parent"),
    (2, "grandparent"),
    (3, "great-grandparent"),
    (5, "great-great-great-grandparent"),
])
def test_ancestor_terms(steps, expected):
    assert _format_ancestor_term(steps) == expected


@pytest.mark.parametrize("steps,expected", [
    (1, "child"),
    (2, "grandchild"),
    (4, "great-great-grandchild"),
])
def test_descendant_terms(steps, expected):
    assert _format_descendant_term(steps) == expected


@pytest.mark.parametrize("steps,expected", [
    (2, "aunt/uncle"),
    (3, "grandaunt/uncle"),
    (4, "great-grandaunt/uncle"),
    (5, "great-great-grandaunt/uncle"),
])
def test_aunt_uncle_terms(steps, expected):
    assert _format_aunt_uncle_term(steps) == expected


@pytest.mark.parametrize("steps,expected", [
    (2, "niece/nephew"),
    (3, "grandniece/nephew"),
    (4, "great-grandniece/nephew"),
])
def test_niece_nephew_terms(steps, expected):
    assert _format_niece_nephew_term(steps) == expected


@pytest.mark.parametrize("neutral,gender,expected", [
    ("parent", "male", "father"),
    ("parent", "female", "mother"),
    ("child", "male", "son"),
    ("sibling", "female", "sister"),
    ("aunt/uncle", "male", "uncle"),
    ("aunt/uncle", "female", "aunt"),
    ("niece/nephew", "male", "nephew"),
    ("grandparent", "female", "grandmother"),
    # great- prefixes must survive the gender substitution
    ("great-great-grandparent", "male", "great-great-grandfather"),
    ("great-grandchild", "female", "great-granddaughter"),
    ("great-grandaunt/uncle", "male", "great-granduncle"),
    ("grandniece/nephew", "female", "grandniece"),
    # in-law forms
    ("parent-in-law", "male", "father-in-law"),
    ("sibling-in-law", "female", "sister-in-law"),
    ("child-in-law", "male", "son-in-law"),
    ("great-grandparent-in-law", "female", "great-grandmother-in-law"),
])
def test_gendered_variant(neutral, gender, expected):
    assert _gendered_variant(neutral, gender) == expected


def test_gendered_variant_without_gender_returns_none():
    assert _gendered_variant("aunt/uncle", None) is None


def test_gendered_variant_unknown_term_returns_none():
    assert _gendered_variant("cousin", "male") is None


@pytest.mark.parametrize("gender,meta,expected", [
    ("Male", None, "male"),
    ("f", None, "female"),
    ("NB", None, "nonbinary"),
    (None, {"gender": "woman"}, "female"),
    (None, {"sex": "M"}, "male"),
    (None, {}, None),
    ("", None, None),
])
def test_gender_from_meta(gender, meta, expected):
    person = SimpleNamespace(gender=gender, meta=meta)
    assert _gender_from_meta(person) == expected


def test_gender_from_meta_none_person():
    assert _gender_from_meta(None) is None


def test_column_takes_priority_over_meta():
    person = SimpleNamespace(gender="female", meta={"gender": "male"})
    assert _gender_from_meta(person) == "female"
