from __future__ import annotations

import pytest

from memory_core import RustVectorIndex


def test_selected_ids_are_applied_before_native_top_k():
    index = RustVectorIndex(2)
    index.upsert("private", [1.0, 0.0])
    index.upsert("allowed", [3.0, 4.0])
    assert index.search([1.0, 0.0], ["allowed", "missing"], 1) == [
        ("allowed", pytest.approx(0.6))
    ]
    index.upsert("allowed", [-1.0, 0.0])
    assert index.search([1.0, 0.0], ["allowed"], 1) == [("allowed", -1.0)]
    assert index.remove("allowed")
    assert not index.remove("allowed")
    assert index.search([1.0, 0.0], ["allowed"], 1) == []


def test_zero_policy_preserves_strict_default_and_supports_lexical_embeddings():
    strict = RustVectorIndex(2)
    with pytest.raises(ValueError, match="zero"):
        strict.upsert("a", [0.0, 0.0])
    permissive = RustVectorIndex(2, allow_zero=True)
    permissive.upsert("zero", [0.0, 0.0])
    permissive.upsert("active", [3.0, 4.0])
    assert permissive.search([0.0, 0.0], ["zero", "active"], 2) == [
        ("active", 0.0),
        ("zero", 0.0),
    ]
    assert permissive.search([1.0, 0.0], ["zero"], 1) == [("zero", 0.0)]
    for vector in ([float("nan"), 0.0], [float("inf"), 0.0], [1.0]):
        with pytest.raises(ValueError):
            permissive.upsert("active", vector)
    assert permissive.search([1.0, 0.0], ["active"], 1) == [
        ("active", pytest.approx(0.6))
    ]


def test_ties_have_canonical_string_identifier_order():
    index = RustVectorIndex(2)
    for uid in ("2", "10", "1"):
        index.upsert(uid, [1.0, 0.0])
    assert index.search([1.0, 0.0], ["2", "10", "1"], 2) == [("1", 1.0), ("10", 1.0)]
