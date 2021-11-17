"""Test utilities."""
from strider.util import remove_null_values
from strider.trapi import filter_ancestor_types


def test_remove_null_values():
    """Test removal of null values from JSON-able objects."""
    assert remove_null_values([{"a": 1, "b": None,}]) == [
        {
            "a": 1,
        }
    ]


def test_filter_ancestor_types():
    """Test filter_ancestor_types()."""
    filtered = filter_ancestor_types(
        ["biolink:Disease", "biolink:DiseaseOrPhenotypicFeature"]
    )
    assert filtered == ["biolink:Disease"]
    filtered = filter_ancestor_types(["biolink:NamedThing"])
    assert filtered == ["biolink:NamedThing"]
    filtered = filter_ancestor_types(["biolink:Disease", "biolink:Disease"])
    assert filtered == ["biolink:Disease", "biolink:Disease"]
