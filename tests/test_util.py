"""Test utilities."""
from strider.util import remove_null_values


def test_remove_null_values():
    """Test removal of null values from JSON-able objects."""
    assert remove_null_values([{
        "a": 1,
        "b": None,
    }]) == [{
        "a": 1,
    }]
