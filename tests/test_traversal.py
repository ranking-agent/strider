"""Test traversal."""

import pytest

from strider.traversal import NoAnswersError, get_traversals


def test_one_hop():
    """Test one-hop qgraph."""
    qgraph = {
        "nodes": {
            "ebola": {"ids": ["MONDO:0005737"]},
            "gene": {"categories": ["biolink:Gene"]},
        },
        "edges": {
            "affects": {
                "subject": "gene",
                "predicates": ["affects"],
                "object": "ebola",
            },
        },
    }
    assert get_traversals(qgraph) == [["affects"]]


def test_two_ended_hop():
    """Test two-hop qgraph with both ends pinned."""
    qgraph = {
        "nodes": {
            "ebola": {"ids": ["MONDO:0005737"]},
            "gene": {"categories": ["biolink:Gene"]},
            "imatinib": {"ids": ["CHEBI:45783"]},
        },
        "edges": {
            "affects": {
                "subject": "gene",
                "predicates": ["biolink:affects"],
                "object": "ebola",
            },
            "downregulates": {
                "subject": "imatinib",
                "predicates": ["biolink:negatively_regulates"],
                "object": "gene",
            },
        },
    }
    traversals = get_traversals(qgraph)
    assert ["affects", "downregulates"] in traversals
    assert ["downregulates", "affects"] in traversals


def test_cycle():
    """Test cyclic qgraph."""
    qgraph = {
        "nodes": {
            "ebola": {"ids": ["MONDO:0005737"]},
            "gene": {"categories": ["biolink:Gene"]},
            "drug": {"categories": ["biolink:Drug"]},
        },
        "edges": {
            "affects": {
                "subject": "gene",
                "predicates": ["biolink:affects"],
                "object": "ebola",
            },
            "downregulates": {
                "subject": "drug",
                "predicates": ["biolink:negatively_regulates"],
                "object": "gene",
            },
            "treats": {
                "subject": "drug",
                "predicates": ["biolink:treats"],
                "object": "ebola",
            },
        },
    }
    traversals = get_traversals(qgraph)
    assert len(traversals) == 4


def test_untraversable():
    """Test untraversable qgraph."""
    qgraph = {
        "nodes": {
            "ebola": {"ids": ["MONDO:0005737"]},
            "gene": {"categories": ["biolink:Gene"]},
            "drug": {"categories": ["biolink:Drug"]},
        },
        "edges": {
            "affects": {
                "subject": "gene",
                "predicates": ["biolink:affects"],
                "object": "ebola",
            },
        },
    }
    with pytest.raises(NoAnswersError):
        traversals = get_traversals(qgraph)
