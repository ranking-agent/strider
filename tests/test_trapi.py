import pytest
from reasoner_pydantic.qgraph import QueryGraph

from strider.trapi import (
    attribute_hash,
    filter_by_qgraph,
    get_canonical_qgraphs,
)


ATTRIBUTE_A = {
    "attribute_type_id": "biolink:knowledge_source",
    "value": "https://automat.renci.org/",
}
ATTRIBUTE_B = {
    "attribute_type_id": "biolink:is_bad",
    "value": True,
}


def test_attribute_equality():
    """Test the attribute_hash function"""

    assert attribute_hash(ATTRIBUTE_A) != attribute_hash(ATTRIBUTE_B)

    # Test with sub-attributes
    ATTRIBUTE_WITH_SUBATTRIBUTES_A = {
        "attribute_type_id": "biolink:i_am_out_of_attribute_type_ideas",
        "value": 4,
        "attributes": [ATTRIBUTE_A],
    }
    ATTRIBUTE_WITH_SUBATTRIBUTES_B = {
        "attribute_type_id": "biolink:i_am_out_of_attribute_type_ideas",
        "value": 4,
        "attributes": [ATTRIBUTE_A, ATTRIBUTE_B],
    }

    assert attribute_hash(ATTRIBUTE_WITH_SUBATTRIBUTES_A) != attribute_hash(
        ATTRIBUTE_WITH_SUBATTRIBUTES_B
    )


def test_filter_by_qgraph_id():
    """
    Test that the filter_by_qgraph method
    removes nodes with the wrong ID
    """

    message = {
        "knowledge_graph": {"nodes": {"MONDO:1": {}, "MONDO:2": {}}, "edges": {}},
        "results": [
            {
                "node_bindings": {"n0": [{"id": "MONDO:1"}]},
                "edge_bindings": {},
            },
            {
                "node_bindings": {"n0": [{"id": "MONDO:2"}]},
                "edge_bindings": {},
            },
        ],
    }

    qgraph = {
        "nodes": {
            "n0": {"ids": ["MONDO:1"]},
        }
    }

    filter_by_qgraph(message, qgraph)

    assert len(message["results"]) == 1
    assert len(message["knowledge_graph"]["nodes"]) == 1


def test_canonicalize_qgraph():
    """Test canonicalize_qgraph()."""
    qgraph = QueryGraph.parse_obj({
        "nodes": {},
        "edges": {
            "e01": {
                "subject": "n0",
                "predicates": ["biolink:treated_by"],
                "object": "n1",
            },
        },
    })

    fixed_qgraph_model = get_canonical_qgraphs(qgraph)[0]
    fixed_qgraph = fixed_qgraph_model.dict()
    e01 = fixed_qgraph["edges"]["e01"]
    assert e01["subject"] == "n1"
    assert e01["predicates"] == ["biolink:treats"]
    assert e01["object"] == "n0"
    assert fixed_qgraph_model == get_canonical_qgraphs(fixed_qgraph_model)[0]


def test_uncanonicalizable_qgraph():
    """Test qgraph with mixed canonical and non-canonical predicates."""
    qgraph = QueryGraph.parse_obj({
        "nodes": {},
        "edges": {
            "e01": {
                "subject": "n0",
                "object": "n1",
                "predicates": [
                    "biolink:treats",
                    "biolink:caused_by",
                ],
            },
        },
    })
    assert len(get_canonical_qgraphs(qgraph)) == 2


def test_filter_by_qgraph_category():
    """
    Test that the filter_by_qgraph method
    removes nodes with the wrong category

    The only time this should happen is with a misbehaving KP
    that doesn't return results that match its own type
    """

    message = {
        "knowledge_graph": {
            "nodes": {
                "MONDO:1": {"categories": ["biolink:Disease"]},
                "MONDO:2": {"categories": ["biolink:ChemicalSubstance"]},
            },
            "edges": {},
        },
        "results": [
            {
                "node_bindings": {"n0": [{"id": "MONDO:1"}]},
                "edge_bindings": {},
            },
            {
                "node_bindings": {"n0": [{"id": "MONDO:2"}]},
                "edge_bindings": {},
            },
        ],
    }

    qgraph = {
        "nodes": {
            "n0": {"categories": ["biolink:DiseaseOrPhenotypicFeature"]},
        }
    }

    filter_by_qgraph(message, qgraph)

    assert len(message["results"]) == 1
    assert len(message["knowledge_graph"]["nodes"]) == 1


def test_filter_by_qgraph_predicate():
    """
    Test that the filter_by_qgraph method
    removes edges with the wrong predicate

    The only time this should happen is with a misbehaving KP
    that doesn't return results that match its own type
    """

    message = {
        "knowledge_graph": {
            "nodes": {"n0": {}},
            "edges": {
                "ke1": {
                    "subject": "n0",
                    "object": "n0",
                    "predicate": "biolink:ameliorates",
                },
                "ke2": {
                    "subject": "n0",
                    "object": "n0",
                    "predicate": "biolink:treats",
                },
            },
        },
        "results": [
            {
                "node_bindings": {},
                "edge_bindings": {"e1": [{"id": "ke1"}]},
            },
            {
                "node_bindings": {},
                "edge_bindings": {"e1": [{"id": "ke2"}]},
            },
        ],
    }

    qgraph = {
        "nodes": {
            "n0": {},
        },
        "edges": {"e1": {"predicates": ["biolink:treats"]}},
    }

    filter_by_qgraph(message, qgraph)

    assert len(message["results"]) == 1
    assert len(message["knowledge_graph"]["edges"]) == 1
