import pytest

from strider.trapi import filter_by_qgraph, canonicalize_qgraph

def test_filter_by_qgraph_id():
    """
    Test that the filter_by_qgraph method
    removes nodes with the wrong ID
    """

    message = {
        "knowledge_graph" : {
            "nodes" : {
                "MONDO:1" : {},
                "MONDO:2" : {}
            },
            "edges" : {}
        },
        "results" : [
            {
                "node_bindings" : {"n0" : [{"id" : "MONDO:1"}]},
                "edge_bindings" : {},
            },
            {
                "node_bindings" : {"n0" : [{"id" : "MONDO:2"}]},
                "edge_bindings" : {},
            },
        ]
    }

    qgraph = {
        "nodes" : {
            "n0" : {"ids" : ["MONDO:1"]},
        }
    }

    filter_by_qgraph(message, qgraph)

    assert len(message["results"]) == 1
    assert len(message["knowledge_graph"]["nodes"]) == 1


def test_canonicalize_qgraph():
    """Test canonicalize_qgraph()."""
    qgraph = {
        "nodes": {},
        "edges": {
            "e01": {
                "subject": "n0",
                "predicates": ["biolink:treated_by"],
                "object": "n1",
            },
        },
    }
    fixed_qgraph = canonicalize_qgraph(qgraph)[0]
    e01 = fixed_qgraph["edges"]["e01"]
    assert e01["subject"] == "n1"
    assert e01["predicates"] == ["biolink:treats"]
    assert e01["object"] == "n0"
    assert fixed_qgraph == canonicalize_qgraph(fixed_qgraph)[0]


def test_uncanonicalizable_qgraph():
    """Test qgraph with mixed canonical and non-canonical predicates."""
    qgraph = {
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
    }
    assert len(canonicalize_qgraph(qgraph)) == 2


def test_filter_by_qgraph_category():
    """
    Test that the filter_by_qgraph method
    removes nodes with the wrong category

    The only time this should happen is with a misbehaving KP
    that doesn't return results that match its own type
    """

    message = {
        "knowledge_graph" : {
            "nodes" : {
                "MONDO:1" : {
                    "categories" : ["biolink:Disease"]
                },
                "MONDO:2" : {
                    "categories" : ["biolink:ChemicalSubstance"]
                }
            },
            "edges" : {}
        },
        "results" : [
            {
                "node_bindings" : {"n0" : [{"id" : "MONDO:1"}]},
                "edge_bindings" : {},
            },
            {
                "node_bindings" : {"n0" : [{"id" : "MONDO:2"}]},
                "edge_bindings" : {},
            },
        ]
    }

    qgraph = {
        "nodes" : {
            "n0" : {"categories" : ["biolink:DiseaseOrPhenotypicFeature"]},
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
        "knowledge_graph" : {
            "nodes" : {"n0" : {}},
            "edges" : {
                "ke1" : {
                    "subject" : "n0",
                    "object" : "n0",
                    "predicate" : "biolink:ameliorates",
                },
                "ke2" : {
                    "subject" : "n0",
                    "object" : "n0",
                    "predicate" : "biolink:treats",
                }
            },
        },
        "results" : [
            {
                "node_bindings" : {},
                "edge_bindings" : {"e1" : [{"id" : "ke1"}]},
            },
            {
                "node_bindings" : {},
                "edge_bindings" : {"e1" : [{"id" : "ke2"}]},
            },
        ]
    }

    qgraph = {
        "nodes" : {
            "n0" : {},
        },
        "edges" : {
            "e1" : {
                "predicates" : ["biolink:treats"]
            }
        },
    }

    filter_by_qgraph(message, qgraph)

    assert len(message["results"]) == 1
    assert len(message["knowledge_graph"]["edges"]) == 1
