import pytest

from strider.trapi import \
    filter_by_qgraph, canonicalize_qgraph, merge_messages


@pytest.mark.asyncio
async def test_deduplicate_results_out_of_order():
    """
    Test that we successfully deduplicate results when given
    the same results but in a different order
    """

    message = {
        "query_graph": {"nodes": {}, "edges": {}},
        "knowledge_graph": {"nodes": {}, "edges": {}},
        "results": [
            {
                "node_bindings": {
                    "a": [{"id": "CHEBI:88916"}, {"id": "MONDO:0011122"}],
                },
                "edge_bindings": {},
            },
            {
                "node_bindings": {
                    "a": [{"id": "MONDO:0011122"}, {"id": "CHEBI:88916"}],
                },
                "edge_bindings": {},
            }
        ]
    }

    output = merge_messages([message])

    assert len(output["results"]) == 1


@pytest.mark.asyncio
async def test_deduplicate_results_different():
    """
    Test that we don't deduplicate results when given
    different binding information
    """

    message = {
        "query_graph": {"nodes": {}, "edges": {}},
        "knowledge_graph": {"nodes": {}, "edges": {}},
        "results": [
            {
                "node_bindings": {
                    "b": [{"id": "CHEBI:88916"}, {"id": "MONDO:0011122"}],
                },
                "edge_bindings": {},
            },
            {
                "node_bindings": {
                    "a": [{"id": "MONDO:0011122"}, {"id": "CHEBI:88916"}],
                },
                "edge_bindings": {},
            }
        ]
    }

    output = merge_messages([message])

    assert len(output["results"]) == 2


ATTRIBUTE_A = {
    "attribute_type_id": "biolink:knowledge_source",
    "value": "https://automat.renci.org/",
}
ATTRIBUTE_B = {
    "attribute_type_id": "biolink:is_bad",
    "value": True,
}


def test_merge_knowledge_graph_nodes():
    """
    Test that we do a smart merge when given knowledge
    graph nodes with the same keys
    """

    message_a = {
        "query_graph": {"nodes": {}, "edges": {}},
        "knowledge_graph": {
            "nodes": {
                "MONDO:1": {
                    "name": "Ebola",
                    "categories": ["biolink:Disease"],
                    "attributes": [ATTRIBUTE_A]
                }
            },
            "edges": {}},
        "results": []
    }

    message_b = {
        "query_graph": {"nodes": {}, "edges": {}},
        "knowledge_graph": {
            "nodes": {
                "MONDO:1": {
                    "name": "Ebola Hemorrhagic Fever",
                    "categories": ["biolink:DiseaseOrPhenotypicFeature"],
                    "attributes": [ATTRIBUTE_B]
                }
            },
            "edges": {}},
        "results": []
    }

    output = merge_messages([message_a, message_b])

    # Validate output
    nodes = output["knowledge_graph"]["nodes"]
    assert len(nodes) == 1
    node = next(iter(nodes.values()))
    assert node["attributes"] == [ATTRIBUTE_B]

    assert sorted(node["categories"]) == ["biolink:DiseaseOrPhenotypicFeature"]


def test_merge_knowledge_graph_edges():
    """
    Test that we do a smart merge when given knowledge
    graph edges with the same subject, object, predicate
    """

    message_a = {
        "query_graph": {"nodes": {}, "edges": {}},
        "knowledge_graph": {
            "nodes": {
                "MONDO:1": {},
                "CHEBI:1": {}
            },
            "edges": {
                "n0n1": {
                    "subject": "MONDO:1",
                    "object": "CHEBI:1",
                    "predicate": "biolink:treated_by",
                    "attributes": [ATTRIBUTE_A],
                }
            }},
        "results": []
    }

    message_b = {
        "query_graph": {"nodes": {}, "edges": {}},
        "knowledge_graph": {
            "nodes": {
                "MONDO:1": {},
                "CHEBI:1": {}
            },
            "edges": {
                "n0n1": {
                    "subject": "MONDO:1",
                    "object": "CHEBI:1",
                    "predicate": "biolink:treated_by",
                    "attributes": [ATTRIBUTE_B],
                }
            }},
        "results": []
    }

    output = merge_messages([message_a, message_b])

    # Validate output
    edges = output["knowledge_graph"]["edges"]
    assert len(edges) == 1
    edge = next(iter(edges.values()))

    assert edge["attributes"] == [ATTRIBUTE_B]


def test_merge_identical_attributes():
    """
    Tests that identical attributes are merged
    """

    message_a = {
        "query_graph": {"nodes": {}, "edges": {}},
        "knowledge_graph": {
            "nodes": {
                "MONDO:1": {
                    "name": "Ebola",
                    "category": "biolink:Disease",
                    "attributes": [ATTRIBUTE_A]
                }
            },
            "edges": {}},
        "results": []
    }

    message_b = {
        "query_graph": {"nodes": {}, "edges": {}},
        "knowledge_graph": {
            "nodes": {
                "MONDO:1": {
                    "name": "Ebola Hemorrhagic Fever",
                    "category": "biolink:DiseaseOrPhenotypicFeature",
                    "attributes": [ATTRIBUTE_A]
                }
            },
            "edges": {}},
        "results": []
    }

    output = merge_messages([message_a, message_b])

    # Validate output
    nodes = output["knowledge_graph"]["nodes"]
    assert len(nodes) == 1
    node = next(iter(nodes.values()))
    assert node["attributes"] == [ATTRIBUTE_A]

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
