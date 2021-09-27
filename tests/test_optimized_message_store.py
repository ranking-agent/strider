import copy
import json
import pytest

from tests.helpers.utils import attribute_from_string
from strider.optimized_message_store import OptimizedMessageStore, freeze_attribute

def get_base_message():
    return {
        "query_graph": {"nodes": {}, "edges": {}},
        "knowledge_graph": {"nodes": {}, "edges": {}},
        "results": []
    }

# Some sample attributes
ATTRIBUTE_A = attribute_from_string("""
    type biolink:knowledge_source value https://automat.renci.org/
        type biolink:has_p-value_evidence value 0.04
""")

ATTRIBUTE_B = attribute_from_string("""
    type biolink:publication value pubmed_central
        type biolink:has_original_source value true
""")

# Freeze these attributes to make it easy to compare to output
ATTRIBUTE_A = freeze_attribute(ATTRIBUTE_A)
ATTRIBUTE_B = freeze_attribute(ATTRIBUTE_B)

def test_result_merging():
    """ Test that duplicate results are merged correctly """

    message = {
        "knowledge_graph" : {
            "nodes" : {},
            "edges" : {
                "ke0" : {
                    "subject" : "kn0",
                    "object" : "kn1",
                    "predicate" : "biolink:ameliorates"
                }
            }
        },
        "results" : [
            {
                "node_bindings" : {"n0" : [{"id" : "kn0"}]},
                "edge_bindings" : {"e0" : [{"id" : "ke0"}]},
            },
            {
                "node_bindings" : {"n0" : [{"id" : "kn0"}]},
                "edge_bindings" : {"e0" : [{"id" : "ke0"}]},
            }
        ]
    }

    store = OptimizedMessageStore()

    store.add_message(message)
    output_message = store.get_message()

    assert len(output_message["results"]) == 1

def test_different_result_merging():
    """ Test that different results are not merged """

    message = {
        "knowledge_graph" : {
            "nodes" : {},
            "edges" : {
                "ke0" : {
                    "subject" : "kn0",
                    "object" : "kn1",
                    "predicate" : "biolink:ameliorates"
                }
            }
        },
        "results" : [
            {
                "node_bindings" : {"n0" : [{"id" : "kn0"}]},
                "edge_bindings" : {"e0" : [{"id" : "ke0"}]},
            },
            {
                "node_bindings" : {"n0" : [{"id" : "kn0"}]},
                "edge_bindings" : {"e0" : [{"id" : "ke0", "attributes" : [ATTRIBUTE_A]}]},
            }
        ]
    }

    store = OptimizedMessageStore()

    store.add_message(message)
    output_message = store.get_message()

    assert len(output_message["results"]) == 2

def test_deduplicate_results_out_of_order():
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

    store = OptimizedMessageStore()
    store.add_message(message)
    output = store.get_message()

    assert len(output["results"]) == 1


def test_deduplicate_results_different():
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

    store = OptimizedMessageStore()
    store.add_message(message)
    output = store.get_message()

    assert len(output["results"]) == 2

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


    store = OptimizedMessageStore()
    store.add_message(message_a)
    store.add_message(message_b)
    output = store.get_message()

    # Validate output
    nodes = output["knowledge_graph"]["nodes"]
    assert len(nodes) == 1
    node = next(iter(nodes.values()))

    assert node["attributes"] == {ATTRIBUTE_A, ATTRIBUTE_B}


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

    store = OptimizedMessageStore()
    store.add_message(message_a)
    store.add_message(message_b)
    output = store.get_message()

    # Validate output
    edges = output["knowledge_graph"]["edges"]
    assert len(edges) == 1
    edge = next(iter(edges.values()))

    assert edge["attributes"] == {ATTRIBUTE_A, ATTRIBUTE_B}


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

    store = OptimizedMessageStore()
    store.add_message(message_a)
    store.add_message(message_b)
    output = store.get_message()

    # Validate output
    nodes = output["knowledge_graph"]["nodes"]
    assert len(nodes) == 1
    node = next(iter(nodes.values()))

    assert node["attributes"] == {ATTRIBUTE_A}
