import pytest

from strider.trapi import \
    merge_messages


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
    assert node["attributes"] == [ATTRIBUTE_A, ATTRIBUTE_B]

    assert sorted(node["category"]) == \
        ["biolink:Disease", "biolink:DiseaseOrPhenotypicFeature"]


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

    assert edge["attributes"] == [ATTRIBUTE_A, ATTRIBUTE_B]
