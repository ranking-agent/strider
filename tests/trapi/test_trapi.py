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
