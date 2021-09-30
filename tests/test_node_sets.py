"""Test node set handling."""
from strider.node_sets import collapse_sets


def test_node_sets():
    """Test collapsing one edge of a two-hop query."""
    qgraph = {
        "nodes": {
            "n0": {},
            "n1": {"is_set": True}
        },
        "edges": {
            "e01": {},
        },
    }
    results = [
        {
            "node_bindings": {
                "n0": [{"id": "a0"}],
                "n1": [{"id": "b0"}],
            },
            "edge_bindings": {
                "e01": [{"id": "c0"}],
            },
        },
        {
            "node_bindings": {
                "n0": [{"id": "a1"}],
                "n1": [{"id": "b0"}],
            },
            "edge_bindings": {
                "e01": [{"id": "c1"}],
            },
        },
        {
            "node_bindings": {
                "n0": [{"id": "a0"}],
                "n1": [{"id": "b1"}],
            },
            "edge_bindings": {
                "e01": [{"id": "c2"}],
            },
        },
    ]
    message = {
        "query_graph": qgraph,
        "results": results,
    }
    collapse_sets(message)
    assert len(message["results"]) == 2
    assert len(message["results"][0]["node_bindings"]["n1"]) == 2
