"""Test node set handling."""
import logging

from reasoner_pydantic import (
    Query,
    Message,
    QueryGraph,
    Results,
)
from reasoner_pydantic.utils import HashableSet
from strider.node_sets import collapse_sets

LOGGER = logging.getLogger(__name__)


def test_node_sets():
    """Test collapsing one edge of a two-hop query."""
    qgraph = {
        "nodes": {"n0": {}, "n1": {"is_set": True}},
        "edges": {
            "e01": {
                "subject": "n0",
                "object": "n1",
            },
        },
    }
    results = [
        {
            "node_bindings": {
                "n0": [{"id": "a0"}],
                "n1": [{"id": "b0"}],
            },
            "analyses": [
                {
                    "resource_id": "infores:aragorn",
                    "edge_bindings": {
                        "e01": [{"id": "c0"}],
                    },
                }
            ],
        },
        {
            "node_bindings": {
                "n0": [{"id": "a1"}],
                "n1": [{"id": "b0"}],
            },
            "analyses": [
                {
                    "resource_id": "infores:aragorn",
                    "edge_bindings": {
                        "e01": [{"id": "c1"}],
                    },
                }
            ],
        },
        {
            "node_bindings": {
                "n0": [{"id": "a0"}],
                "n1": [{"id": "b1"}],
            },
            "analyses": [
                {
                    "resource_id": "infores:aragorn",
                    "edge_bindings": {
                        "e01": [{"id": "c2"}],
                    },
                }
            ],
        },
    ]

    query = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(qgraph),
            results=Results.parse_obj(results),
        )
    )
    collapse_sets(query, LOGGER)
    assert len(query.message.results) == 2
