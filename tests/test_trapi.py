import pytest
from reasoner_pydantic.qgraph import QueryGraph

from strider.trapi import (
    get_canonical_qgraphs,
)


def test_canonicalize_qgraph():
    """Test canonicalize_qgraph()."""
    qgraph = QueryGraph.parse_obj(
        {
            "nodes": {},
            "edges": {
                "e01": {
                    "subject": "n0",
                    "predicates": ["biolink:treated_by"],
                    "object": "n1",
                },
            },
        }
    )

    fixed_qgraph_model = get_canonical_qgraphs(qgraph)[0]
    fixed_qgraph = fixed_qgraph_model.dict()
    e01 = fixed_qgraph["edges"]["e01"]
    assert e01["subject"] == "n1"
    assert e01["predicates"] == ["biolink:treats"]
    assert e01["object"] == "n0"
    assert fixed_qgraph_model == get_canonical_qgraphs(fixed_qgraph_model)[0]


def test_uncanonicalizable_qgraph():
    """Test qgraph with mixed canonical and non-canonical predicates."""
    qgraph = QueryGraph.parse_obj(
        {
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
    )
    qgraphs = get_canonical_qgraphs(qgraph)
    assert len(qgraphs) == 2
