"""Test Strider."""
import json

from fastapi.responses import Response
import pytest
import redis.asyncio

from tests.helpers.context import (
    with_response_overlay,
    with_norm_overlay,
)
from tests.helpers.logger import setup_logger
from tests.helpers.utils import query_graph_from_string, validate_message
from tests.helpers.redisMock import redisMock

from strider.config import settings
from strider.server import lookup

# Switch prefix path before importing server
settings.normalizer_url = "http://normalizer"


setup_logger()


@pytest.mark.asyncio
@with_norm_overlay(
    settings.normalizer_url,
)
async def test_mixed_canonical(monkeypatch, mocker):
    """Test qedge with mixed canonical and non-canonical predicates."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.trapi_throttle.throttle.ThrottledServer._query",
        return_value={"message": {}},
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:treats biolink:phenotype_of -->n1
        """
    )

    # Create query
    q = {
        "message": {"query_graph": QGRAPH},
        "log_level": "ERROR",
    }

    # Run
    await lookup(q)

    query.assert_called_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["CHEBI:6801"],
                            "categories": ["biolink:ChemicalSubstance"],
                            "is_set": False,
                            "constraints": [],
                        },
                        "n1": {
                            "categories": ["biolink:Disease"],
                            "is_set": False,
                            "constraints": [],
                        },
                    },
                    "edges": {
                        "n0n1": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats", "biolink:phenotype_of"],
                            "attribute_constraints": [],
                            "qualifier_constraints": [],
                        },
                    },
                },
            },
        },
        timeout=60.0,
    )


@pytest.mark.asyncio
@with_norm_overlay(
    settings.normalizer_url,
)
async def test_symmetric_noncanonical(monkeypatch, mocker):
    """Test qedge with the symmetric, non-canonical predicate genetically_interacts_with."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.trapi_throttle.throttle.ThrottledServer._query",
        return_value={"message": {}},
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:genetically_interacts_with -->n1
        """
    )

    # Create query
    q = {
        "message": {"query_graph": QGRAPH},
        "log_level": "INFO",
    }

    # Run
    await lookup(q)

    query.assert_called_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["CHEBI:6801"],
                            "categories": ["biolink:ChemicalSubstance"],
                            "is_set": False,
                            "constraints": [],
                        },
                        "n1": {
                            "categories": ["biolink:Disease"],
                            "is_set": False,
                            "constraints": [],
                        },
                    },
                    "edges": {
                        "n0n1": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:genetically_interacts_with"],
                            "attribute_constraints": [],
                            "qualifier_constraints": [],
                        },
                    },
                },
            },
        },
        timeout=60.0,
    )


@pytest.mark.asyncio
@with_norm_overlay(
    settings.normalizer_url,
)
# Add attributes to kp1 response
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps(
            {
                "message": {
                    "query_graph": {
                        "nodes": {
                            "n0": {"ids": ["CHEBI:6801"]},
                            "n1": {"categories": ["biolink:Disease"]},
                        },
                        "edges": {
                            "n0n1": {
                                "subject": "n0",
                                "predicate": "biolink:treats",
                                "object": "n1",
                            },
                        },
                    },
                    "knowledge_graph": {
                        "nodes": {
                            "CHEBI:XXX": {},
                            "MONDO:0005148": {
                                "attributes": [
                                    {
                                        "attribute_type_id": "test_constraint",
                                        "value": "foo",
                                    },
                                ],
                            },
                        },
                        "edges": {
                            "n0n1": {
                                "subject": "CHEBI:XXX",
                                "predicate": "biolink:treats",
                                "object": "MONDO:0005148",
                            },
                        },
                    },
                    "results": [
                        {
                            "node_bindings": {
                                "n0": [
                                    {
                                        "id": "CHEBI:XXX",
                                        "qnode_id": "CHEBI:6801",
                                    }
                                ],
                                "n1": [{"id": "MONDO:0005148"}],
                            },
                            "edge_bindings": {
                                "n0n1": [{"id": "n0n1"}],
                            },
                        },
                    ],
                }
            }
        ),
    ),
)
async def test_disambiguation(monkeypatch):
    """
    Test disambiguating batch results with qnode_id.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0-- biolink:treats -->n1
        n1(( categories[] biolink:Disease ))
        """
    )

    # Create query
    q = {
        "message": {"query_graph": QGRAPH},
        "log_level": "ERROR",
    }

    # Run
    _, output = await lookup(q)
    assert len(output["message"]["results"]) == 1

    validate_message(
        {
            "knowledge_graph": """
                CHEBI:XXX biolink:treats MONDO:0005148
                """,
            "results": [
                """
                node_bindings:
                    n0 CHEBI:XXX
                    n1 MONDO:0005148
                edge_bindings:
                    n0n1 CHEBI:XXX-MONDO:0005148
                """,
            ],
        },
        output["message"],
    )


@pytest.mark.asyncio
@with_norm_overlay(
    settings.normalizer_url,
)
# Add attributes to ctd response
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps(
            {
                "message": {
                    "query_graph": {
                        "nodes": {
                            "n0": {"ids": ["CHEBI:6801"]},
                            "n1": {"categories": ["biolink:Disease"]},
                        },
                        "edges": {
                            "n0n1": {
                                "subject": "n0",
                                "predicate": "biolink:treats",
                                "object": "n1",
                            },
                        },
                    },
                    "knowledge_graph": {
                        "nodes": {
                            "CHEBI:XXX": {},
                            "MONDO:0005148": {
                                "attributes": [
                                    {
                                        "attribute_type_id": "test_constraint",
                                        "value": "foo",
                                    },
                                ],
                            },
                        },
                        "edges": {
                            "n0n1": {
                                "subject": "CHEBI:XXX",
                                "predicate": "biolink:treats",
                                "object": "MONDO:0005148",
                            },
                        },
                    },
                    "results": [
                        {
                            "node_bindings": {
                                "n0": [{"id": "CHEBI:XXX"}],
                                "n1": [{"id": "MONDO:0005148"}],
                            },
                            "edge_bindings": {
                                "n0n1": [{"id": "n0n1"}],
                            },
                        },
                    ],
                }
            }
        ),
    ),
)
async def test_trivial_unbatching(monkeypatch):
    """Test trivial unbatching with batch size one."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0-- biolink:treats -->n1
        n1(( categories[] biolink:Disease ))
        """
    )

    # Create query
    q = {
        "message": {"query_graph": QGRAPH},
        "log_level": "ERROR",
    }

    # Run
    _, output = await lookup(q)
    assert len(output["message"]["results"]) == 1

    validate_message(
        {
            "knowledge_graph": """
                CHEBI:XXX biolink:treats MONDO:0005148
                """,
            "results": [
                """
                node_bindings:
                    n0 CHEBI:XXX
                    n1 MONDO:0005148
                edge_bindings:
                    n0n1 CHEBI:XXX-MONDO:0005148
                """,
            ],
        },
        output["message"],
    )


@pytest.mark.asyncio
@with_norm_overlay(
    settings.normalizer_url,
)
async def test_protein_gene_conflation(monkeypatch, mocker):
    """Test conflation of biolink:Gene and biolink:Protein categories.
    e0 checks that Gene is added to Protein nodes."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.trapi_throttle.throttle.ThrottledServer._query",
        return_value={"message": {}},
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] MONDO:0008114 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:Protein ))
        n0-- biolink:related_to -->n1
        """
    )

    # Create query
    q = {"message": {"query_graph": QGRAPH}, "log_level": "INFO"}

    # Run query
    await lookup(q)

    query.assert_called_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["MONDO:0008114"],
                            "categories": ["biolink:Disease"],
                            "is_set": False,
                            "constraints": [],
                        },
                        "n1": {
                            "categories": ["biolink:Protein", "biolink:Gene"],
                            "is_set": False,
                            "constraints": [],
                        },
                    },
                    "edges": {
                        "n0n1": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                            "attribute_constraints": [],
                            "qualifier_constraints": [],
                        },
                    },
                },
            },
        },
        timeout=60.0,
    )


@pytest.mark.asyncio
@with_norm_overlay(
    settings.normalizer_url,
)
async def test_gene_protein_conflation(monkeypatch, mocker):
    """Test conflation of biolink:Gene and biolink:Protein categories.
    e0 checks to make sure that Protein is added to Gene nodes."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.trapi_throttle.throttle.ThrottledServer._query",
        return_value={"message": {}},
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:Gene ))
        n1(( ids[] MONDO:0008114 ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:related_to -->n1
        """
    )

    # Create query
    q = {"message": {"query_graph": QGRAPH}, "log_level": "INFO"}

    # Run query
    await lookup(q)

    query.assert_called_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "categories": ["biolink:Gene", "biolink:Protein"],
                            "is_set": False,
                            "constraints": [],
                        },
                        "n1": {
                            "ids": ["MONDO:0008114"],
                            "categories": ["biolink:Disease"],
                            "is_set": False,
                            "constraints": [],
                        },
                    },
                    "edges": {
                        "n0n1": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                            "attribute_constraints": [],
                            "qualifier_constraints": [],
                        },
                    },
                },
            },
        },
        timeout=60.0,
    )


@pytest.mark.asyncio
@with_norm_overlay(
    settings.normalizer_url,
)
async def test_node_set(monkeypatch, mocker):
    """Test that is_set is handled correctly."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.trapi_throttle.throttle.ThrottledServer._query",
        return_value={"message": {}},
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:treats -->n1
        """
    )
    QGRAPH["nodes"]["n1"]["is_set"] = True

    # Create query
    q = {
        "message": {"query_graph": QGRAPH},
        "log_level": "WARNING",
    }

    # Run
    await lookup(q)

    query.assert_called_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["CHEBI:6801"],
                            "categories": ["biolink:ChemicalSubstance"],
                            "is_set": False,
                            "constraints": [],
                        },
                        "n1": {
                            "categories": ["biolink:Disease"],
                            "is_set": True,
                            "constraints": [],
                        },
                    },
                    "edges": {
                        "n0n1": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                            "attribute_constraints": [],
                            "qualifier_constraints": [],
                        },
                    },
                },
            },
        },
        timeout=60.0,
    )
