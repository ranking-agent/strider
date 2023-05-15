"""Test weird kp responses."""
import httpx
import json
import pytest
import redis.asyncio

from fastapi.responses import Response

from tests.helpers.context import (
    with_norm_overlay,
    with_response_overlay,
)
from tests.helpers.redisMock import redisMock
from tests.helpers.logger import setup_logger
from tests.helpers.utils import query_graph_from_string, validate_message
import tests.helpers.mock_responses as mock_responses

from strider.server import APP
from strider.config import settings

# Switch prefix path before importing server
settings.normalizer_url = "http://normalizer"


@pytest.fixture()
async def client():
    """Yield httpx client."""
    async with httpx.AsyncClient(app=APP, base_url="http://test") as client_:
        yield client_


setup_logger()


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps({"message": {}}),
    ),
)
async def test_kp_response_empty_message(client, monkeypatch):
    """
    Test when a KP returns null query graph.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:Drug ))
        n0(( ids[] CHEBI:6801 ))
        n0-- biolink:treats -->n1
        n1(( category biolink:Disease ))
        """
    )

    # Create query
    q = {
        "message": {"query_graph": QGRAPH},
        # "log_level": "WARNING"
    }

    # Run
    response = await client.post("/query", json=q)
    output = response.json()
    assert "knowledge_graph" in output["message"]
    assert "results" in output["message"]
    assert "query_graph" in output["message"]


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps({"message": {}}),
    ),
)
async def test_kp_response_empty_message_pinned_two_hop(client, monkeypatch):
    """
    Test when a KP returns null query graph.
    """
    # mock the return of the kp registry from redis
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] MONDO:1 ))
        n0-- biolink:related_to -->n1
        n1(( category biolink:Gene ))
        n1-- biolink:related_to -->n2
        n2(( ids[] MONDO:2 ))

        """
    )

    # Create query
    q = {
        "message": {"query_graph": QGRAPH},
        # "log_level": "INFO",
    }

    # Run
    response = await client.post("/query", json=q)
    output = response.json()
    assert "knowledge_graph" in output["message"]
    assert "results" in output["message"]
    assert "query_graph" in output["message"]


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
# Override one KP with an invalid response
@with_response_overlay(
    "http://kp0/query",
    Response(
        status_code=500,
        content="Internal server error",
    ),
)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps(mock_responses.kp_response),
    ),
)
async def test_kp_500(client, monkeypatch):
    """
    Test that when a KP returns a 500 error we add
    a message to the log but continue running
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:treats -->n1
        """
    )

    # Create query
    q = {
        "message": {"query_graph": QGRAPH},
        "log_level": "WARNING",
    }

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    # Check that we stored the error
    assert "Response Error contacting kp0" in output["logs"][0]["message"]
    assert "Internal server error" in output["logs"][0]["response"]["data"]
    # Ensure we have results from the other KPs
    assert len(output["message"]["knowledge_graph"]["nodes"]) > 0
    assert len(output["message"]["knowledge_graph"]["edges"]) > 0
    assert len(output["message"]["results"]) > 0


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps({"message": None}),
    ),
)
async def test_kp_not_trapi(client, monkeypatch):
    """
    Test that when a KP is unavailable we add a message to
    the log but continue running
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:Disease ))
        n0(( ids[] MONDO:0005737 ))
        n0-- biolink:treated_by -->n1
        n1(( category biolink:Drug ))
        """
    )

    # Create query
    q = {
        "message": {"query_graph": QGRAPH},
        "log_level": "WARNING",
    }

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    # Check that we stored the error
    assert (
        "Received non-TRAPI compliant response from kp1" in output["logs"][0]["message"]
    )


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps({"message": {"query_graph": {"nodes": {}, "edges": {}}}}),
    ),
)
@with_response_overlay(
    "http://kp0/query",
    Response(
        status_code=200,
        content=json.dumps(mock_responses.kp_response),
    ),
)
async def test_kp_no_kg(client, monkeypatch):
    """
    Test when a KP returns a TRAPI-valid qgraph but no kgraph.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:ChemicalSubstance ))
        n0(( ids[] CHEBI:0001 ))
        n0-- biolink:treats -->n1
        n1(( category biolink:Disease ))
        """
    )

    # Create query
    q = {"message": {"query_graph": QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    assert response.status_code < 300
    output = response.json()
    assert output["message"]["results"]


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_response_overlay(
    "http://kp0/query",
    Response(
        status_code=200,
        content=json.dumps(
            {
                "message": {
                    "query_graph": None,
                    "knowledge_graph": None,
                    "results": None,
                }
            }
        ),
    ),
)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps(mock_responses.kp_response),
    ),
)
async def test_kp_response_no_qg(client, monkeypatch):
    """
    Test when a KP returns null query graph.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:ChemicalSubstance ))
        n0(( ids[] CHEBI:6801 ))
        n0-- biolink:treats -->n1
        n1(( category biolink:Disease ))
        """
    )

    # Create query
    q = {"message": {"query_graph": QGRAPH}}

    # Run
    await client.post("/query", json=q)


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
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
                            "CHEBI:6801": {},
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
                                "subject": "CHEBI:6801",
                                "predicate": "biolink:treats",
                                "object": "MONDO:0005148",
                            },
                        },
                    },
                    "results": [
                        {
                            "node_bindings": {
                                "n0": [{"id": "CHEBI:6801"}],
                                "n1": [{"id": "MONDO:0005148"}],
                            },
                            "analyses":[
                                {
                                    "resource_id": "infores:ara0",
                                    "edge_bindings": {
                                        "n0n1": [{"id": "n0n1"}],
                                    },
                                }
                            ]
                        },
                    ],
                }
            }
        ),
    ),
)
async def test_constraint_error(client, monkeypatch):
    """
    Test that we properly handle attributes
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n0-- biolink:treats -->n1
        n1(( categories[] biolink:Disease ))
        """
    )

    QGRAPH["nodes"]["n1"]["constraints"] = [
        {
            "name": "test_constraint",
            "id": "test_constraint",
            "not": True,
            "operator": "==",
            "value": "bar",
        }
    ]

    # Create query
    q = {"message": {"query_graph": QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    validate_message(
        {
            "knowledge_graph": """
                CHEBI:6801 biolink:treats MONDO:0005148
                """,
            "results": [
                """
                node_bindings:
                    n0 CHEBI:6801
                    n1 MONDO:0005148
                analyses: [
                    edge_bindings:
                        n0n1 CHEBI:6801-MONDO:0005148
                ]
                """,
            ],
        },
        output["message"],
    )
