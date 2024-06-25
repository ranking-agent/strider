"""Test weird kp responses."""

import httpx
import json
import pytest
from pytest_httpx import HTTPXMock
import redis.asyncio

from fastapi.responses import Response
from reasoner_pydantic import Query

from tests.helpers.redisMock import redisMock
from tests.helpers.logger import setup_logger
from tests.helpers.utils import query_graph_from_string, validate_message
import tests.helpers.mock_responses as mock_responses

from strider.server import sync_query
from strider.config import settings

# Switch prefix path before importing server
settings.normalizer_url = "http://normalizer"


setup_logger()


@pytest.mark.asyncio
async def test_kp_response_empty_message(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test when a KP returns null query graph.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(url="http://kp1/query", json={"message": {}})
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:Drug ))
        n0(( ids[] CHEBI:6801 ))
        n0-- biolink:treats -->n1
        n1(( category biolink:Disease ))
        """
    )

    # Create query
    q = Query.parse_obj({
        "message": {"query_graph": QGRAPH},
        # "log_level": "WARNING"
    })

    # Run
    response = await sync_query(q)
    output = json.loads(response.body)
    # output = response.content
    assert "knowledge_graph" in output["message"]
    assert "results" in output["message"]
    assert "query_graph" in output["message"]


@pytest.mark.asyncio
async def test_kp_response_empty_message_pinned_two_hop(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test when a KP returns null query graph.
    """
    # mock the return of the kp registry from redis
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(url="http://kp1/query", json={"message": {}})
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
    q = Query.parse_obj({
        "message": {"query_graph": QGRAPH},
        # "log_level": "INFO",
    })

    # Run
    response = await sync_query(q)
    output = json.loads(response.body)
    assert "knowledge_graph" in output["message"]
    assert "results" in output["message"]
    assert "query_graph" in output["message"]


@pytest.mark.asyncio
async def test_kp_500(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test that when a KP returns a 500 error we add
    a message to the log but continue running
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(url="http://kp0/query", status_code=500, text="Internal Server Error")
    httpx_mock.add_response(url="http://kp1/query", json=mock_responses.kp_response)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:treats -->n1
        """
    )

    # Create query
    q = Query.parse_obj({
        "message": {"query_graph": QGRAPH},
        "log_level": "WARNING",
    })

    # Run
    response = await sync_query(q)
    output = json.loads(response.body)

    # Ensure we have results from the other KPs
    assert len(output["message"]["knowledge_graph"]["nodes"]) > 0
    assert len(output["message"]["knowledge_graph"]["edges"]) > 0
    assert len(output["message"]["results"]) > 0


@pytest.mark.asyncio
async def test_kp_not_trapi(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test that when a KP is unavailable we add a message to
    the log but continue running
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(url="http://kp1/query", json={"message": None})
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:Disease ))
        n0(( ids[] MONDO:0005737 ))
        n0-- biolink:treated_by -->n1
        n1(( category biolink:Drug ))
        """
    )

    # Create query
    q = Query.parse_obj({
        "message": {"query_graph": QGRAPH},
        "log_level": "WARNING",
    })

    # Run
    response = await sync_query(q)
    output = json.loads(response.body)

    # Check that we stored the error
    assert (
        "Received non-TRAPI compliant response from infores:kp1"
        in output["logs"][0]["message"]
    )


@pytest.mark.asyncio
async def test_kp_no_kg(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test when a KP returns a TRAPI-valid qgraph but no kgraph.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(url="http://kp0/query", json=mock_responses.kp_response)
    httpx_mock.add_response(url="http://kp1/query", json={"message": {"query_graph": {"nodes": {}, "edges": {}}}})
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:ChemicalSubstance ))
        n0(( ids[] CHEBI:0001 ))
        n0-- biolink:treats -->n1
        n1(( category biolink:Disease ))
        """
    )

    # Create query
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}})

    # Run
    response = await sync_query(q)
    assert response.status_code < 300
    output = json.loads(response.body)
    assert output["message"]["results"]


@pytest.mark.asyncio
async def test_kp_response_no_qg(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test when a KP returns null query graph.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(url="http://kp0/query", json={
        "message": {
            "query_graph": None,
            "knowledge_graph": None,
            "results": None,
        }
    })
    httpx_mock.add_response(url="http://kp1/query", json=mock_responses.kp_response)
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:ChemicalSubstance ))
        n0(( ids[] CHEBI:6801 ))
        n0-- biolink:treats -->n1
        n1(( category biolink:Disease ))
        """
    )

    # Create query
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}})

    # Run
    await sync_query(q)


constraint_error_response = {
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
                    "sources": [
                        {
                            "resource_id": "infores:kp1",
                            "resource_role": "primary_knowledge_source",
                        }
                    ],
                },
            },
        },
        "results": [
            {
                "node_bindings": {
                    "n0": [{"id": "CHEBI:6801"}],
                    "n1": [{"id": "MONDO:0005148"}],
                },
                "analyses": [
                    {
                        "resource_id": "infores:kp1",
                        "edge_bindings": {
                            "n0n1": [{"id": "n0n1"}],
                        },
                    }
                ],
            },
        ],
    }
}
@pytest.mark.asyncio
async def test_constraint_error(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test that we properly handle attributes
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(url="http://kp1/query", json=constraint_error_response)
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
            "value": "foo",
        }
    ]

    # Create query
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}})

    # Run
    response = await sync_query(q)
    output = json.loads(response.body)

    assert output["message"]["results"] == []
