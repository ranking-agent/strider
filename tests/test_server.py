"""Test Strider."""
import asyncio
import json
from pathlib import Path
import redis.asyncio

from fastapi.responses import Response
import httpx
import pytest
from reasoner_pydantic import Query, Message, QueryGraph

from tests.helpers.context import (
    with_translator_overlay,
    with_norm_overlay,
    with_response_overlay,
    callback_overlay,
)
from tests.helpers.logger import setup_logger
from tests.helpers.utils import query_graph_from_string
from tests.helpers.redisMock import redisMock
import tests.helpers.mock_responses as mock_responses

import strider
from strider.config import settings
from strider.server import APP
from strider.fetcher import Binder

# Switch prefix path before importing server
settings.normalizer_url = "http://normalizer"


@pytest.fixture()
async def client():
    """Yield httpx client."""
    async with httpx.AsyncClient(app=APP, base_url="http://test") as client_:
        yield client_


logger = setup_logger()

DEFAULT_PREFIXES = {
    "biolink:Disease": ["MONDO", "DOID"],
    "biolink:ChemicalSubstance": ["CHEBI", "MESH"],
    "biolink:PhenotypicFeature": ["HP"],
}
MYCHEM_PREFIXES = {
    **DEFAULT_PREFIXES,
    "biolink:ChemicalSubstance": ["MESH"],
    "biolink:Disease": ["DOID"],
}
CTD_PREFIXES = {
    **DEFAULT_PREFIXES,
    "biolink:Disease": ["DOID"],
}


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_response_overlay(
    "http://kp0/query",
    Response(
        status_code=200,
        content=json.dumps(mock_responses.duplicate_result_response),
    ),
)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps(mock_responses.duplicate_result_response_2),
    ),
)
async def test_duplicate_results(client, monkeypatch):
    """
    Test that we filter out duplicate results if we
    get the same nodes back with slightly different categories.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:DiseaseOrPhenotypicFeature ))
        n0-- biolink:treats -->n1
        """
    )

    # Create query
    q = {"message": {"query_graph": QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    assert len(output["message"]["results"]) == 1


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_response_overlay(
    "http://kp0/query",
    Response(
        status_code=200,
        content=json.dumps(mock_responses.duplicate_result_response),
    ),
)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps(
            mock_responses.duplicate_result_response_different_predicate
        ),
    ),
)
async def test_merge_results_different_predicates(client, monkeypatch):
    """
    Test that if we get results from KPs with different predicates
    then the results are not merged together
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:related_to -->n1
        """
    )

    # Create query
    q = {"message": {"query_graph": QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    assert response.headers.get("content-type") == "application/json"
    output = response.json()

    # Check message structure
    assert len(output["message"]["knowledge_graph"]["edges"]) == 2
    assert len(output["message"]["results"]) == 1
    result = output["message"]["results"][0]
    assert len(result["analyses"]) == 2


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
async def test_solve_missing_predicate(client, monkeypatch, mocker):
    """Test solving a query graph, in which the predicate is missing."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.trapi_throttle.throttle.ThrottledServer._query",
        return_value={"message": {}},
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] HP:001 ))
        n0(( categories[] biolink:Gene ))
        n1(( categories[] biolink:Gene ))
        n0-- biolink:treats -->n1
        """
    )

    del QGRAPH["edges"]["n0n1"]["predicates"]

    # Create query
    q = {"message": {"query_graph": QGRAPH}}

    # Run
    await client.post("/query", json=q)

    query.assert_called_once_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["HP:001"],
                            "categories": ["biolink:Gene"],
                            "is_set": False,
                            "constraints": [],
                        },
                        "n1": {
                            "categories": ["biolink:Gene", "biolink:Protein"],
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
                        }
                    },
                }
            }
        },
        timeout=60.0,
    )


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
async def test_solve_missing_category(client, monkeypatch, mocker):
    """Test solving the ex1 query graph, in which one of the categories is missing."""
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

    del QGRAPH["nodes"]["n0"]["categories"]

    # Create query
    q = {"message": {"query_graph": QGRAPH}}

    # Run
    await client.post("/query", json=q)

    query.assert_called_once_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["CHEBI:6801"],
                            "categories": ["biolink:NamedThing"],
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
                            "predicates": ["biolink:treats"],
                            "attribute_constraints": [],
                            "qualifier_constraints": [],
                        }
                    },
                }
            }
        },
        timeout=60.0,
    )


@pytest.mark.asyncio
@with_norm_overlay(
    settings.normalizer_url,
    normalizer_data="""
        CHEBI:6801 categories biolink:Vitamin
        """,
)
async def test_normalizer_different_category(client, monkeypatch, mocker):
    """
    Test solving a query graph where the category provided doesn't match
    the one in the node normalizer.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.trapi_throttle.throttle.ThrottledServer._query",
        return_value={"message": {}},
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:ChemicalSubstance ))
        n0(( ids[] CHEBI:6801 ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:treats -->n1
        """
    )

    # Create query
    q = {"message": {"query_graph": QGRAPH}}

    # Run
    await client.post("/query", json=q)

    query.assert_called_once_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["CHEBI:6801"],
                            "categories": ["biolink:Vitamin"],
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
                            "predicates": ["biolink:treats"],
                            "attribute_constraints": [],
                            "qualifier_constraints": [],
                        }
                    },
                }
            }
        },
        timeout=60.0,
    )


@pytest.mark.asyncio
@with_norm_overlay(
    settings.normalizer_url,
    normalizer_data="""
        MONDO:0008114 categories biolink:Disease
        """,
)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps(mock_responses.kp_response),
    ),
)
async def test_solve_loop(client, caplog, monkeypatch):
    """
    Test that we correctly solve a query with a loop
    """
    # TODO replace has_phenotype with the correct predicate phenotype_of
    # when BMT is updated to recognize it as a valid predicate
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] MONDO:0008114 ))
        n1(( categories[] biolink:ChemicalSubstance ))
        n2(( categories[] biolink:PhenotypicFeature ))
        n0-- biolink:related_to -->n1
        n1-- biolink:related_to -->n2
        n2-- biolink:related_to -->n0
        """
    )

    # Create query
    q = {"message": {"query_graph": QGRAPH}}

    # Run
    await client.post("/query", json=q)


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps(mock_responses.kp_response),
    ),
)
async def test_log_level_param(client, monkeypatch):
    """Test that changing the log level changes the output"""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:PhenotypicFeature ))
        n0-- biolink:treats -->n1
        """
    )

    # Create query
    q = {"message": {"query_graph": QGRAPH}}

    # Check there are no debug logs
    response = await client.post("/query", json=q)
    output = response.json()
    assert not any(l["level"] == "DEBUG" for l in output["logs"])

    q["log_level"] = "DEBUG"

    # Check there are now debug logs
    response = await client.post("/query", json=q)
    output = response.json()
    assert any(l["level"] == "DEBUG" for l in output["logs"])


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
async def test_mutability_bug(client, monkeypatch):
    """
    Test that qgraph is not mutated between KP calls.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = {
        "nodes": {
            "n0": {
                "ids": None,
                "categories": ["biolink:ChemicalSubstance"],
                "is_set": False,
                "constraints": [],
            },
            "n1": {
                "ids": ["MONDO:0005148"],
                "categories": ["biolink:Disease"],
                "is_set": False,
                "constraints": [],
            },
        },
        "edges": {
            "n0n1": {
                "subject": "n0",
                "object": "n1",
                "knowledge_type": None,
                "predicates": ["biolink:treats"],
                "attribute_constraints": [],
                "qualifier_constraints": [],
            },
        },
    }

    # Create query
    q = {"message": {"query_graph": QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()
    assert output["message"]["query_graph"] == QGRAPH


@pytest.mark.asyncio
@with_norm_overlay(
    settings.normalizer_url,
    normalizer_data="""
        CHEBI:6801 categories biolink:ChemicalSubstance
        MONDO:0005148 categories biolink:Disease
        """,
)
async def test_solve_not_real_predicate(client, monkeypatch, mocker):
    """
    Test that we can solve a query graph with
    predicates that we don't recognize
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.trapi_throttle.throttle.ThrottledServer._query",
        return_value={"message": {}},
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:not_a_real_predicate -->n1
        """
    )

    # Create query
    q = {"message": {"query_graph": QGRAPH}, "log_level": "INFO"}

    # Run
    await client.post("/query", json=q)

    query.assert_called_once()


@pytest.mark.asyncio
async def test_exception_response(client, monkeypatch):
    """
    Test that an exception's response is a 500 error
    includes the correct CORS headers,
    and is a valid TRAPI message with a log included
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qgraph = {"nodes": {}, "edges": {}}

    # Temporarily break the fetcher module to induce a 500 error
    _Message = strider.fetcher.Message
    strider.fetcher.Message = None

    response = await client.post(
        "/query",
        json={
            "message": {"query_graph": qgraph},
            "log_level": "DEBUG",
        },
        headers={"origin": "http://localhost:80"},
    )

    # Put fetcher back together
    strider.fetcher.Message = _Message

    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert response.json()["status_communication"]


@pytest.mark.asyncio
async def test_normalizer_unavailable(client, monkeypatch):
    """
    Test that we log a message properly if the Node Normalizer is not available.
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
        "log_level": "WARNING",
    }

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    output_log_messages = [log_entry["message"] for log_entry in output["logs"]]

    # Check that the correct error messages are in the log
    assert "Request Error contacting Node Normalizer" in output_log_messages


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps(mock_responses.kp_response),
    ),
)
async def test_workflow(client, monkeypatch):
    """
    Test query workflow handling.
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
        "workflow": [
            {
                "id": "lookup",
            },
        ],
    }

    # Run
    response = await client.post("/query", json=q)
    output = response.json()
    assert output["message"]["results"]


@pytest.mark.asyncio
@with_norm_overlay(
    settings.normalizer_url,
    normalizer_data="""
        UMLS:C0032961 categories biolink:PhenotypicFeature
        UMLS:C0032961 synonyms NCIT:C25742 NCIT:C92933
        NCIT:C25742 categories biolink:PhenotypicFeature
        NCIT:C25742 synonyms NCIT:C25742 NCIT:C92933
        NCIT:C92933 categories biolink:PhenotypicFeature
        NCIT:C92933 synonyms NCIT:C25742 NCIT:C92933
        CHEBI:2904 categories biolink:ChemicalSubstance
        CHEBI:30146 categories biolink:ChemicalSubstance
        """,
)
async def test_multiple_identifiers(client, monkeypatch):
    """
    Test that we correctly handle the case where we have multiple identifiers
    for the preferred prefix
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:PhenotypicFeature ))
        n1(( ids[] UMLS:C0032961 ))
        n0-- biolink:contraindicated_for -->n1
        """
    )

    # Create query
    q = {"message": {"query_graph": QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()


@pytest.mark.asyncio
@with_norm_overlay(
    settings.normalizer_url,
    normalizer_data="""
        MONDO:0005148 categories biolink:Vitamin
        """,
)
@with_response_overlay(
    "http://kp3/query",
    Response(
        status_code=200,
        content=json.dumps(mock_responses.response_with_attributes),
    ),
)
async def test_provenance(client, monkeypatch):
    """
    Tests that provenance is properly reported by strider.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n1(( categories[] biolink:NamedThing ))
        n0-- biolink:related_to -->n1
        """
    )
    q = {"message": {"query_graph": QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()
    sources = list(output["message"]["knowledge_graph"]["edges"].values())[0]["sources"]
    assert len(sources) == 2
    resource_ids = [i["resource_id"] for i in sources]
    assert "infores:aragorn" in resource_ids
    assert "infores:kp3" in resource_ids
    for source in sources:
        if source["resource_id"] == "infores:aragorn":
            assert source["resource_role"] == "aggregator_knowledge_source"
            assert source["upstream_resource_ids"] == ["infores:kp3"]
        if source["resource_id"] == "infores:kp3":
            assert source["resource_role"] == "primary_knowledge_source"


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps(mock_responses.kp_response),
    ),
)
async def test_async_query(client, monkeypatch):
    """Test asyncquery endpoint using the ex1 query graph"""
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
    q = {"callback": "http://test/", "message": {"query_graph": QGRAPH}}

    # Run
    queue = asyncio.Queue()
    async with callback_overlay("http://test/", queue):
        response = await client.post("/asyncquery", json=q)
        output = response.json()
        assert response.status_code == 200
        assert not output
        output = await queue.get()
        assert output


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
async def test_different_callbacks_multiquery(client):
    """Test multiquery endpoint."""
    QGRAPH1 = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:ChemicalSubstance ))
        n1-- biolink:treats -->n0
        """
    )
    QGRAPH2 = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:PhenotypicFeature ))
        n0-- biolink:has_phenotype -->n1
        """
    )

    q_error = {
        "query1": {"callback": "http://test1/", "message": {"query_graph": QGRAPH1}},
        "query2": {"callback": "http://test2/", "message": {"query_graph": QGRAPH2}},
    }
    response = await client.post("/multiquery", json=q_error)
    assert response.status_code == 400


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps(mock_responses.kp_response),
    ),
)
@with_response_overlay(
    "http://kp0/query",
    Response(
        status_code=200,
        content=json.dumps(mock_responses.kp_response),
    ),
)
async def test_multiquery(client, monkeypatch):
    """Test multiquery endpoint."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH1 = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:Vitamin ))
        n1-- biolink:treats -->n0
        """
    )
    QGRAPH2 = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:PhenotypicFeature ))
        n0-- biolink:has_phenotype -->n1
        """
    )

    q = {
        "query1": {
            "callback": "http://test_callback/",
            "message": {"query_graph": QGRAPH1},
        },
        "query2": {
            "callback": "http://test_callback/",
            "message": {"query_graph": QGRAPH2},
        },
    }
    queue = asyncio.Queue()
    async with callback_overlay("http://test_callback/", queue):
        response = await client.post("/multiquery", json=q)
        output = response.json()
        assert response.status_code == 200
        assert not output
        outputs = []
        outputs.append(await queue.get())
        outputs.append(await queue.get())
        output_final = await queue.get()

    expected_status = {
        "message": {},
        "status_communication": {"strider_multiquery_status": "complete"},
    }

    assert dict(output_final) == expected_status
