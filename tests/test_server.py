"""Test Strider."""

import asyncio
import json
import redis.asyncio

from fastapi import HTTPException
from fastapi.responses import Response
import httpx
import pytest
from pytest_httpx import HTTPXMock
from reasoner_pydantic import (
  Response as PydanticResponse,
  Query,
  AsyncQuery,
)

from tests.helpers.logger import setup_logger
from tests.helpers.utils import query_graph_from_string, get_normalizer_response
from tests.helpers.redisMock import redisMock
import tests.helpers.mock_responses as mock_responses

import strider
from strider.config import settings
from strider.server import sync_query, async_lookup, multi_query, multi_lookup

# Switch prefix path before importing server
settings.normalizer_url = "http://normalizer"

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
async def test_duplicate_results(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test that we filter out duplicate results if we
    get the same nodes back with slightly different categories.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(url="http://kp0/query", json=mock_responses.duplicate_result_response)
    httpx_mock.add_response(url="http://kp1/query", json=mock_responses.duplicate_result_response_2)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:DiseaseOrPhenotypicFeature ))
        n0-- biolink:related_to -->n1
        """
    )

    # Create query
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}})

    # Run
    response = await sync_query(q)
    output = json.loads(response.body)

    assert len(output["message"]["results"]) == 1


@pytest.mark.asyncio
async def test_merge_results_different_predicates(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test that if we get results from KPs with different predicates
    then the results are not merged together
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(url="http://kp0/query", json=mock_responses.duplicate_result_response)
    httpx_mock.add_response(url="http://kp1/query", json=mock_responses.duplicate_result_response_different_predicate)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:related_to -->n1
        """
    )

    # Create query
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}})

    # Run
    response = await sync_query(q)
    assert response.headers.get("content-type") == "application/json"
    output = json.loads(response.body)

    # Check message structure
    assert len(output["message"]["knowledge_graph"]["edges"]) == 2
    assert len(output["message"]["results"]) == 1
    result = output["message"]["results"][0]
    assert len(result["analyses"]) == 1


@pytest.mark.asyncio
async def test_solve_missing_predicate(monkeypatch, mocker):
    """Test solving a query graph, in which the predicate is missing."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.throttle.ThrottledServer._query",
        return_value=PydanticResponse.parse_obj({"message": {}}),
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
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}})

    # Run
    await sync_query(q)

    query.assert_called_once_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["HP:001"],
                            "categories": ["biolink:Gene"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                        "n1": {
                            "categories": ["biolink:Gene", "biolink:Protein"],
                            "set_interpretation": "BATCH",
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
        False,
        ["infores:kp2"],
        True,
    )


@pytest.mark.asyncio
async def test_solve_missing_category(monkeypatch, mocker):
    """Test solving the ex1 query graph, in which one of the categories is missing."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.throttle.ThrottledServer._query",
        return_value=PydanticResponse.parse_obj({"message": {}}),
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
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}})

    # Run
    await sync_query(q)

    query.assert_called_once_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["CHEBI:6801"],
                            "categories": ["biolink:NamedThing"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                        "n1": {
                            "categories": ["biolink:Disease"],
                            "set_interpretation": "BATCH",
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
        False,
        ["infores:kp1"],
        True,
    )


@pytest.mark.asyncio
async def test_normalizer_different_category(monkeypatch, mocker, httpx_mock: HTTPXMock):
    """
    Test solving a query graph where the category provided doesn't match
    the one in the node normalizer.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(
        url="http://normalizer/get_normalized_nodes",
        json=get_normalizer_response("""
            CHEBI:6801 categories biolink:Vitamin
        """)
    )
    query = mocker.patch(
        "strider.throttle.ThrottledServer._query",
        return_value=PydanticResponse.parse_obj({"message": {}}),
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
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}})

    # Run
    await sync_query(q)

    query.assert_called_once_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["CHEBI:6801"],
                            "categories": ["biolink:Vitamin"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                        "n1": {
                            "categories": ["biolink:Disease"],
                            "set_interpretation": "BATCH",
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
        False,
        ["infores:kp3"],
        True,
    )


@pytest.mark.asyncio
async def test_solve_loop(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test that we correctly solve a query with a loop
    """
    # TODO replace has_phenotype with the correct predicate phenotype_of
    # when BMT is updated to recognize it as a valid predicate
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(
        url="http://normalizer/get_normalized_nodes",
        json=get_normalizer_response("""
            MONDO:0008114 categories biolink:Disease
        """)
    )
    httpx_mock.add_response(url="http://kp1/query", json=mock_responses.kp_response)
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
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}})

    # Run
    await sync_query(q)


@pytest.mark.asyncio
async def test_log_level_param(monkeypatch, httpx_mock: HTTPXMock):
    """Test that changing the log level changes the output"""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(url="http://kp1/query", json=mock_responses.kp_response)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:PhenotypicFeature ))
        n0-- biolink:treats -->n1
        """
    )

    # Create query
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}})

    # Check there are no debug logs
    response = await sync_query(q)
    output = json.loads(response.body)
    assert not any(l["level"] == "DEBUG" for l in output["logs"])

    q.log_level = "DEBUG"

    # Check there are now debug logs
    response = await sync_query(q)
    output = json.loads(response.body)
    assert any(l["level"] == "DEBUG" for l in output["logs"])


@pytest.mark.asyncio
async def test_mutability_bug(monkeypatch):
    """
    Test that qgraph is not mutated between KP calls.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    QGRAPH = {
        "nodes": {
            "n0": {
                "categories": ["biolink:ChemicalSubstance"],
                "set_interpretation": "BATCH",
                "constraints": [],
            },
            "n1": {
                "ids": ["MONDO:0005148"],
                "categories": ["biolink:Disease"],
                "set_interpretation": "BATCH",
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
    }

    # Create query
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}})

    # Run
    response = await sync_query(q)
    output = json.loads(response.body)
    assert output["message"]["query_graph"] == QGRAPH


@pytest.mark.asyncio
async def test_solve_not_real_predicate(monkeypatch, mocker, httpx_mock: HTTPXMock):
    """
    Test that we can solve a query graph with
    predicates that we don't recognize
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(
        url="http://normalizer/get_normalized_nodes",
        json=get_normalizer_response("""
            CHEBI:6801 categories biolink:ChemicalSubstance
            MONDO:0005148 categories biolink:Disease
        """)
    )
    query = mocker.patch(
        "strider.throttle.ThrottledServer._query",
        return_value=PydanticResponse.parse_obj({"message": {}}),
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:not_a_real_predicate -->n1
        """
    )

    # Create query
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}, "log_level": "INFO"})

    # Run
    await sync_query(q)

    query.assert_called_once()


@pytest.mark.asyncio
async def test_exception_response(monkeypatch):
    """
    Test that an exception's response is a 500 error
    and is a valid TRAPI message with a log included
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qgraph = {"nodes": {}, "edges": {}}

    # Temporarily break the fetcher module to induce a 500 error
    _Message = strider.fetcher.Message
    strider.fetcher.Message = None

    q = Query.parse_obj({
        "message": {"query_graph": qgraph},
        "log_level": "DEBUG",
    })
    response = await sync_query(q)

    # Put fetcher back together
    strider.fetcher.Message = _Message

    assert response.status_code == 200
    assert json.loads(response.body)["status_communication"]


@pytest.mark.asyncio
async def test_normalizer_unavailable(monkeypatch):
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
    q = Query.parse_obj({
        "message": {"query_graph": QGRAPH},
        "log_level": "WARNING",
    })

    # Run
    response = await sync_query(q)
    output = json.loads(response.body)

    output_log_messages = [log_entry["message"] for log_entry in output["logs"]]

    # Check that the correct error messages are in the log
    assert "Request Error contacting Node Normalizer" in output_log_messages


@pytest.mark.asyncio
async def test_workflow(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test query workflow handling.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
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
        "workflow": [
            {
                "id": "lookup",
            },
        ],
    })

    # Run
    response = await sync_query(q)
    output = json.loads(response.body)
    assert output["message"]["results"]


@pytest.mark.asyncio
async def test_multiple_identifiers(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test that we correctly handle the case where we have multiple identifiers
    for the preferred prefix
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(
        url="http://normalizer/get_normalized_nodes",
        json=get_normalizer_response("""
            UMLS:C0032961 categories biolink:PhenotypicFeature
            UMLS:C0032961 synonyms NCIT:C25742 NCIT:C92933
            NCIT:C25742 categories biolink:PhenotypicFeature
            NCIT:C25742 synonyms NCIT:C25742 NCIT:C92933
            NCIT:C92933 categories biolink:PhenotypicFeature
            NCIT:C92933 synonyms NCIT:C25742 NCIT:C92933
            CHEBI:2904 categories biolink:ChemicalSubstance
            CHEBI:30146 categories biolink:ChemicalSubstance
        """)
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:PhenotypicFeature ))
        n1(( ids[] UMLS:C0032961 ))
        n0-- biolink:contraindicated_for -->n1
        """
    )

    # Create query
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}})

    # Run
    response = await sync_query(q)
    output = json.loads(response.body)


@pytest.mark.asyncio
async def test_provenance(monkeypatch, httpx_mock: HTTPXMock):
    """
    Tests that provenance is properly reported by strider.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(
        url="http://normalizer/get_normalized_nodes",
        json=get_normalizer_response("""
            MONDO:0005148 categories biolink:Vitamin
        """)
    )
    httpx_mock.add_response(url="http://kp3/query", json=mock_responses.response_with_attributes)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n1(( categories[] biolink:NamedThing ))
        n0-- biolink:related_to -->n1
        """
    )
    q = Query.parse_obj({"message": {"query_graph": QGRAPH}})

    # Run
    response = await sync_query(q)
    output = json.loads(response.body)
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
async def test_async_query(monkeypatch, httpx_mock: HTTPXMock):
    """Test asyncquery endpoint using the ex1 query graph"""
    callback_url = "http://test/"
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(url="http://kp1/query", json=mock_responses.kp_response)
    httpx_mock.add_response(url=callback_url, status_code=200)
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:treats -->n1
        """
    )

    # Run
    await async_lookup(callback_url, {"callback": callback_url, "message": {"query_graph": QGRAPH}})
    requests = httpx_mock.get_requests()
    # this might be finicky. But we expect the last request to be the final response to the callback url
    assert requests[-1].url == callback_url


@pytest.mark.asyncio
async def test_different_callbacks_multiquery():
    """Test multiquery endpoint fails with different callbacks."""
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
        "query1": AsyncQuery.parse_obj({"callback": "http://test1/", "message": {"query_graph": QGRAPH1}}),
        "query2": AsyncQuery.parse_obj({"callback": "http://test2/", "message": {"query_graph": QGRAPH2}}),
    }
    with pytest.raises(HTTPException):
        await multi_query(None, q_error)


@pytest.mark.asyncio
async def test_multi_lookup(monkeypatch, httpx_mock: HTTPXMock):
    """Test multilookup function."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    callback_url = "http://test_callback/"
    expected_status = {
        "message": {},
        "status_communication": {"strider_multiquery_status": "complete"},
    }
    httpx_mock.add_response(url="http://kp1/query", json=mock_responses.kp_response)
    httpx_mock.add_response(url="http://kp3/query", json=mock_responses.kp_response)
    httpx_mock.add_response(url=callback_url, match_json=expected_status)
    httpx_mock.add_response(url=callback_url, status_code=200)
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
            "callback": callback_url,
            "message": {"query_graph": QGRAPH1},
        },
        "query2": {
            "callback": callback_url,
            "message": {"query_graph": QGRAPH2},
        },
    }
    await multi_lookup("123", callback_url, q, list(q.keys()))

    requests = httpx_mock.get_requests()
    print(requests)
