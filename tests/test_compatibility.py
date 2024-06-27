import httpx
import logging
import pytest
from pytest_httpx import HTTPXMock
from fastapi.responses import JSONResponse, Response
from reasoner_pydantic import QueryGraph, Message

from strider.config import settings

# Modify settings before importing things
settings.normalizer_url = "http://normalizer"

from strider.knowledge_provider import KnowledgeProvider
from strider.trapi import map_qgraph_curies

logger = logging.getLogger(__name__)
kp = {
    "url": "http://test",
    "details": {
        "preferred_prefixes": {
            "biolink:Disease": ["MONDO"],
        },
    },
}


def test_map_prefixes_small_example():
    """
    Test that prefixes are mapped properly and that already
    mapped prefixes are unchanged.
    """
    query_graph = {
        "nodes": {
            "n0": {"ids": ["DOID:9352"]},
            "n1": {"ids": ["MONDO:0005148"]},
        },
        "edges": {},
    }
    curie_map = {
        "DOID:9352": ["MONDO:0005148"],
        "MONDO:0005148": ["MONDO:0005148"],
    }

    msg = QueryGraph.parse_obj(query_graph)

    map_qgraph_curies(
        msg,
        curie_map,
    )
    msg = msg.dict()

    # n0 should be converted to the correct prefix
    assert msg["nodes"]["n0"]["ids"] == ["MONDO:0005148"]
    # There should be no change to n1
    assert msg["nodes"]["n1"]["ids"] == ["MONDO:0005148"]


def test_unknown_prefix():
    """
    Test that if passed an unknown prefix we
    assume it doesn't need to be changed.
    """
    query_graph = {
        "nodes": {
            "n0": {"ids": ["UNKNOWN:000000"]},
        },
        "edges": {},
    }
    curie_map = {}

    msg = QueryGraph.parse_obj(query_graph)

    map_qgraph_curies(
        msg,
        curie_map,
    )

    # n0 should be unchanged
    assert msg.dict()["nodes"]["n0"]["ids"] == ["UNKNOWN:000000"]


def test_prefix_not_specified():
    """
    Test that if we get a category with no preferred_prefixes
    we make no changes.
    """
    query_graph = {
        "nodes": {
            "n0": {"ids": ["DOID:9352"]},
        },
        "edges": {},
    }
    curie_map = {}

    msg = QueryGraph.parse_obj(query_graph)

    map_qgraph_curies(
        msg,
        curie_map,
    )

    # n0 should be unchanged
    assert msg.dict()["nodes"]["n0"]["ids"] == ["DOID:9352"]


@pytest.mark.asyncio
async def test_normalizer_no_synonyms_available(httpx_mock: HTTPXMock):
    """
    Test that if we send a node with no synonyms
    to the normalizer that we continue working and add
    a warning to the log
    """
    httpx_mock.add_response(
        url="http://normalizer/get_normalized_nodes", status_code=404
    )
    provider = KnowledgeProvider("test", kp, logger)

    preferred_prefixes = {"biolink:Disease": ["MONDO"]}

    query_graph = {
        "nodes": {
            "n0": {"ids": ["DOID:9352"]},
        },
        "edges": {},
    }

    msg = Message.parse_obj({"query_graph": query_graph})

    await provider.map_prefixes(
        msg,
        preferred_prefixes,
    )

    # n0 should be unchanged
    assert msg.dict()["query_graph"]["nodes"]["n0"]["ids"] == ["DOID:9352"]


@pytest.mark.asyncio
async def test_normalizer_500(httpx_mock: HTTPXMock):
    """
    Test that if the normalizer returns 500 we make
    no changes to the query graph and continue on while adding a
    a warning to the log
    """
    httpx_mock.add_response(
        url="http://normalizer/get_normalized_nodes", status_code=500
    )
    provider = KnowledgeProvider("test", kp, logger)

    preferred_prefixes = {"biolink:Disease": ["MONDO"]}

    query_graph = {
        "nodes": {
            "n0": {"ids": ["DOID:9352"]},
        },
        "edges": {},
    }

    msg = Message.parse_obj({"query_graph": query_graph})

    await provider.map_prefixes(
        msg,
        preferred_prefixes,
    )

    # n0 should be unchanged
    assert msg.dict()["query_graph"]["nodes"]["n0"]["ids"] == ["DOID:9352"]


@pytest.mark.asyncio
async def test_normalizer_not_reachable():
    """
    Test that if the normalizer is completely unavailable we make
    no changes to the query graph and continue on while adding a
    a warning to the log
    """
    provider = KnowledgeProvider("test", kp, logger)

    preferred_prefixes = {"biolink:Disease": ["MONDO"]}

    query_graph = {
        "nodes": {
            "n0": {"ids": ["DOID:9352"]},
        },
        "edges": {},
    }

    msg = Message.parse_obj({"query_graph": query_graph})

    await provider.map_prefixes(
        msg,
        preferred_prefixes,
    )

    # n0 should be unchanged
    assert msg.dict()["query_graph"]["nodes"]["n0"]["ids"] == ["DOID:9352"]
