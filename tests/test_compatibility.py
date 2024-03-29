import logging
import pytest
from fastapi.responses import JSONResponse, Response
from reasoner_pydantic.message import Message

from tests.helpers.context import (
    with_norm_overlay,
    with_response_overlay,
)

from strider.config import settings

# Modify settings before importing things
settings.normalizer_url = "http://normalizer"

from strider.knowledge_provider import KnowledgeProvider

logger = logging.getLogger(__name__)
kp = {
    "url": "http://test",
    "details": {
        "preferred_prefixes": {
            "biolink:Disease": ["MONDO"],
        },
    },
}


@pytest.mark.asyncio
@with_norm_overlay(
    settings.normalizer_url,
    """
    MONDO:0005148 categories biolink:Disease
    MONDO:0005148 synonyms DOID:9352
""",
)
async def test_map_prefixes_small_example():
    """
    Test that prefixes are mapped properly and that already
    mapped prefixes are unchanged.
    """
    provider = KnowledgeProvider("test", kp, logger)

    preferred_prefixes = {"biolink:Disease": ["MONDO"]}

    query_graph = {
        "nodes": {
            "n0": {"ids": ["DOID:9352"]},
            "n1": {"ids": ["MONDO:0005148"]},
        },
        "edges": {},
    }

    msg = Message.parse_obj({"query_graph": query_graph})

    await provider.map_prefixes(
        msg,
        preferred_prefixes,
    )
    msg = msg.dict()

    # n0 should be converted to the correct prefix
    assert msg["query_graph"]["nodes"]["n0"]["ids"] == ["MONDO:0005148"]
    # There should be no change to n1
    assert msg["query_graph"]["nodes"]["n1"]["ids"] == ["MONDO:0005148"]


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
async def test_unknown_prefix():
    """
    Test that if passed an unknown prefix we
    assume it doesn't need to be changed.
    """
    provider = KnowledgeProvider("test", kp, logger)

    preferred_prefixes = {"biolink:Disease": ["MONDO"]}

    query_graph = {
        "nodes": {
            "n0": {"ids": ["UNKNOWN:000000"]},
        },
        "edges": {},
    }

    msg = Message.parse_obj({"query_graph": query_graph})

    await provider.map_prefixes(
        msg,
        preferred_prefixes,
    )

    # n0 should be unchanged
    assert msg.dict()["query_graph"]["nodes"]["n0"]["ids"] == ["UNKNOWN:000000"]


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
async def test_prefix_not_specified():
    """
    Test that if we get a category with no preferred_prefixes
    we make no changes.
    """
    provider = KnowledgeProvider("test", kp, logger)

    preferred_prefixes = {}

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


normalizer_error_no_matches = "No matches found for the specified curie(s)"


@pytest.mark.asyncio
@with_response_overlay(
    settings.normalizer_url + "/get_normalized_nodes",
    JSONResponse(content={"detail": normalizer_error_no_matches}, status_code=404),
)
async def test_normalizer_no_synonyms_available(caplog):
    """
    Test that if we send a node with no synonyms
    to the normalizer that we continue working and add
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

    # The error we recieved from the normalizer should be in the logs
    assert normalizer_error_no_matches in caplog.text


@pytest.mark.asyncio
@with_response_overlay(
    settings.normalizer_url,
    Response(
        status_code=500,
        content="Internal server error",
    ),
)
async def test_normalizer_500(caplog):
    """
    Test that if the normalizer returns 500 we make
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

    assert "Request Error contacting Node Normalizer" in caplog.text


@pytest.mark.asyncio
async def test_normalizer_not_reachable(caplog):
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

    assert "Request Error contacting Node Normalizer" in caplog.text
