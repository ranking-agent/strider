"""Test Fetcher."""

import httpx
import json
import pytest
from pytest_httpx import HTTPXMock
import redis.asyncio

from fastapi.responses import Response

from tests.helpers.redisMock import redisMock
from tests.helpers.logger import setup_logger
from tests.helpers.utils import query_graph_from_string, get_normalizer_response
import tests.helpers.mock_responses as mock_responses

from strider.fetcher import Fetcher
from strider.config import settings
from strider.server import APP

# Switch prefix path before importing server
settings.normalizer_url = "http://normalizer"


@pytest.fixture()
async def client():
    """Yield httpx client."""
    async with httpx.AsyncClient(app=APP, base_url="http://test") as client_:
        yield client_


logger = setup_logger()


normalizer_data = """
UMLS:C0156146 categories biolink:Disease
UMLS:C0156146 information_content 78
MONDO:0021074 categories biolink:Disease
MONDO:0021074 information_content 95
"""


@pytest.mark.asyncio
async def test_fetcher_bad_response(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test when a KP returns null query graph.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(
        url="http://normalizer/get_normalized_nodes",
        json=get_normalizer_response(normalizer_data),
    )
    httpx_mock.add_response(
        url="http://kp1/query", json=mock_responses.response_with_pinned_node_subclasses
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:Disease ))
        n0(( ids[] MONDO:0005011 ))
        n0-- biolink:related_to -->n1
        n1(( category biolink:NamedThing ))
        """
    )

    fetcher = Fetcher(logger, False, {})
    await fetcher.setup(QGRAPH, {}, 75)

    num_responses = 0

    async with fetcher:
        async for result_kgraph, result, result_auxgraph, sub_qid in fetcher.lookup(
            None
        ):
            num_responses += 1
    # we shouldn't remove any results
    assert num_responses == len(
        mock_responses.response_with_pinned_node_subclasses["message"]["results"]
    )
