"""Test weird kp responses."""
import fakeredis
import httpx
import json
import pytest

from fastapi.responses import Response

from tests.helpers.context import (
    with_translator_overlay,
    with_registry_overlay,
    with_norm_overlay,
    with_response_overlay,
)
from tests.helpers.logger import setup_logger
from tests.helpers.utils import query_graph_from_string

from strider.server import APP
from strider.config import settings
from strider.storage import get_client

# Switch prefix path before importing server
settings.kpregistry_url = "http://registry"
settings.normalizer_url = "http://normalizer"

APP.dependency_overrides[get_client] = lambda: fakeredis.FakeRedis(
    encoding="utf-8",
    decode_responses=True,
)


@pytest.fixture()
async def client():
    """Yield httpx client."""
    async with httpx.AsyncClient(app=APP, base_url="http://test") as client_:
        yield client_


setup_logger()


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_registry_overlay(
    settings.kpregistry_url,
    {
        "ctd": {
            "url": "http://ctd/query",
            "infores": "strider",
            "maturity": "development",
            "operations": [
                {
                    "subject_category": "biolink:ChemicalSubstance",
                    "predicate": "biolink:treats",
                    "object_category": "biolink:Disease",
                }
            ],
            "details": {"preferred_prefixes": {}},
        }
    },
)
@with_response_overlay(
    "http://ctd/query",
    Response(
        status_code=200,
        content=json.dumps({"message": {}}),
    ),
)
async def test_kp_response_empty_message(client):
    """
    Test when a KP returns null query graph.
    """
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:ChemicalSubstance ))
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


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    kp_data={
        "kp0": """
            MONDO:1(( category biolink:Disease ))
            MONDO:1-- predicate biolink:related_to -->Gene:1
            MONDO:1-- predicate biolink:related_to -->Gene:2
            Gene:1(( category biolink:Gene ))
            Gene:2(( category biolink:Gene ))
            MONDO:2(( category biolink:Disease ))
            MONDO:2-- predicate biolink:related_to -->Gene:1
            MONDO:2-- predicate biolink:related_to -->Gene:2
        """,
        "kp1": """
            MONDO:1(( category biolink:Disease ))
            MONDO:1-- predicate biolink:related_to -->Gene:1
            Gene:1(( category biolink:Gene ))
            Gene:1-- predicate biolink:related_to -->MONDO:2
            MONDO:2(( category biolink:Disease ))
        """,
        "kp2": """
            MONDO:1(( category biolink:Disease ))
            MONDO:1-- predicate biolink:related_to -->Gene:1
            Gene:1(( category biolink:Gene ))
            Gene:1-- predicate biolink:related_to -->MONDO:2
            MONDO:2(( category biolink:Disease ))
        """,
    },
    normalizer_data="""
        MONDO:1 categories biolink:Disease
        Gene:1 categories biolink:Gene
        MONDO:2 categories biolink:Disease
        """,
)
@with_response_overlay(
    "http://kp1/query",
    Response(
        status_code=200,
        content=json.dumps(
            {
                "message": {},
            }
        ),
    ),
)
async def test_kp_response_empty_message_pinned_two_hop(client):
    """
    Test when a KP returns null query graph.
    """
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
        "log_level": "WARNING",
    }

    # Run
    response = await client.post("/query", json=q)
    output = response.json()
    assert len(output["logs"]) == 1
    assert output["logs"][0]["message"] == "Something went wrong while querying kp1"
