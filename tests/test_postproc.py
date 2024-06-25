import copy
import logging
import pytest
from pytest_httpx import HTTPXMock
from reasoner_pydantic import Response

from tests.helpers.mock_responses import (
    kp_response,
    response_with_aux_graphs,
    blocked_response,
)
from tests.helpers.utils import get_normalizer_response

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
async def test_node_filtered(httpx_mock: HTTPXMock):
    """
    Test that node with an information content lower than the threshold are removed
    """
    httpx_mock.add_response(
        url="http://normalizer/get_normalized_nodes",
        json=get_normalizer_response(
            """
            MONDO:0005148 categories biolink:Disease
            MONDO:0005148 synonyms DOID:9352
            MONDO:0005148 information_content 2
        """
        ),
    )
    provider = KnowledgeProvider("test", kp, logger)

    preferred_prefixes = {"biolink:Disease": ["MONDO"]}

    msg = Response.parse_obj(kp_response)

    processor = provider.get_postprocessor(preferred_prefixes)

    assert "MONDO:0005148" in msg.message.knowledge_graph.nodes
    assert len(msg.message.results) == 1

    await processor(msg, False)

    # MONDO:0005148 should be removed
    assert "MONDO:0005148" not in msg.message.knowledge_graph.nodes
    assert len(msg.message.results) == 0


@pytest.mark.asyncio
async def test_aux_graph_filtering(httpx_mock: HTTPXMock):
    """
    Test that node with an information content lower than the threshold are removed
    """
    httpx_mock.add_response(
        url="http://normalizer/get_normalized_nodes",
        json=get_normalizer_response(
            """
            MONDO:0005148 categories biolink:Disease
            MONDO:0005148 synonyms DOID:9352
            MONDO:0005148 information_content 100
            MESH:D008687 categories biolink:ChemicalEntity
            MESH:D008687 synonyms PUBCHEM.COMPOUND:4901
            MESH:D008687 information_content 100
            MESH:D014867 categories biolink:ChemicalEntity
            MESH:D014867 synonyms PUBCHEM.COMPOUND:4901
            MESH:D014867 information_content 74
        """
        ),
    )
    provider = KnowledgeProvider("test", kp, logger)

    preferred_prefixes = {"biolink:Disease": ["MONDO"]}

    response = copy.deepcopy(response_with_aux_graphs)

    msg = Response.parse_obj(response)

    assert len(list(msg.message.auxiliary_graphs.keys())) == 1
    assert len(msg.message.results) == 2

    processor = provider.get_postprocessor(preferred_prefixes)

    await processor(msg, False)

    # test aux graph should be removed
    assert len(list(msg.message.auxiliary_graphs.keys())) == 0
    assert len(msg.message.results) == 1


@pytest.mark.asyncio
async def test_blocklist(httpx_mock: HTTPXMock):
    """
    Test that nodes in the blocklist are not taken in as results
    """
    httpx_mock.add_response(
        url="http://normalizer/get_normalized_nodes",
        json=get_normalizer_response(
            """
            MESH:D014867 categories biolink:SmallMolecule
            MESH:D014867 synonyms MESH:D000838
        """
        ),
    )
    provider = KnowledgeProvider("test", kp, logger)

    preferred_prefixes = {"biolink:Disease": ["MONDO"]}

    msg = Response.parse_obj(blocked_response)

    processor = provider.get_postprocessor(preferred_prefixes)

    assert "MESH:D014867" in msg.message.knowledge_graph.nodes
    assert "MESH:D000588" in msg.message.knowledge_graph.nodes
    assert len(msg.message.results) == 2

    await processor(msg, False)

    # MONDO:0005148 should be removed
    assert "MESH:D000588" not in msg.message.knowledge_graph.nodes
    assert "MESH:D014867" not in msg.message.knowledge_graph.nodes
    assert len(msg.message.results) == 0


@pytest.mark.asyncio
async def test_aux_graph_edges_are_kept(httpx_mock: HTTPXMock):
    """
    Test that node with an information content lower than the threshold are removed
    """
    httpx_mock.add_response(
        url="http://normalizer/get_normalized_nodes",
        json=get_normalizer_response(
            """
            MONDO:0005148 categories biolink:Disease
            MONDO:0005148 synonyms DOID:9352
            MONDO:0005148 information_content 100
            MESH:D008687 categories biolink:ChemicalEntity
            MESH:D008687 synonyms PUBCHEM.COMPOUND:4901
            MESH:D008687 information_content 100
        """
        ),
    )
    provider = KnowledgeProvider("test", kp, logger)

    preferred_prefixes = {"biolink:Disease": ["MONDO"]}

    extra_edge_2 = {
        "subject": "MONDO:0005148",
        "object": "MESH:D014867",
        "predicate": "biolink:subclass_of",
        "attributes": [
            {
                "value": "infores:kp1",
                "attribute_type_id": "biolink:knowledge_source",
            },
        ],
        "sources": [
            {
                "resource_id": "infores:kp1",
                "resource_role": "primary_knowledge_source",
            },
        ],
    }

    response = copy.deepcopy(response_with_aux_graphs)

    response["message"]["knowledge_graph"]["edges"]["extra_edge_2"] = extra_edge_2

    response["message"]["auxiliary_graphs"]["1"]["edges"].append("extra_edge_2")

    msg = Response.parse_obj(response)

    assert len(list(msg.message.knowledge_graph.edges.keys())) == 3

    processor = provider.get_postprocessor(preferred_prefixes)

    await processor(msg, False)

    print(msg.message.auxiliary_graphs.json())

    assert len(list(msg.message.auxiliary_graphs.keys())) == 1
    assert len(msg.message.results) == 2
    # extra edge should be kept
    assert len(list(msg.message.knowledge_graph.edges.keys())) == 3
