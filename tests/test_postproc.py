import logging
import pytest
from reasoner_pydantic import Response

from tests.helpers.context import (
    with_norm_overlay,
)
from tests.helpers.mock_responses import kp_response

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
    MONDO:0005148 information_content 2
""",
)
async def test_node_filtered():
    """
    Test that node with an information content lower than the threshold are removed
    """
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
