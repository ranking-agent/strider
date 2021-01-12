import json
import pytest
from fastapi.responses import JSONResponse

from tests.helpers.context import \
    with_norm_overlay, with_response_overlay

from strider.config import settings

# Modify settings before importing things
settings.kpregistry_url = "http://registry"
settings.normalizer_url = "http://normalizer"

from strider.compatibility import \
    KnowledgePortal


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
async def test_map_prefixes_small_example():
    """
    Test that prefixes are mapped properly and that already
    mapped prefixes are unchanged.
    """
    portal = KnowledgePortal()

    preferred_prefixes = {
        "biolink:Disease": [
            "MONDO"
        ]
    }

    query_graph = {
        "nodes": {
            "n0": {
                "id": "DOID:9352"
            },
            "n1": {
                "id": "MONDO:0005148"
            },
        },
        "edges": {
        },
    }

    fixed_msg = await portal.map_prefixes(
        {"query_graph": query_graph},
        preferred_prefixes,
    )

    # n0 should be converted to the correct prefix
    assert fixed_msg['query_graph']['nodes']['n0']['id'] == "MONDO:0005148"
    # There should be no change to n1
    assert fixed_msg['query_graph']['nodes']['n1']['id'] == "MONDO:0005148"


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
async def test_unknown_prefix():
    """
    Test that if passed an unknown prefix we 
    assume it doesn't need to be changed.
    """
    portal = KnowledgePortal()

    preferred_prefixes = {
        "biolink:Disease": [
            "MONDO"
        ]
    }

    query_graph = {
        "nodes": {
            "n0": {
                "id": "UNKNOWN:000000"
            },
        },
        "edges": {
        },
    }

    fixed_msg = await portal.map_prefixes(
        {"query_graph": query_graph},
        preferred_prefixes,
    )

    # n0 should be unchanged
    assert fixed_msg['query_graph']['nodes']['n0']['id'] == "UNKNOWN:000000"


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
async def test_prefix_not_specified():
    """
    Test that if we get a category with no preferred_prefixes
    we make no changes.
    """
    portal = KnowledgePortal()

    preferred_prefixes = {}

    query_graph = {
        "nodes": {
            "n0": {
                "id": "DOID:9352"
            },
        },
        "edges": {
        },
    }

    fixed_msg = await portal.map_prefixes(
        {"query_graph": query_graph},
        preferred_prefixes,
    )

    # n0 should be unchanged
    assert fixed_msg['query_graph']['nodes']['n0']['id'] == "DOID:9352"


@pytest.mark.asyncio
@with_response_overlay(
    settings.normalizer_url,
    JSONResponse(
        content={"detail": "No matches found for the specified curie(s)"},
        status_code=404
    )
)
async def test_normalizer_not_available():
    """
    Test that if we get an invalid response
    from the normalizer we make no changes.

    This is the current behavior of the normalizer if
    we send an unknown node.
    """
    portal = KnowledgePortal()

    preferred_prefixes = {
        "biolink:Disease": [
            "MONDO"
        ]
    }

    query_graph = {
        "nodes": {
            "n0": {
                "id": "DOID:9352"
            },
        },
        "edges": {
        },
    }

    fixed_msg = await portal.map_prefixes(
        {"query_graph": query_graph},
        preferred_prefixes,
    )

    # n0 should be unchanged
    assert fixed_msg['query_graph']['nodes']['n0']['id'] == "DOID:9352"
