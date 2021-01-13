import json
import pytest
from fastapi.responses import JSONResponse, Response

from tests.helpers.context import \
    with_norm_overlay, with_response_overlay, with_translator_overlay

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


normalizer_error_no_matches = "No matches found for the specified curie(s)"


@pytest.mark.asyncio
@with_response_overlay(
    settings.normalizer_url,
    JSONResponse(
        content={"detail": normalizer_error_no_matches},
        status_code=404
    )
)
async def test_normalizer_no_synonyms_available(caplog):
    """
    Test that if we send a node with no synonyms
    to the normalizer that we continue working and add
    a warning to the log
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

    # The error we recieved from the normalizer should be in the logs
    assert normalizer_error_no_matches in caplog.text


@pytest.mark.asyncio
@with_response_overlay(
    settings.normalizer_url,
    Response(
        status_code=500,
        content="Internal server error",
    )
)
async def test_normalizer_not_reachable(caplog):
    """
    Test that if the normalizer is completely unavailable we make
    no changes to the query graph and continue on while adding a
    a warning to the log
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

    assert "Failed to contact normalizer" in caplog.text

CTD_PREFIXES = {
    "biolink:Disease": ["MONDO"],
    "biolink:ChemicalSubstance": ["CHEBI"],
}


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    [("ctd", CTD_PREFIXES)]
)
async def test_fetch():
    """
    Test that the fetch method converts to and from
    the specified prefixes when contacting a KP
    """
    portal = KnowledgePortal()

    preferred_prefixes = {
        "biolink:Disease": ["DOID"],
        "biolink:ChemicalSubstance": ["MESH"],
    }

    query_graph = {
        "nodes": {
            "n0": {
                "id": "MESH:D008687"
            },
            "n1": {
                "category": "biolink:Disease"
            },
        },
        "edges": {
            "e01": {
                "subject": "n0",
                "object": "n1",
                "predicate": "biolink:treats"
            }
        },
    }

    response = await portal.fetch(
        url="http://ctd/query",
        request={"query_graph": query_graph},
        input_prefixes=CTD_PREFIXES,
        output_prefixes=preferred_prefixes,
    )

    allowed_response_prefixes = [
        prefix for prefix_list in preferred_prefixes.values()
        for prefix in prefix_list
    ]

    # Check query graph node prefixes
    for node in response['query_graph']['nodes'].values():
        if node.get('id', None):
            assert any(node['id'].startswith(prefix)
                       for prefix in allowed_response_prefixes)
    # Check node binding prefixes
    for result in response['results']:
        for binding_list in result['node_bindings'].values():
            for binding in binding_list:
                assert any(binding['id'].startswith(prefix)
                           for prefix in allowed_response_prefixes)
