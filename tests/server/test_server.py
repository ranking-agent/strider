"""Test Strider."""
from pathlib import Path
import asyncio
import itertools
import json
import os
import httpx

from reasoner_pydantic import Query, Message, QueryGraph
import pytest

from tests.helpers.logger import setup_logger
from tests.helpers.context import with_translator_overlay

from strider.config import settings

cwd = Path(__file__).parent

# Switch prefix path before importing server
settings.prefixes_path = cwd / "prefixes.json"
settings.kpregistry_url = "http://registry"
settings.normalizer_url = "http://normalizer"
settings.redis_url = "redis://fakeredis"


from strider.kp_registry import Registry
from strider.server import sync_query


setup_logger()

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
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    [
        ("ctd", CTD_PREFIXES),
        ("hetio", DEFAULT_PREFIXES),
        ("mychem", MYCHEM_PREFIXES),
    ])
async def test_strider():
    """Test Strider."""
    with open(cwd / "ex1_qg.json", "r") as f:
        QGRAPH = json.load(f)

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(QGRAPH)
        )
    )

    # Run
    output = await sync_query(q)

    assert output
    # Check for any errors in the log
    assert all(l['level'] != 'ERROR' for l in output['logs'])
    # Ensure we have some results
    assert len(output['message']['results']) > 0

    print("========================= RESULTS =========================")
    print(output['message']['results'])
    print(output['message']['knowledge_graph'])
