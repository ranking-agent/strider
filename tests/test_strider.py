"""Test Strider."""
import asyncio
import itertools
import json
import os

from reasoner_pydantic import Query, Message, QueryGraph
import pytest

from .logger import setup_logger
from .util import with_translator_overlay

from strider.config import settings

# Switch settings before importing server
settings.redis_url = "redis://fakeredis"
settings.prefixes_path = "tests/data/prefixes.json"
settings.kpregistry_url = "http://registry"
settings.normalizer_url = "http://normalizer"

from strider.server import sync_query


setup_logger()

with open("tests/data/query_graphs/two_hop.json", "r") as stream:
    QGRAPH = json.load(stream)

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
@with_translator_overlay([
    ("ctd", CTD_PREFIXES),
    ("hetio", DEFAULT_PREFIXES),
    ("mychem", MYCHEM_PREFIXES),
])
async def test_strider():
    """Test Strider."""

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
