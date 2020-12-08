"""Test Strider."""
import asyncio
import itertools
import json
import os

import pytest

# Set redis host before importing
os.environ["REDIS_HOST"] = "fakeredis"

from strider.server import sync_query

from .util import with_translator_overlay
from .logger import setup_logger

from reasoner_pydantic import Query, Message, QueryGraph

setup_logger()

with open("tests/data/query_graphs/two_hop.json", "r") as stream:
    QGRAPH = json.load(stream)

os.environ["PREFIXES"] = "tests/data/prefixes.json"
os.environ["KPREGISTRY_URL"] = "http://registry"
os.environ["NORMALIZER_HOST"] = "http://normalizer"

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
            message = Message(
                query_graph = QueryGraph.parse_obj(QGRAPH)
            )
        )

    # Run
    output = await sync_query(q)

    print(output.logs)

    assert output
    # Check for any errors in the log
    assert all(l.level != 'ERROR' for l in output.logs)
    # Ensure we have some results
    assert len(output.message.results) > 0

    print("========================= RESULTS =========================")
    print(output.message.results)
    print(output.message.knowledge_graph)
