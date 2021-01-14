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
from strider.server import sync_query, generate_traversal_plan


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
async def test_solve_ex1_two_hop():
    """Test solving the ex1_two_hop query graph"""
    with open(cwd / "ex1_two_hop.json", "r") as f:
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
    # Ensure we have a knowledge graph with nodes and edges
    assert len(output['message']['knowledge_graph']['nodes']) > 0
    assert len(output['message']['knowledge_graph']['edges']) > 0

    print("========================= RESULTS =========================")
    print(output['message']['results'])
    print(output['message']['knowledge_graph'])


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    [
        ("ctd", CTD_PREFIXES),
        ("hetio", DEFAULT_PREFIXES),
        ("mychem", MYCHEM_PREFIXES),
    ])
async def test_plan_ex1_two_hop():
    """Test /plan endpoint"""
    with open(cwd / "ex1_two_hop.json", "r") as f:
        QGRAPH = json.load(f)

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(QGRAPH)
        )
    )

    # Run
    output = await generate_traversal_plan(q)

    assert output

    # Two steps in the plan each with KPs to contact
    assert len(output[('n0', 'e01', 'n1')]) == 2
    assert len(output[('n1', 'e12', 'n2')]) == 1
