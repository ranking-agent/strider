"""Test Strider."""
import asyncio
import itertools
import json
import os

import pytest

from strider.fetcher import StriderWorker

from .util import with_translator_overlay
from .logger import setup_logger

setup_logger()

with open("tests/data/query_graphs/two_hop.json", "r") as stream:
    QGRAPH = json.load(stream)

os.environ["PREFIXES"] = "tests/data/prefixes.json"

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
    queue = asyncio.PriorityQueue()
    counter = itertools.count()

    # setup strider
    strider = StriderWorker(
        queue,
        num_workers=2,
        counter=counter,
    )

    await strider.run(QGRAPH, wait=True)
    assert strider.results
    print("========================= RESULTS =========================")
    print(strider.results)
    print(strider.kgraph)
