from pathlib import Path
import os
import json
import logging

import pytest

from tests.helpers.context import with_registry_overlay

from strider.config import settings

# Switch settings before importing strider things
registry_host = "registry"
settings.kpregistry_url = f"http://{registry_host}"

from strider.query_planner import generate_plans


cwd = Path(__file__).parent


with open(cwd / "ex1_kps.json", "r") as f:
    kps = json.load(f)

DEFAULT_PREFIXES = {
    "biolink:Disease": ["MONDO", "DOID"],
    "biolink:ChemicalSubstance": ["CHEBI", "MESH"],
    "biolink:PhenotypicFeature": ["HP"],
}
# Add prefixes
for kp in kps.values():
    kp['details'] = {'preferred_prefixes': DEFAULT_PREFIXES}


@pytest.mark.asyncio
@with_registry_overlay(registry_host, kps)
async def test_permute_curies():
    """ Check that nodes with ID are correctly permuted """

    qg = {
        "nodes": {"n0": {"id": ["MONDO:0005737", "MONDO:0005738"]}},
        "edges": {},
    }

    plans = await generate_plans(qg)

    assert plans
    # We should have two plans
    assert len(plans) == 2


@pytest.mark.asyncio
@with_registry_overlay(registry_host, kps)
async def test_ex1():
    """ Test first example """
    with open(cwd / "ex1_qg.json", "r") as f:
        qg = json.load(f)

    plans = await generate_plans(qg)

    assert plans

    # We should have two valid plans
    assert len(plans) == 2


@pytest.mark.asyncio
@with_registry_overlay(registry_host, kps)
async def test_namedthing(caplog):
    caplog.set_level(logging.DEBUG)
    """ Test NamedThing -related_to-> NamedThing """

    qg = {
        "nodes": {
            "n0": {"type": "biolink:NamedThing"},
            "n1": {"type": "biolink:NamedThing"},
        },
        "edges": {
            "e01": {"subject": "n0", "object": "n1", "predicate": "biolink:related_to"}
        },
    }

    plans = await generate_plans(qg)
