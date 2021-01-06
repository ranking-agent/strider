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

from strider.query_planner import \
    generate_plan, find_valid_permutations, permute_qg, expand_qg


cwd = Path(__file__).parent


def load_kps(fname):
    """ Load KPs from a file for use in a test """
    with open(cwd / fname, "r") as f:
        kps = json.load(f)
    DEFAULT_PREFIXES = {
        "biolink:Disease": ["MONDO", "DOID"],
        "biolink:ChemicalSubstance": ["CHEBI", "MESH"],
        "biolink:PhenotypicFeature": ["HP"],
    }
    # Add prefixes
    for kp in kps.values():
        kp['details'] = {'preferred_prefixes': DEFAULT_PREFIXES}
    return kps


@pytest.mark.asyncio
@with_registry_overlay(registry_host, [])
async def test_permute_curies():
    """ Check that nodes with ID are correctly permuted """

    qg = {
        "nodes": {"n0": {"id": ["MONDO:0005737", "MONDO:0005738"]}},
        "edges": {},
    }

    permutations = permute_qg(qg)

    assert permutations
    # We should have two plans
    assert len(list(permutations)) == 2


@pytest.mark.asyncio
@with_registry_overlay(registry_host, [])
async def test_permute_simple(caplog):
    """ Check that a simple permutation is done correctly using Biolink """

    qg = {
        "nodes": {
            "n0": {"category": "biolink:AnatomicalEntity"},  # Three children
            "n1": {"category": "biolink:Protein"},     # One child
        },
        "edges": {
            "e01": {
                "subject": "n0",
                # Two children and two directions
                "predicate": "biolink:affects_abundance_of",
                "object": "n1",
            },
        },
    }

    qg = expand_qg(qg, logging.getLogger())
    permutations = permute_qg(qg)
    assert permutations

    # We should have:
    # 4 * 2 * 3 * 2 = 24
    # permutations
    assert len(list(permutations)) == 48


simple_kp = load_kps("simple_kp.json")


@pytest.mark.asyncio
@with_registry_overlay(registry_host, simple_kp)
async def no_path_from_pinned_node():
    """ 
    Check that when we submit a graph where there is a pinned node
    but no path through it that we get no plans back
    """
    qg = {
        "nodes": {
            "n0": {"id": "MONDO:0005737"},
            "n1": {"category": "biolink:Drug"},
        },
        "edges": {
            "e01": {
                "subject": "n0",
                # Two children and two directions
                "predicate": "biolink:treats",
                "object": "n1",
            },
        },
    }
    plan = await generate_plan(qg)
    assert not plan

ex1_kps = load_kps("ex1_kps.json")


@pytest.mark.asyncio
@with_registry_overlay(registry_host, ex1_kps)
async def test_valid_permute_ex1():
    """ Test first example is permuted correctly """
    with open(cwd / "ex1_qg.json", "r") as f:
        qg = json.load(f)

    plans = await find_valid_permutations(qg)

    assert plans

    # We should have two valid plans
    assert len(plans) == 1


@pytest.mark.asyncio
@with_registry_overlay(registry_host, ex1_kps)
async def test_plan_ex1():
    """ Test that we get a good plan for our first example """
    with open(cwd / "ex1_qg.json", "r") as f:
        qg = json.load(f)

    plan = await generate_plan(qg)
    assert plan
    # One step per edge
    assert len(plan.keys()) == len(qg['edges'])

    # First step in the plan starts
    # at a pinned node
    step_1 = next(iter(plan.keys()))
    assert 'id' in qg['nodes'][step_1.source]


namedthing_kps = load_kps("namedthing_kps.json")


@pytest.mark.asyncio
@pytest.mark.longrun  # Don't run by default
@with_registry_overlay(registry_host, namedthing_kps)
async def test_permute_namedthing(caplog):
    """ Test NamedThing -related_to-> NamedThing """
    caplog.set_level(logging.DEBUG)

    qg = {
        "nodes": {
            "n0": {"type": "biolink:NamedThing"},
            "n1": {"type": "biolink:NamedThing"},
        },
        "edges": {
            "e01": {"subject": "n0", "object": "n1", "predicate": "biolink:related_to"}
        },
    }

    # Based on the biolink hierarchy this should build 1.2 million permutations
    # and then filter down to the number of operations (4)
    plans = await find_valid_permutations(qg)

    assert plans
    assert len(plans) == 4
