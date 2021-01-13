from pathlib import Path
from functools import partial
import os
import json
import logging
import itertools
import random
import time

import pytest
from bmt import Toolkit as BMToolkit

from tests.helpers.context import \
    with_registry_overlay, with_norm_overlay


from strider.query_planner import \
    generate_plan, find_valid_permutations, permute_qg, expand_qg, NoAnswersError

from strider.util import WrappedBMT
WBMT = WrappedBMT()

from strider.config import settings

cwd = Path(__file__).parent

# Switch prefix path before importing server
settings.kpregistry_url = "http://registry"
settings.normalizer_url = "http://normalizer"


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
@with_registry_overlay(settings.kpregistry_url, [])
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
@with_registry_overlay(settings.kpregistry_url, [])
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
    # 4 * 2 * 3 * 2 = 48
    # permutations
    assert len(list(permutations)) == 48


simple_kp = load_kps("simple_kp.json")


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, simple_kp)
@with_norm_overlay(settings.normalizer_url)
async def test_not_enough_kps():
    """
    Check we get no plans when we submit a query graph
    that has edges we can't solve
    """
    qg = {
        "nodes": {
            "n0": {"category": "biolink:ExposureEvent"},
            "n1": {"category": "biolink:Drug"},
        },
        "edges": {
            "e01": {
                "subject": "n0",
                "object": "n1",
                "predicate": "biolink:related_to",
            },
        },
    }
    with pytest.raises(NoAnswersError):
        await generate_plan(
            qg,
            logger=logging.getLogger()
        )


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, simple_kp)
@with_norm_overlay(settings.normalizer_url)
async def test_no_path_from_pinned_node():
    """
    Check there is no plan when
    we submit a graph where there is a pinned node
    but no path through it
    """
    qg = {
        "nodes": {
            "n0": {"id": "MONDO:0005148"},
            "n1": {"category": "biolink:Drug"},
        },
        "edges": {
            "e01": {
                "subject": "n1",
                "predicate": "biolink:treats",
                "object": "n0",
            },
        },
    }
    # We should have valid permutations
    permutations = await find_valid_permutations(qg)
    assert len(list(permutations))

    # We should not have any plans
    with pytest.raises(NoAnswersError):
        await generate_plan(
            qg,
            logger=logging.getLogger(),
        )

ex1_kps = load_kps("ex1_kps.json")


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, ex1_kps)
@with_norm_overlay(settings.normalizer_url)
async def test_valid_permute_ex1():
    """ Test first example is permuted correctly """
    with open(cwd / "ex1_qg.json", "r") as f:
        qg = json.load(f)

    plans = await find_valid_permutations(qg)

    assert plans

    # We should have one valid plan
    assert len(plans) == 1


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, ex1_kps)
@with_norm_overlay(settings.normalizer_url)
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


# Generate some KPs using permutations of BMT
# if we consume all of the KPs, this will create a KP
# for every operation
def create_kp(args):
    source, edge, target = args
    return {
        "url": "http://mykp",
        "operations": [{
            "source_type": source,
            "edge_type": f"-{edge}->",
            "target_type": target,
        }]
    }


kp_generator = map(
    create_kp,
    itertools.product(
        WBMT.get_descendants('biolink:NamedThing')[:25],
        WBMT.get_descendants('biolink:related_to')[:10],
        WBMT.get_descendants('biolink:NamedThing')[:25],
    )
)

many_kps = {str(i): kp for i, kp in enumerate(kp_generator)}


async def time_and_display(f, msg):
    """ Time a function and print the time """
    start_time = time.time()
    await f()
    total = time.time() - start_time
    print("-------------------------------------------")
    print(f"Total time to {msg}: {total}")
    print("-------------------------------------------")


@pytest.mark.asyncio
@pytest.mark.longrun
@with_registry_overlay(settings.kpregistry_url, many_kps)
@with_norm_overlay(settings.normalizer_url)
async def test_planning_performance():
    """
    Test our performance when planning a very generic query graph
    with a lot of KPs available.
    """

    qg = {
        "nodes": {
            "n0": {"id": "MONDO:000000"},
            "n1": {"category": "biolink:NamedThing"},
            "n2": {"category": "biolink:NamedThing"},
            "n3": {"category": "biolink:NamedThing"},
        },
        "edges": {
            "e01": {"subject": "n0", "object": "n1", "predicate": "biolink:related_to"},
            "e12": {"subject": "n1", "object": "n2", "predicate": "biolink:related_to"},
            "e23": {"subject": "n2", "object": "n3", "predicate": "biolink:related_to"},
        },
    }

    await time_and_display(
        partial(generate_plan, qg, logger=logging.getLogger()),
        "generate plan for a generic query graph",
    )
