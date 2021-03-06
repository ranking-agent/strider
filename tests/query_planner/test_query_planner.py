from pathlib import Path
from functools import partial
import os
import math
import json
import logging
import random

import pytest
from bmt import Toolkit as BMToolkit

from tests.helpers.context import \
    with_registry_overlay, with_norm_overlay
from tests.helpers.utils import load_kps, generate_kps, \
    time_and_display, query_graph_from_string, kps_from_string
from tests.helpers.logger import assert_no_level


from strider.query_planner import \
    generate_plans, find_valid_permutations, permute_qg, expand_qg, NoAnswersError

from strider.config import settings

cwd = Path(__file__).parent

# Switch prefix path before importing server
settings.kpregistry_url = "http://registry"
settings.normalizer_url = "http://normalizer"

LOGGER = logging.getLogger()


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, [])
async def test_permute_curies(caplog):
    """ Check that nodes with ID are correctly permuted """

    qg = {
        "nodes": {"n0": {"id": ["MONDO:0005737", "MONDO:0005738"]}},
        "edges": {},
    }

    permutations = permute_qg(qg)

    assert permutations
    # We should have two plans
    assert len(list(permutations)) == 2
    assert_no_level(caplog, logging.WARNING)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, [])
async def test_permute_simple(caplog):
    """ Check that a simple permutation is done correctly using Biolink """

    qg = query_graph_from_string(
        """
        n0(( category biolink:AnatomicalEntity ))
        n1(( category biolink:Protein ))
        n0-- biolink:affects_abundance_of -->n1
        """
    )

    qg = expand_qg(qg, logging.getLogger())
    permutations = permute_qg(qg)
    assert permutations

    # We should have:
    # 4 * 2 * 3 = 24
    # permutations
    assert len(list(permutations)) == 24
    assert_no_level(caplog, logging.WARNING)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Drug -biolink:treats-> biolink:Disease
    """
))
@with_norm_overlay(settings.normalizer_url)
async def test_not_enough_kps(caplog):
    """
    Check we get no plans when we submit a query graph
    that has edges we can't solve
    """

    qg = query_graph_from_string(
        """
        n0(( category biolink:ExposureEvent ))
        n1(( category biolink:Drug ))
        n0-- biolink:related_to -->n1
        """
    )

    plans = await generate_plans(
        qg,
        logger=logging.getLogger()
    )
    assert len(plans) == 0
    assert_no_level(caplog, logging.WARNING, 1)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Drug -biolink:treats-> biolink:Disease
    """
))
@with_norm_overlay(settings.normalizer_url)
async def test_no_path_from_pinned_node(caplog):
    """
    Check there is no plan when
    we submit a graph where there is a pinned node
    but no path through it
    """

    qg = query_graph_from_string(
        """
        n0(( id MONDO:0005148 ))
        n1(( category biolink:Drug ))
        n1-- biolink:treats -->n0
        """
    )

    # We should have valid permutations
    permutations = await find_valid_permutations(qg)
    assert len(list(permutations))

    plans = await generate_plans(
        qg,
        logger=logging.getLogger(),
    )
    assert len(plans) == 0
    assert_no_level(caplog, logging.WARNING, 1)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Disease <-biolink:treats- biolink:Drug
    """
))
@with_norm_overlay(settings.normalizer_url)
async def test_solve_reverse_edge(caplog):
    """
    Test that we can solve a simple query graph
    where we have to traverse an edge in the opposite
    direction of one that was given
    """

    qg = query_graph_from_string(
        """
        n0(( id MONDO:0005148 ))
        n1(( category biolink:Drug ))
        n1-- biolink:treats -->n0
        """
    )

    plans = await generate_plans(qg)
    assert len(plans) == 1
    assert_no_level(caplog, logging.WARNING)


ex1_kps = load_kps(cwd / "ex1_kps.json")


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, ex1_kps)
@with_norm_overlay(settings.normalizer_url)
async def test_valid_permute_ex1(caplog):
    """ Test first example is permuted correctly """
    with open(cwd / "ex1_qg.json", "r") as f:
        qg = json.load(f)

    plans = await find_valid_permutations(qg)

    assert plans

    # We should have one valid plan
    assert len(plans) == 1
    assert_no_level(caplog, logging.WARNING)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, ex1_kps)
@with_norm_overlay(settings.normalizer_url)
async def test_plan_ex1(caplog):
    """ Test that we get a good plan for our first example """
    with open(cwd / "ex1_qg.json", "r") as f:
        qg = json.load(f)

    plans = await generate_plans(qg)
    plan = plans[0]
    # One step per edge
    assert len(plan.keys()) == len(qg['edges'])

    # First step in the plan starts
    # at a pinned node
    step_1 = next(iter(plan.keys()))
    assert 'id' in qg['nodes'][step_1.source]
    assert_no_level(caplog, logging.WARNING)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Disease -biolink:treated_by-> biolink:Drug
    """
))
@with_norm_overlay(settings.normalizer_url)
async def test_invalid_two_pinned_nodes(caplog):
    """
    Test Pinned -> Unbound + Pinned
    This should be valid because we only care about
    a path from a pinned node to all unbound nodes.
    """

    qg = query_graph_from_string(
        """
        n0(( id MONDO:0005148 ))
        n1(( category biolink:Drug ))
        n0-- biolink:treated_by -->n1
        n2(( id MONDO:0011122 ))
        """
    )

    plans = await generate_plans(qg)
    assert len(plans) == 1
    assert_no_level(caplog, logging.WARNING)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Disease -biolink:treated_by-> biolink:Drug
    """
))
@with_norm_overlay(settings.normalizer_url)
async def test_unbound_unconnected_node(caplog):
    """
    Test Pinned -> Unbound + Unbound
    This should be invalid because there is no path
    to the unbound node
    """

    qg = query_graph_from_string(
        """
        n0(( id MONDO:0005148 ))
        n1(( category biolink:Drug ))
        n0-- biolink:treated_by -->n1
        n2(( category biolink:PhenotypicFeature ))
        """
    )

    plans = await generate_plans(qg)
    assert len(plans) == 0
    assert_no_level(caplog, logging.WARNING, 1)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Disease -biolink:treated_by-> biolink:Drug
    """
))
@with_norm_overlay(settings.normalizer_url)
async def test_invalid_two_disconnected_components(caplog):
    """ 
    Test Pinned -> Unbound + Pinned -> Unbound
    This should be invalid because there is no path from
    a pinned node to all unbound nodes.
    """
    qg = query_graph_from_string(
        """
        n0(( id MONDO:0005148 ))
        n1(( category biolink:Drug ))
        n0-- biolink:treated_by -->n1
        n2(( id MONDO:0011122 ))
        n3(( category biolink:Drug ))
        n2-- biolink:treated_by -->n3
        """
    )

    plans = await generate_plans(qg)
    assert len(plans) == 0
    assert_no_level(caplog, logging.WARNING, 1)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Disease -biolink:treated_by-> biolink:Drug
    """
))
@with_norm_overlay(settings.normalizer_url)
async def test_bad_norm(caplog):
    """
    Test that the pinned node "XXX:123" that the normalizer does not know
    is still handled correctly based on the provided category.
    """

    qg = {
        "nodes": {
            "n0": {
                "id": "XXX:123",
                "category": "biolink:Disease"
            },
            "n1": {
                "category": "biolink:Drug"
            }
        },
        "edges": {
            "e01": {
                "subject": "n0",
                "object": "n1",
                "predicate": "biolink:treated_by"
            }
        }
    }

    plans = await generate_plans(qg)
    assert any(
        (
            record.levelname == "WARNING"
            and record.message == "Normalizer knows nothing about XXX:123"
        )
        for record in caplog.records
    )

    assert len(plans) == 1


@pytest.mark.asyncio
@pytest.mark.longrun
@with_registry_overlay(settings.kpregistry_url, generate_kps(1_000))
@with_norm_overlay(settings.normalizer_url)
async def test_planning_performance_generic_qg():
    """
    Test our performance when planning a very generic query graph.

    This is a use case we hopefully don't encounter very much.
    """

    qg = query_graph_from_string(
        """
        n0(( id MONDO:0005737 ))
        n1(( category biolink:NamedThing ))
        n2(( category biolink:NamedThing ))
        n0-- biolink:related_to -->n1
        n1-- biolink:related_to -->n2
        """
    )
    await time_and_display(
        partial(generate_plans, qg, logger=logging.getLogger()),
        "generate plan for a generic query graph (1000 kps)",
    )


@pytest.mark.asyncio
@pytest.mark.longrun
@with_registry_overlay(settings.kpregistry_url, generate_kps(50_000))
@with_norm_overlay(settings.normalizer_url)
async def test_planning_performance_typical_example():
    """
    Test our performance when planning a more typical query graph
    (modeled after a COP) with a lot of KPs available.

    We should be able to do better on performance using our filtering methods.
    """

    with open(cwd / "ex2_qg.json", "r") as f:
        qg = json.load(f)

    async def testable_generate_plans():
        await generate_plans(qg, logger=logging.getLogger())

    await time_and_display(
        testable_generate_plans,
        "generate plan for a typical query graph (50k KPs)",
    )
