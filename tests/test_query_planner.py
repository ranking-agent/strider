from functools import partial
import logging
from strider.traversal import NoAnswersError

import pytest

from tests.helpers.context import \
    with_registry_overlay, with_norm_overlay
from tests.helpers.utils import generate_kps, \
    time_and_display, query_graph_from_string, \
    kps_from_string
from tests.helpers.logger import assert_no_level


from strider.query_planner import generate_plan

from strider.trapi import fill_categories_predicates

from strider.config import settings

# Switch prefix path before importing server
settings.kpregistry_url = "http://registry"
settings.normalizer_url = "http://normalizer"

LOGGER = logging.getLogger()


async def prepare_query_graph(query_graph):
    """ Prepare a query graph for the generate_plans method """
    await fill_categories_predicates(query_graph, logging.getLogger())


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
        n0(( categories[] biolink:ExposureEvent ))
        n1(( categories[] biolink:Drug ))
        n0-- biolink:related_to -->n1
        """
    )

    await prepare_query_graph(qg)

    with pytest.raises(NoAnswersError, match=r"cannot reach"):
        plan, kps = await generate_plan(
            qg,
            logger=logging.getLogger()
        )


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Drug biolink:treats biolink:Disease
    """
))
@with_norm_overlay(settings.normalizer_url, """
    MONDO:0005148 categories biolink:Disease
""")
async def test_plan_reverse_edge(caplog):
    """
    Test that we can plan a simple query graph
    where we have to traverse an edge in the opposite
    direction of one that was given
    """

    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n1(( categories[] biolink:Drug ))
        n1-- biolink:treats -->n0
        """
    )
    await prepare_query_graph(qg)

    plan, kps = await generate_plan(qg)
    assert plan == {"n1n0": ["kp0"]}

    assert "kp0" in kps


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp1 biolink:ChemicalSubstance biolink:treats biolink:Disease
    kp2 biolink:ChemicalSubstance biolink:treats biolink:PhenotypicFeature
    kp3 biolink:Disease biolink:has_phenotype biolink:PhenotypicFeature
    """
))
@with_norm_overlay(settings.normalizer_url)
async def test_plan_loop():
    """
    Test that we create a plan for a query with a loop
    """

    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0008114 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:PhenotypicFeature ))
        n2(( categories[] biolink:ChemicalSubstance ))
        n0-- biolink:has_phenotype -->n1
        n2-- biolink:treats -->n0
        n2-- biolink:treats -->n1
        """
    )
    await prepare_query_graph(qg)

    plan, _ = await generate_plan(qg)

    assert len(plan) == 3


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Disease biolink:related_to biolink:Disease
    """
))
@with_norm_overlay(settings.normalizer_url)
async def test_plan_reuse_pinned():
    """
    Test that we create a plan that uses a pinned node twice
    """

    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:Disease ))
        n2(( categories[] biolink:Disease ))
        n3(( categories[] biolink:Disease ))
        n0-- biolink:related_to -->n1
        n1-- biolink:related_to -->n2
        n2-- biolink:related_to -->n0
        n0-- biolink:related_to -->n3
        """
    )
    await prepare_query_graph(qg)

    plan, kps = await generate_plan(qg)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Disease biolink:related_to biolink:Disease
    """
))
@with_norm_overlay(settings.normalizer_url, """
    MONDO:0005148 categories biolink:Disease
""")
async def test_plan_double_loop(caplog):
    """
    Test valid plan for a more complex query with two loops
    """

    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:Disease ))
        n2(( categories[] biolink:Disease ))
        n3(( categories[] biolink:Disease ))
        n4(( categories[] biolink:Disease ))
        n0-- biolink:related_to -->n1
        n1-- biolink:related_to -->n2
        n2-- biolink:related_to -->n0
        n2-- biolink:related_to -->n3
        n3-- biolink:related_to -->n4
        n4-- biolink:related_to -->n2
        """
    )
    await prepare_query_graph(qg)

    plan, kps = await generate_plan(qg)


@pytest.mark.asyncio
@with_registry_overlay(
    settings.kpregistry_url,
    kps_from_string(
        """
        kp0 biolink:Disease biolink:treated_by biolink:Drug
        kp1 biolink:Drug biolink:affects biolink:Gene
        kp2 biolink:MolecularEntity biolink:decreases_abundance_of biolink:GeneOrGeneProduct
        kp3 biolink:Disease biolink:treated_by biolink:MolecularEntity
        """
    )
)
@with_norm_overlay(settings.normalizer_url, """
    MONDO:0005148 categories biolink:Disease
""")
async def test_plan_ex1(caplog):
    """ Test that we get a good plan for our first example """
    qg = query_graph_from_string(
        """
        n0(( categories[] biolink:MolecularEntity ))
        n1(( ids[] MONDO:0005148 ))
        n2(( categories[] biolink:GeneOrGeneProduct ))
        n1-- biolink:treated_by -->n0
        n0-- biolink:affects_abundance_of -->n2
        """
    )
    await prepare_query_graph(qg)

    plan, kps = await generate_plan(qg)
    # One step per edge
    assert len(plan) == len(qg['edges'])


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Disease biolink:treated_by biolink:Drug
    """
))
@with_norm_overlay(settings.normalizer_url, """
    MONDO:0005148 categories biolink:Disease
    MONDO:0011122 categories biolink:Disease
""")
async def test_valid_two_pinned_nodes(caplog):
    """
    Test Pinned -> Unbound + Pinned
    This should be valid because we only care about
    a path from a pinned node to all unbound nodes.
    """

    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n1(( categories[] biolink:Drug ))
        n0-- biolink:treated_by -->n1
        n2(( ids[] MONDO:0011122 ))
        """
    )
    await prepare_query_graph(qg)

    plan, kps = await generate_plan(qg)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Disease biolink:treated_by biolink:Drug
    kp1 biolink:Disease biolink:has_phenotype biolink:PhenotypicFeature
    """
))
@with_norm_overlay(settings.normalizer_url, """
    MONDO:0005148 categories biolink:Disease
""")
async def test_fork(caplog):
    """
    Test Unbound <- Pinned -> Unbound

    This should be valid because we allow
    a fork to multiple paths.
    """

    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n1(( categories[] biolink:Drug ))
        n2(( categories[] biolink:PhenotypicFeature ))
        n0-- biolink:treated_by -->n1
        n0-- biolink:has_phenotype -->n2
        """
    )
    await prepare_query_graph(qg)

    plan, kps = await generate_plan(qg)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Disease biolink:treated_by biolink:Drug
    """
))
@with_norm_overlay(settings.normalizer_url, """
    MONDO:0005148 categories biolink:Disease
""")
async def test_unbound_unconnected_node(caplog):
    """
    Test Pinned -> Unbound + Unbound
    This should be invalid because there is no path
    to the unbound node
    """

    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n1(( categories[] biolink:Drug ))
        n0-- biolink:treated_by -->n1
        n2(( categories[] biolink:PhenotypicFeature ))
        """
    )
    await prepare_query_graph(qg)

    with pytest.raises(NoAnswersError, match=r"cannot reach"):
        plan, kps = await generate_plan(qg)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Disease biolink:treated_by biolink:Drug
    """
))
@with_norm_overlay(settings.normalizer_url, """
    MONDO:0005148 categories biolink:Disease
    MONDO:0011122 categories biolink:Disease
""")
async def test_valid_two_disconnected_components(caplog):
    """
    Test Pinned -> Unbound + Pinned -> Unbound
    This should be valid because there is a path from
    a pinned node to all unbound nodes.
    """
    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n1(( categories[] biolink:Drug ))
        n0-- biolink:treated_by -->n1
        n2(( ids[] MONDO:0011122 ))
        n3(( categories[] biolink:Drug ))
        n2-- biolink:treated_by -->n3
        """
    )
    await prepare_query_graph(qg)

    plan, kps = await generate_plan(qg)


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Disease biolink:treated_by biolink:Drug
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
                "ids": ["XXX:123"],
                "categories": ["biolink:Disease"],
            },
            "n1": {
                "categories": ["biolink:Drug"],
            }
        },
        "edges": {
            "e01": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:treated_by"],
            }
        }
    }
    await prepare_query_graph(qg)

    plan, kps = await generate_plan(qg)
    assert any(
        (
            record.levelname == "WARNING"
            and record.message == "Normalizer knows nothing about XXX:123"
        )
        for record in caplog.records
    )


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:ChemicalSubstance biolink:treats biolink:Disease
    """
))
@with_norm_overlay(settings.normalizer_url)
async def test_descendant_reverse_category(caplog):
    """
    Test that when we are given related_to that descendants
    will be filled either in the forward or backwards direction
    """
    valid_qg = query_graph_from_string(
        """
        n0(( categories[] biolink:Disease ))
        n0(( ids[] MONDO:0005737 ))
        n0-- biolink:related_to -->n1
        n1(( categories[] biolink:ChemicalSubstance ))
        """
    )
    await prepare_query_graph(valid_qg)
    plan, kps = await generate_plan(valid_qg)
    assert_no_level(caplog, logging.WARNING, 1)

    invalid_qg = query_graph_from_string(
        """
        n0(( categories[] biolink:Disease ))
        n0(( ids[] MONDO:0005737 ))
        n0-- biolink:treats -->n1
        n1(( categories[] biolink:ChemicalSubstance ))
        """
    )
    await prepare_query_graph(invalid_qg)

    with pytest.raises(NoAnswersError, match=r"No KPs"):
        plan, kps = await generate_plan(invalid_qg)


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
        n0(( ids[] MONDO:0005737 ))
        n1(( categories[] biolink:NamedThing ))
        n2(( categories[] biolink:NamedThing ))
        n0-- biolink:related_to -->n1
        n1-- biolink:related_to -->n2
        """
    )
    await prepare_query_graph(qg)
    await time_and_display(
        partial(generate_plan, qg, logger=logging.getLogger()),
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

    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005737 ))
        n1(( categories[] biolink:BiologicalProcessOrActivity ))
        n2(( categories[] biolink:AnatomicalEntity ))
        n3(( categories[] biolink:PhenotypicFeature ))
        n4(( categories[] biolink:PhenotypicFeature ))
        n5(( ids[] MONDO:6801 ))
        n0-- biolink:related_to -->n1
        n1-- biolink:related_to -->n2
        n2-- biolink:related_to -->n3
        n3-- biolink:related_to -->n4
        n4-- biolink:related_to -->n5
        """
    )
    await prepare_query_graph(qg)

    async def testable_generate_plans():
        await generate_plan(qg, logger=logging.getLogger())

    await time_and_display(
        testable_generate_plans,
        "generate plan for a typical query graph (50k KPs)",
    )


@pytest.mark.asyncio
@with_registry_overlay(settings.kpregistry_url, kps_from_string(
    """
    kp0 biolink:Disease biolink:treated_by biolink:Drug
    """
))
@with_norm_overlay(settings.normalizer_url)
async def test_double_sided(caplog):
    """
    Test planning when a KP provides edges in both directions.
    """

    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005737 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:Drug ))
        n0-- biolink:treated_by -->n1
        """
    )
    await prepare_query_graph(qg)
    plan, kps = await generate_plan(qg, logger=logging.getLogger())
    assert plan == {"n0n1": ["kp0"]}
    assert "kp0" in kps
