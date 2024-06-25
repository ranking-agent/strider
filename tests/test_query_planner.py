import logging
import pytest
from pytest_httpx import HTTPXMock
import redis.asyncio

from tests.helpers.utils import (
    query_graph_from_string,
    get_normalizer_response,
)
from tests.helpers.redisMock import redisMock

from strider.traversal import NoAnswersError
from strider.query_planner import generate_plan, get_next_qedge
from strider.trapi import fill_categories_predicates
from strider.config import settings

LOGGER = logging.getLogger()

# Modify settings before importing things
settings.normalizer_url = "http://normalizer"


async def prepare_query_graph(query_graph):
    """Prepare a query graph for the generate_plans method"""
    await fill_categories_predicates(query_graph, logging.getLogger())


@pytest.mark.asyncio
async def test_not_enough_kps(monkeypatch):
    """
    Check we get no plans when we submit a query graph
    that has edges we can't solve
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = query_graph_from_string(
        """
        n0(( categories[] biolink:ExposureEvent ))
        n1(( categories[] biolink:Drug ))
        n0-- biolink:related_to -->n1
        """
    )

    with pytest.raises(NoAnswersError, match=r"cannot reach"):
        plan, kps = await generate_plan(qg, {}, logger=logging.getLogger())


@pytest.mark.asyncio
async def test_plan_reverse_edge(monkeypatch):
    """
    Test that we can plan a simple query graph
    where we have to traverse an edge in the opposite
    direction of one that was given
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:Drug ))
        n1-- biolink:treats -->n0
        """
    )

    plan, kps = await generate_plan(qg, {})
    assert plan == {"n1n0": ["infores:kp1"]}

    assert "infores:kp1" in kps


@pytest.mark.asyncio
async def test_plan_loop(monkeypatch):
    """
    Test that we create a plan for a query with a loop
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
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

    plan, _ = await generate_plan(qg, {})

    assert len(plan) == 3


@pytest.mark.asyncio
async def test_plan_reuse_pinned(monkeypatch):
    """
    Test that we create a plan that uses a pinned node twice
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
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

    plan, kps = await generate_plan(qg, {})


@pytest.mark.asyncio
async def test_plan_double_loop(monkeypatch):
    """
    Test valid plan for a more complex query with two loops
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
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
    plan, kps = await generate_plan(qg, {})


@pytest.mark.asyncio
async def test_valid_two_pinned_nodes(monkeypatch):
    """
    Test Pinned -> Unbound + Pinned
    This should be valid because we only care about
    a path from a pinned node to all unbound nodes.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:Drug ))
        n0-- biolink:treated_by -->n1
        n2(( ids[] MONDO:0011122 ))
        n2(( categories[] biolink:Disease ))
        """
    )
    await prepare_query_graph(qg)

    plan, kps = await generate_plan(qg, {})


@pytest.mark.asyncio
async def test_fork(monkeypatch):
    """
    Test Unbound <- Pinned -> Unbound

    This should be valid because we allow
    a fork to multiple paths.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:Drug ))
        n2(( categories[] biolink:PhenotypicFeature ))
        n0-- biolink:treated_by -->n1
        n0-- biolink:has_phenotype -->n2
        """
    )

    plan, kps = await generate_plan(qg, {})


@pytest.mark.asyncio
async def test_unbound_unconnected_node(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test Pinned -> Unbound + Unbound
    This should be invalid because there is no path
    to the unbound node
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(
        url="http://normalizer/get_normalized_nodes",
        json=get_normalizer_response("""
            MONDO:0005148 categories biolink:Disease
        """)
    )
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
        plan, kps = await generate_plan(qg, {})


@pytest.mark.asyncio
async def test_valid_two_disconnected_components(monkeypatch):
    """
    Test Pinned -> Unbound + Pinned -> Unbound
    This should be valid because there is a path from
    a pinned node to all unbound nodes.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005148 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:Drug ))
        n0-- biolink:treated_by -->n1
        n2(( ids[] MONDO:0011122 ))
        n2(( categories[] biolink:Disease ))
        n3(( categories[] biolink:Drug ))
        n2-- biolink:treated_by -->n3
        """
    )
    plan, kps = await generate_plan(qg, {})
    assert plan == {"n0n1": ["infores:kp1"], "n2n3": ["infores:kp1"]}


@pytest.mark.asyncio
async def test_bad_norm(monkeypatch):
    """
    Test that the pinned node "XXX:123" that the normalizer does not know
    is still handled correctly based on the provided category.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = {
        "nodes": {
            "n0": {
                "ids": ["XXX:123"],
                "categories": ["biolink:Disease"],
            },
            "n1": {
                "categories": ["biolink:Drug"],
            },
        },
        "edges": {
            "e01": {
                "subject": "n0",
                "object": "n1",
                "predicates": ["biolink:treated_by"],
            }
        },
    }

    plan, kps = await generate_plan(qg, {})
    assert plan == {"e01": ["infores:kp1"]}


@pytest.mark.asyncio
async def test_double_sided(monkeypatch):
    """
    Test planning when a KP provides edges in both directions.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:0005737 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:Drug ))
        n0-- biolink:treated_by -->n1
        """
    )
    plan, kps = await generate_plan(qg, {}, logger=logging.getLogger())
    assert plan == {"n0n1": ["infores:kp1"]}
    assert "infores:kp1" in kps


def test_get_next_qedge():
    """Test get_next_qedge()."""
    qgraph = {
        "nodes": {
            "n0": {"ids": ["01", "02"]},
            "n1": {},
            "n2": {"ids": ["03"]},
        },
        "edges": {
            "e01": {
                "subject": "n0",
                "object": "n1",
            },
            "e12": {
                "subject": "n1",
                "object": "n2",
            },
        },
    }
    qedge_id, _ = get_next_qedge(qgraph)
    assert qedge_id == "e12"


def test_get_next_qedge_with_self_edge():
    """Test get_next_qedge() with a self edge."""
    qgraph = {
        "nodes": {
            "n0": {"ids": ["01", "02"]},
            "n1": {},
        },
        "edges": {
            "e01": {
                "subject": "n0",
                "object": "n1",
            },
            "e00": {
                "subject": "n0",
                "object": "n0",
            },
        },
    }
    qedge_id, _ = get_next_qedge(qgraph)
    assert qedge_id == "e00"


def test_get_next_qedge_multi_edges():
    """Test get_next_qedge() with multiple edges between two nodes."""
    qgraph = {
        "nodes": {
            "n0": {"ids": ["01", "02"]},
            "n1": {},
            "n2": {"ids": ["03"]},
        },
        "edges": {
            "e01": {
                "subject": "n0",
                "object": "n1",
            },
            "e12": {
                "subject": "n1",
                "object": "n2",
            },
            "e012": {
                "subject": "n0",
                "object": "n1",
            },
        },
    }
    qedge_id, _ = get_next_qedge(qgraph)
    assert qedge_id == "e01"


@pytest.mark.asyncio
async def test_predicate_fanout(monkeypatch):
    """Test that all predicate descendants are explored."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = {
        "nodes": {
            "a": {"categories": ["biolink:ChemicalSubstance"], "ids": ["CHEBI:34253"]},
            "b": {"categories": ["biolink:Gene"]},
        },
        "edges": {
            "ab": {
                "subject": "a",
                "object": "b",
                "predicates": ["biolink:affects"],
            }
        },
    }

    plan, kps = await generate_plan(qg, {}, logger=logging.getLogger())
    assert plan == {"ab": ["infores:kp2"]}
    assert "infores:kp2" in kps


@pytest.mark.asyncio
async def test_inverse_predicate(monkeypatch):
    """
    Test solving a query graph where we have to look up
    the inverse of a given predicate to get the right answer.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = query_graph_from_string(
        """
        n0(( categories[] biolink:Disease ))
        n1(( ids[] CHEBI:6801 ))
        n1(( categories[] biolink:Drug ))
        n0-- biolink:treated_by -->n1
        """
    )

    plan, kps = await generate_plan(qg, {}, logger=logging.getLogger())
    assert plan == {"n0n1": ["infores:kp1"]}
    assert "infores:kp1" in kps


@pytest.mark.asyncio
async def test_symmetric_predicate(monkeypatch):
    """
    Test that we get a kp in the plan with reverse categories and a symmetric predicate.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = query_graph_from_string(
        """
        n0(( categories[] biolink:Disease ))
        n0(( ids[] MONDO:0005148 ))
        n1(( categories[] biolink:Drug ))
        n1-- biolink:correlated_with -->n0
        """
    )
    plan, kps = await generate_plan(qg, {}, logger=logging.getLogger())
    assert plan == {"n1n0": ["infores:kp1"]}
    assert "infores:kp1" in kps


@pytest.mark.asyncio
async def test_subpredicate(monkeypatch):
    """Test that KPs are sent the correct predicate subclasses."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = {
        "nodes": {
            "a": {"categories": ["biolink:ChemicalSubstance"], "ids": ["CHEBI:34253"]},
            "b": {"categories": ["biolink:Disease"]},
        },
        "edges": {
            "ab": {
                "subject": "a",
                "object": "b",
                "predicates": ["biolink:interacts_with"],
            }
        },
    }

    plan, kps = await generate_plan(qg, {}, logger=logging.getLogger())
    assert plan == {"ab": ["infores:kp1"]}
    assert "infores:kp1" in kps


@pytest.mark.asyncio
async def test_solve_double_subclass(monkeypatch):
    """
    Test that when given a node with a general type that we subclass
    it and contact all KPs available for information about that node
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:1 ))
        n0(( categories[] biolink:NamedThing ))
        n1(( categories[] biolink:NamedThing ))
        n0-- biolink:ameliorates -->n1
        """
    )

    plan, kps = await generate_plan(qg, {}, logger=logging.getLogger())
    assert plan == {"n0n1": ["infores:kp1"]}
    assert "infores:kp1" in kps


@pytest.mark.asyncio
async def test_pinned_to_pinned(monkeypatch):
    """
    Test that we can solve a query to check if a pinned node is
    connected to another pinned node
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = query_graph_from_string(
        """
        n0(( ids[] MONDO:1 ))
        n0(( categories[] biolink:Disease ))
        n1(( ids[] CHEBI:1 ))
        n1(( categories[] biolink:Vitamin ))
        n0-- biolink:related_to -->n1
        """
    )

    plan, kps = await generate_plan(qg, {}, logger=logging.getLogger())
    assert plan == {"n0n1": ["infores:kp3"]}
    assert "infores:kp3" in kps


@pytest.mark.asyncio
async def test_self_edge(monkeypatch):
    """
    Test that we can solve a query with a self-edge
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    qg = query_graph_from_string(
        """
        n0(( ids[] CHEBI:1 ))
        n0(( categories[] biolink:Gene ))
        n0-- biolink:related_to -->n0
        """
    )

    # await prepare_query_graph(qg)
    plan, kps = await generate_plan(qg, {}, logger=logging.getLogger())
    assert plan == {"n0n0": ["infores:kp2"]}
    assert "infores:kp2" in kps
