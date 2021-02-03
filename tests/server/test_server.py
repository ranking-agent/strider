"""Test Strider."""
from pathlib import Path
import asyncio
import itertools
import json
import os
import httpx
from fastapi.responses import Response

from reasoner_pydantic import Query, Message, QueryGraph
import pytest

from tests.helpers.logger import setup_logger
from tests.helpers.context import \
    with_translator_overlay, with_registry_overlay, \
    with_norm_overlay, with_response_overlay

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


def assert_no_warnings_trapi(resp):
    """
    Check for any errors or warnings in the log of a trapi response
    """
    invalid_levels = ['WARNING', 'ERROR', 'CRITICAL']
    for log in resp['logs']:
        if log['level'] in invalid_levels:
            raise Exception(f"Invalid log record: {log}")


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    {
        "ctd":
        """
            CHEBI:6801(( category biolink:ChemicalSubstance ))
            MONDO:0005148(( category biolink:Disease ))
            HP:0004324(( category biolink:PhenotypicFeature ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
            MONDO:0005148-- predicate biolink:has_phenotype -->HP:0004324
        """
    }
)
async def test_solve_ex1():
    """Test solving the ex1 query graph"""
    with open(cwd / "ex1_qg.json", "r") as f:
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
    # Ensure we have some results
    assert len(output['message']['results']) > 0
    # Ensure we have a knowledge graph with nodes and edges
    assert len(output['message']['knowledge_graph']['nodes']) > 0
    assert len(output['message']['knowledge_graph']['edges']) > 0
    assert_no_warnings_trapi(output)

    print("========================= RESULTS =========================")
    print(output['message']['results'])
    print(output['message']['knowledge_graph'])


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    {
        "ctd":
        """
            CHEBI:6801(( category biolink:ChemicalSubstance ))
            MONDO:0005148(( category biolink:Disease ))
            HP:0004324(( category biolink:PhenotypicFeature ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
            MONDO:0005148-- predicate biolink:has_phenotype -->HP:0004324
        """
    }
)
async def test_solve_missing_predicate():
    """Test solving the ex1 query graph, in which one of the predicates is missing. """
    with open(cwd / "ex1_qg.json", "r") as f:
        QGRAPH = json.load(f)

    del QGRAPH['edges']['e01']['predicate']

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(QGRAPH)
        )
    )

    # Run
    output = await sync_query(q)
    assert_no_warnings_trapi(output)


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    {
        "ctd":
        """
            CHEBI:6801(( category biolink:ChemicalSubstance ))
            MONDO:0005148(( category biolink:Disease ))
            HP:0004324(( category biolink:PhenotypicFeature ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
            MONDO:0005148-- predicate biolink:has_phenotype -->HP:0004324
        """
    }
)
async def test_solve_missing_category():
    """Test solving the ex1 query graph, in which one of the categories is missing. """
    with open(cwd / "ex1_qg.json", "r") as f:
        QGRAPH = json.load(f)

    del QGRAPH['nodes']['n0']['category']

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(QGRAPH)
        )
    )

    # Run
    output = await sync_query(q)


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    {
        "ctd":
        """
            CHEBI:6801(( category biolink:ChemicalSubstance ))
            MONDO:0005148(( category biolink:Disease ))
            HP:0004324(( category biolink:PhenotypicFeature ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
            MONDO:0005148-- predicate biolink:has_phenotype -->HP:0004324
        """
    }
)
async def test_log_level_param():
    """Test that changing the log level given to sync_query changes the output """
    with open(cwd / "ex1_qg.json", "r") as f:
        QGRAPH = json.load(f)

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(QGRAPH)
        )
    )

    # Check there are no debug logs
    output = await sync_query(q, 'INFO')
    assert not any(l['level'] == 'DEBUG' for l in output['logs'])

    # Check there are now debug logs
    output = await sync_query(q, 'DEBUG')
    assert any(l['level'] == 'DEBUG' for l in output['logs'])


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    {
        "ctd":
        """
            CHEBI:6801(( category biolink:ChemicalSubstance ))
            MONDO:0005148(( category biolink:Disease ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
        """,
        "hetio":
        """
            CHEBI:6801(( category biolink:ChemicalSubstance ))
            MONDO:0005148(( category biolink:Disease ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
        """,
        "mychem":
        """
            MONDO:0005148(( category biolink:Disease ))
            HP:0004324(( category biolink:PhenotypicFeature ))
            MONDO:0005148-- predicate biolink:has_phenotype -->HP:0004324
        """
    }
)
async def test_plan_ex1():
    """Test /plan endpoint"""
    with open(cwd / "ex1_qg.json", "r") as f:
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

    # Check that output is JSON serializeable
    json.dumps(output)

    plan = output[0]

    # Two steps in the plan each with KPs to contact
    assert len(plan['n0-e01->n1']) == 2
    assert len(plan['n1-e12->n2']) == 1


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    {
        "ctd":
        """
            CHEBI:6801(( category biolink:ChemicalSubstance ))
            MONDO:0005148(( category biolink:Disease ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
        """,
        "mychem":
        """
            MONDO:0005148(( category biolink:Disease ))
            HP:0004324(( category biolink:PhenotypicFeature ))
            MONDO:0005148-- predicate biolink:has_phenotype -->HP:0004324
        """,
        "hetio":
        """
            MONDO:0005148(( category biolink:Disease ))
            HP:0004324(( category biolink:PhenotypicFeature ))
            MONDO:0005148-- predicate biolink:has_phenotype -->HP:0004324
        """,
    }
)
# Override one KP with an invalid response
@with_response_overlay(
    "http://mychem/query",
    Response(
        status_code=500,
        content="Internal server error",
    )
)
async def test_kp_unavailable():
    """
    Test that when a KP is unavailable we add a message to
    the log but continue running
    """
    with open(cwd / "ex1_qg.json", "r") as f:
        QGRAPH = json.load(f)

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(QGRAPH)
        )
    )

    # Run
    output = await sync_query(q)

    # Check that we stored the error
    assert 'Error contacting KP' in output['logs'][0]['message']
    assert 'Internal server error' in output['logs'][0]['message']
    # Ensure we have results from the other KPs
    assert len(output['message']['knowledge_graph']['nodes']) > 0
    assert len(output['message']['knowledge_graph']['edges']) > 0
    assert len(output['message']['results']) > 0


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    {
        "ctd": """
        CHEBI:34253(( category biolink:ChemicalSubstance ))
        NCBIGene:xxx(( category biolink:Gene ))
        NCBIGene:yyy(( category biolink:Gene ))
        CHEBI:34253-- predicate biolink:interacts_with -->NCBIGene:xxx
        CHEBI:34253-- predicate biolink:directly_interacts_with -->NCBIGene:yyy
        """,
    },
)
async def test_predicate_fanout():
    """Test that all predicate descendants are explored."""
    qgraph = {
        "nodes": {
            "a": {
                "category": "biolink:ChemicalSubstance",
                "id": "CHEBI:34253"
            },
            "b": {
                "category": "biolink:Gene"
            }
        },
        "edges": {
            "ab": {
                "subject": "a",
                "object": "b",
                "predicate": "biolink:interacts_with"
            }
        }
    }

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(qgraph)
        )
    )

    # Run
    output = await sync_query(q, log_level="DEBUG")
    assert len(output["message"]["results"]) == 2


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    {
        "ctd": """
        CHEBI:34253(( category biolink:ChemicalSubstance ))
        NCBIGene:yyy(( category biolink:Gene ))
        CHEBI:34253-- predicate biolink:directly_interacts_with -->NCBIGene:yyy
        """,
    },
)
async def test_subpredicate():
    """Test that KPs are sent the correct predicate subclasses."""
    qgraph = {
        "nodes": {
            "a": {
                "category": "biolink:ChemicalSubstance",
                "id": "CHEBI:34253"
            },
            "b": {
                "category": "biolink:Gene"
            }
        },
        "edges": {
            "ab": {
                "subject": "a",
                "object": "b",
                "predicate": "biolink:interacts_with"
            }
        }
    }

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(qgraph)
        )
    )

    # Run
    output = await sync_query(q, log_level="DEBUG")
    assert output["message"]["results"]
