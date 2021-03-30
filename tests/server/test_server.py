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
from tests.helpers.utils import query_graph_from_string, validate_message

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
    validate_message(
        {
            "knowledge_graph":
                """
                CHEBI:6801 biolink:treats MONDO:0005148
                MONDO:0005148 biolink:has_phenotype HP:0004324
                """,
            "results": [
                """
                node_bindings:
                    n0 CHEBI:6801
                    n1 MONDO:0005148
                    n2 HP:0004324
                edge_bindings:
                    e01 CHEBI:6801-MONDO:0005148
                    e12 MONDO:0005148-HP:0004324
                """
            ],
        },
        output["message"]
    )


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    {
        "ctd":
        """
            CHEBI:6801(( category biolink:ChemicalSubstance ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
            MONDO:0005148(( category biolink:DiseaseOrPhenotypicFeature ))
            MONDO:0005148(( category biolink:Disease ))
        """
    }
)
async def test_duplicate_results():
    """
    Some KPs will advertise multiple operations from the biolink hierarchy.

    Test that we filter out duplicate results if we
    contact the KP multiple times.
    """
    QGRAPH = query_graph_from_string(
        """
        n0(( id CHEBI:6801 ))
        n0(( category biolink:ChemicalSubstance ))
        n1(( category biolink:DiseaseOrPhenotypicFeature ))
        n0-- biolink:treats -->n1
        """
    )

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(QGRAPH)
        )
    )

    # Run
    output = await sync_query(q)
    assert_no_warnings_trapi(output)

    assert len(output['message']['results']) == 1


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    kp_data={
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
    kp_data={
        "ctd":
        """
            CHEBI:6801(( category biolink:Drug ))
            MONDO:0005148(( category biolink:Disease ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
        """
    },
    normalizer_data="""
        CHEBI:6801 categories biolink:Drug
        """
)
async def test_normalizer_different_category():
    """
    Test solving a query graph where the category provided doesn't match
    the one in the node normalizer.
    """

    QGRAPH = query_graph_from_string(
        """
        n0(( category biolink:ChemicalSubstance ))
        n0(( id CHEBI:6801 ))
        n1(( category biolink:Disease ))
        n0-- biolink:treats -->n1
        """
    )

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
    kp_data={
        "kp0":
        """
            MONDO:0008114(( category biolink:Disease ))
            MONDO:0008114-- predicate biolink:has_phenotype -->HP:0007430
            HP:0007430<-- predicate biolink:has_phenotype --MONDO:0008114
            HP:0007430(( category biolink:PhenotypicFeature ))
        """,
        "kp1":
        """
            MESH:C035133(( category biolink:ChemicalSubstance ))
            MESH:C035133-- predicate biolink:treats -->HP:0007430
            HP:0007430<-- predicate biolink:treats --MESH:C035133
            HP:0007430(( category biolink:PhenotypicFeature ))
        """,
        "kp2":
        """
            MESH:C035133(( category biolink:ChemicalSubstance ))
            MESH:C035133-- predicate biolink:treats -->MONDO:0008114
            MONDO:0008114<-- predicate biolink:treats --MESH:C035133
            MONDO:0008114(( category biolink:Disease ))
        """,
    },
    normalizer_data="""
        MONDO:0008114 categories biolink:Disease
        MESH:C035133 categories biolink:ChemicalSubstance
        HP:0007430 categories biolink:PhenotypicFeature
        """
)
async def test_solve_loop(caplog):
    """
    Test that we correctly solve a query with a loop
    """

    QGRAPH = query_graph_from_string(
        """
        n0(( id MONDO:0008114 ))
        n0(( category biolink:Disease ))
        n1(( category biolink:PhenotypicFeature ))
        n2(( category biolink:ChemicalSubstance ))
        n0-- biolink:has_phenotype -->n1
        n2-- biolink:treats -->n0
        n2-- biolink:treats -->n1
        """
    )

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(QGRAPH)
        )
    )
    # Run
    output = await sync_query(q)

    # Ensure we have some results
    assert len(output['message']['results']) > 0

    # Ensure we have a knowledge graph with data that matches
    # the query graph
    assert len(output['message']['knowledge_graph']['nodes']) == 3
    assert len(output['message']['knowledge_graph']['edges']) == 3
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
    assert len(plan['n0-e01-n1']) == 2
    assert len(plan['n1-e12-n2']) == 1


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


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    {
        "mychem":
        """
            MONDO:0005148(( category biolink:Disease ))
            HP:xxx(( category biolink:PhenotypicFeature ))
            MONDO:0005148-- predicate biolink:related_to -->HP:xxx
        """,
        "hetio":
        """
            MONDO:0005148(( category biolink:Disease ))
            MONDO:0005148-- predicate biolink:has_phenotype -->HP:yyy
            HP:yyy(( category biolink:PhenotypicFeature ))
        """,
    }
)
async def test_mutability_bug():
    """
    Test that qgraph is not mutated between KP calls.
    """
    qg = {
        'nodes': {
            'n0': {
                'id': ['MONDO:0005148'],
                'category': ["biolink:Disease"],
            },
            'n1': {
                'category': ['biolink:PhenotypicFeature'],
            },
        },
        'edges': {
            'n0n1': {
                'subject': 'n0',
                'object': 'n1',
                'predicate': ['biolink:related_to'],
            },
        }
    }

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(qg)
        )
    )

    # Run
    output = await sync_query(q)
    assert len(output["message"]["results"]) == 2


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    kp_data={
        "ctd":
        """
            CHEBI:6801(( category biolink:Drug ))
            MONDO:0005148(( category biolink:Disease ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
        """
    },
    normalizer_data="""
        CHEBI:6801 categories biolink:Drug
        MONDO:0005148 categories biolink:Disease
        """
)
async def test_inverse_predicate():
    """
    Test solving a query graph where we have to look up
    the inverse of a given predicate to get the right answer.
    """

    QGRAPH = query_graph_from_string(
        """
        n0(( category biolink:Disease ))
        n1(( id CHEBI:6801 ))
        n0-- biolink:treated_by -->n1
        """
    )

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(QGRAPH)
        )
    )

    # Run
    output = await sync_query(q)

    validate_message({
        "knowledge_graph":
            """
            CHEBI:6801 biolink:treats MONDO:0005148
            """,
        "results": [
            """
            node_bindings:
                n0 MONDO:0005148
                n1 CHEBI:6801
            edge_bindings:
                n0n1 CHEBI:6801-MONDO:0005148
            """
        ]
    },
        output["message"])


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    kp_data={
        "ctd":
        """
            CHEBI:6801(( category biolink:Drug ))
            MONDO:0005148(( category biolink:Disease ))
            MONDO:0005148-- predicate biolink:correlated_with -->CHEBI:6801
        """
    },
    normalizer_data="""
        CHEBI:6801 categories biolink:Drug
        MONDO:0005148 categories biolink:Disease
        """
)
async def test_symmetric_predicate():
    """
    Test solving a query graph where we have a symmetric predicate
    that we have to look up in reverse.
    """

    QGRAPH = query_graph_from_string(
        """
        n0(( category biolink:Disease ))
        n0(( id MONDO:0005148 ))
        n1(( category biolink:Drug ))
        n1-- biolink:correlated_with -->n0
        """
    )

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(QGRAPH)
        )
    )

    # Run
    output = await sync_query(q)
    validate_message({
        "knowledge_graph":
            """
            CHEBI:6801 biolink:correlated_with MONDO:0005148
            """,
        "results": [
            """
            node_bindings:
                n0 MONDO:0005148
                n1 CHEBI:6801
            edge_bindings:
                n0n1 CHEBI:6801-MONDO:0005148
            """
        ]}, output["message"]
    )


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    kp_data={
        "automat_kegg":
        """
            NCBIGene:2710(( category biolink:Gene ))
            CHEBI:15422(( category biolink:BiologicalEntity ))
            CHEBI:17754(( category biolink:BiologicalEntity ))
            NCBIGene:2710-- predicate biolink:increases_degradation_of -->CHEBI:15422
            NCBIGene:2710-- predicate biolink:increases_degradation_of -->CHEBI:17754
        """
    },
    normalizer_data="""
        UniProtKB:P32189 synonyms NCBIGene:2710
        UniProtKB:P32189 categories biolink:Gene
        """
)
async def test_issue_102():
    """
    Test solving a query graph where the category provided doesn't match
    the one in the node normalizer.
    """

    QGRAPH = query_graph_from_string(
        """
        a(( category biolink:ChemicalSubstance ))
        b(( id UniProtKB:P32189 ))
        a-- biolink:related_to -->b
        """
    )

    # Create query
    q = Query(
        message=Message(
            query_graph=QueryGraph.parse_obj(QGRAPH)
        )
    )

    # Run
    output = await sync_query(q)

    validate_message({
        "knowledge_graph":
            """
            NCBIGene:2710 biolink:increases_degradation_of CHEBI:17754
            NCBIGene:2710 biolink:increases_degradation_of CHEBI:15422
            """,
        "results": [
            """
            node_bindings:
                a CHEBI:17754
                b NCBIGene:2710
            edge_bindings:
                ab NCBIGene:2710-CHEBI:17754
            """,
            """
            node_bindings:
                a CHEBI:15422
                b NCBIGene:2710
            edge_bindings:
                ab NCBIGene:2710-CHEBI:15422
            """
        ]
    },
        output["message"]
    )
