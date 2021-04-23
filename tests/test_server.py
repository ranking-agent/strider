"""Test Strider."""
import json
from pathlib import Path

from fastapi.responses import Response
import httpx
import pytest
from reasoner_pydantic import Query, Message, QueryGraph

from tests.helpers.context import \
    response_overlay, with_translator_overlay, with_registry_overlay, \
    with_norm_overlay, with_response_overlay
from tests.helpers.logger import setup_logger
from tests.helpers.utils import query_graph_from_string, validate_message

from strider.config import settings

# Switch prefix path before importing server
settings.kpregistry_url = "http://registry"
settings.normalizer_url = "http://normalizer"
settings.redis_url = "redis://fakeredis:6379/0"

from strider.server import APP

client = httpx.AsyncClient(app=APP, base_url="http://test")

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
    QGRAPH = query_graph_from_string(
        """
        n0(( id CHEBI:6801 ))
        n0(( category biolink:ChemicalSubstance ))
        n1(( category biolink:Disease ))
        n2(( category biolink:PhenotypicFeature ))
        n0-- biolink:treats -->n1
        n1-- biolink:has_phenotype -->n2
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

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
                    n0n1 CHEBI:6801-MONDO:0005148
                    n1n2 MONDO:0005148-HP:0004324
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
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

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
    """Test solving a query graph, in which one of the predicates is missing. """
    QGRAPH = query_graph_from_string(
        """
        n0(( id CHEBI:6801 ))
        n0(( category biolink:ChemicalSubstance ))
        n1(( category biolink:Disease ))
        n2(( category biolink:PhenotypicFeature ))
        n0-- biolink:treats -->n1
        n1-- biolink:has_phenotype -->n2
        """
    )

    del QGRAPH['edges']['n0n1']['predicate']

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    assert output["message"]["results"]


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
    QGRAPH = query_graph_from_string(
        """
        n0(( id CHEBI:6801 ))
        n0(( category biolink:ChemicalSubstance ))
        n1(( category biolink:Disease ))
        n2(( category biolink:PhenotypicFeature ))
        n0-- biolink:treats -->n1
        n1-- biolink:has_phenotype -->n2
        """
    )

    del QGRAPH['nodes']['n0']['category']

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()


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
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    assert output["message"]["results"]


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    kp_data={
        "kp0":
        """
            MONDO:0008114(( category biolink:Disease ))
            MONDO:0008114-- predicate biolink:treated_by -->MESH:C035133
            MESH:C035133(( category biolink:ChemicalSubstance ))
        """,
        "kp1":
        """
            MESH:C035133(( category biolink:ChemicalSubstance ))
            MESH:C035133-- predicate biolink:ameliorates -->HP:0007430
            HP:0007430(( category biolink:PhenotypicFeature ))
        """,
        "kp2":
        """
            HP:0007430(( category biolink:PhenotypicFeature ))
            HP:0007430-- predicate biolink:has_phenotype -->MONDO:0008114
            MONDO:0008114(( category biolink:Disease ))
        """,
    },
    normalizer_data="""
        MONDO:0008114 categories biolink:Disease
        HP:0007430 categories biolink:PhenotypicFeature
        MESH:C035133 categories biolink:ChemicalSubstance
        """
)
async def test_solve_loop(caplog):
    """
    Test that we correctly solve a query with a loop
    """

    # TODO replace has_phenotype with the correct predicate phenotype_of
    # when BMT is updated to recognize it as a valid predicate

    QGRAPH = query_graph_from_string(
        """
        n0(( id MONDO:0008114 ))
        n1(( category biolink:ChemicalSubstance ))
        n2(( category biolink:PhenotypicFeature ))
        n0-- biolink:related_to -->n1
        n1-- biolink:related_to -->n2
        n2-- biolink:related_to -->n0
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    validate_message({
        "knowledge_graph":
            """
            MONDO:0008114 biolink:treated_by MESH:C035133
            MESH:C035133 biolink:ameliorates HP:0007430
            HP:0007430 biolink:has_phenotype MONDO:0008114
            """,
        "results": [
            """
            node_bindings:
                n0 MONDO:0008114
                n1 MESH:C035133
                n2 HP:0007430
            edge_bindings:
                n0n1 MONDO:0008114-MESH:C035133
                n1n2 MESH:C035133-HP:0007430
                n2n0 HP:0007430-MONDO:0008114
            """
        ],
    },
        output["message"])


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
    """Test that changing the log level changes the output """
    QGRAPH = query_graph_from_string(
        """
        n0(( id CHEBI:6801 ))
        n0(( category biolink:ChemicalSubstance ))
        n1(( category biolink:Disease ))
        n2(( category biolink:PhenotypicFeature ))
        n0-- biolink:treats -->n1
        n1-- biolink:has_phenotype -->n2
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Check there are no debug logs
    response = await client.post("/query", json=q)
    output = response.json()
    assert not any(l['level'] == 'DEBUG' for l in output['logs'])

    q["log_level"] = "DEBUG"

    # Check there are now debug logs
    response = await client.post("/query", json=q)
    output = response.json()
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
async def test_plan_endpoint():
    """Test /plan endpoint"""
    QGRAPH = query_graph_from_string(
        """
        n0(( id CHEBI:6801 ))
        n0(( category biolink:ChemicalSubstance ))
        n1(( category biolink:Disease ))
        n2(( category biolink:PhenotypicFeature ))
        n0-- biolink:treats -->n1
        n1-- biolink:has_phenotype -->n2
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/plan", json=q)
    output = response.json()

    plan = output[0]

    # Two steps in the plan each with KPs to contact
    assert len(plan['n0-n0n1-n1']) == 2
    assert len(plan['n1-n1n2-n2']) == 1


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
async def test_kp_500():
    """
    Test that when a KP returns a 500 error we add
    a message to the log but continue running
    """
    QGRAPH = query_graph_from_string(
        """
        n0(( id CHEBI:6801 ))
        n0(( category biolink:ChemicalSubstance ))
        n1(( category biolink:Disease ))
        n2(( category biolink:PhenotypicFeature ))
        n0-- biolink:treats -->n1
        n1-- biolink:has_phenotype -->n2
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    # Check that we stored the error
    assert 'Error contacting KP' in output['logs'][0]['message']
    assert 'Internal server error' in output['logs'][0]['message']
    # Ensure we have results from the other KPs
    assert len(output['message']['knowledge_graph']['nodes']) > 0
    assert len(output['message']['knowledge_graph']['edges']) > 0
    assert len(output['message']['results']) > 0


@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_registry_overlay(
    settings.kpregistry_url, {
        'ctd': {
            'url':
            'http://ctd/query',
            'operations': [{
                'source_type': 'biolink:ChemicalSubstance',
                'edge_type': '-biolink:treats->',
                'target_type': 'biolink:Disease'
            }],
            'details': {
                'preferred_prefixes': {}
            }
        }
    })
async def test_kp_unavailable():
    """
    Test that when a KP is unavailable we add a message to
    the log but continue running
    """
    QGRAPH = query_graph_from_string(
        """
        n0(( category biolink:ChemicalSubstance ))
        n0(( id CHEBI:6801 ))
        n0-- biolink:treats -->n1
        n1(( category biolink:Disease ))
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    # Check that we stored the error
    assert 'RequestError contacting KP' in output['logs'][0]['message']

@pytest.mark.asyncio
@with_norm_overlay(settings.normalizer_url)
@with_registry_overlay(
    settings.kpregistry_url, {
        'ctd': {
            'url':
            'http://ctd/query',
            'operations': [{
                'source_type': 'biolink:ChemicalSubstance',
                'edge_type': '-biolink:treats->',
                'target_type': 'biolink:Disease'
            }],
            'details': {
                'preferred_prefixes': {}
            }
        }
    })
@with_response_overlay(
    "http://ctd/query",
    Response(
        status_code=200,
        content=json.dumps({"message": None}),
    )
)
async def test_kp_not_trapi():
    """
    Test that when a KP is unavailable we add a message to
    the log but continue running
    """
    QGRAPH = query_graph_from_string(
        """
        n0(( category biolink:ChemicalSubstance ))
        n0(( id CHEBI:6801 ))
        n0-- biolink:treats -->n1
        n1(( category biolink:Disease ))
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    # Check that we stored the error
    assert 'Received non-TRAPI compliant response from KP' in output['logs'][0]['message']


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
    QGRAPH = {
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
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()
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
    QGRAPH = {
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
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()
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
    QGRAPH = {
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
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()
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
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    validate_message(
        {
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
        output["message"],
    )


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
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()
    validate_message({
        "knowledge_graph":
            """
            MONDO:0005148 biolink:correlated_with CHEBI:6801
            """,
        "results": [
            """
            node_bindings:
                n0 MONDO:0005148
                n1 CHEBI:6801
            edge_bindings:
                n1n0 MONDO:0005148-CHEBI:6801
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
            CHEBI:15422(( category biolink:ChemicalSubstance ))
            CHEBI:17754(( category biolink:ChemicalSubstance ))
            NCBIGene:2710-- predicate biolink:increases_degradation_of -->CHEBI:15422
            NCBIGene:2710-- predicate biolink:increases_degradation_of -->CHEBI:17754
        """
    },
    normalizer_data="""
        UniProtKB:P32189 categories biolink:Gene
        UniProtKB:P32189 synonyms NCBIGene:2710
        NCBIGene:2710 categories biolink:Gene
        NCBIGene:2710 synonyms UniProtKB:P32189
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
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

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


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    kp_data={
        "kp0":
        """
            CHEBI:6801(( category biolink:ChemicalSubstance ))
            MONDO:0005148(( category biolink:Disease ))
            MONDO:0005148<-- predicate biolink:treats --CHEBI:6801
        """
    },
    normalizer_data="""
        MONDO:0005148 categories biolink:Disease
        CHEBI:6801 categories biolink:ChemicalSubstance
        """
)
async def test_solve_reverse_edge():
    """
    Test that we can solve a simple query graph
    where we have to traverse an edge in the opposite
    direction of one that was given
    """

    QGRAPH = query_graph_from_string(
        """
        n0(( id MONDO:0005148 ))
        n1(( category biolink:ChemicalSubstance ))
        n1-- biolink:treats -->n0
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    validate_message(
        {
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
                    n1n0 CHEBI:6801-MONDO:0005148
                """
            ]
        },
        output["message"],
    )


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    kp_data={
        "kp0":
        """
            MONDO:0005148(( category biolink:Disease ))
            MONDO:0005148<-- predicate biolink:treats --CHEBI:6801
            CHEBI:6801(( category biolink:ChemicalSubstance ))
        """,
        "kp1":
        """
            CHEBI:6801(( category biolink:ChemicalSubstance ))
            CHEBI:6801<-- predicate biolink:has_biomarker --HP:0004324
            HP:0004324(( category biolink:PhenotypicFeature ))
        """
    },
    normalizer_data="""
        MONDO:0005148 categories biolink:Disease
        CHEBI:6801 categories biolink:ChemicalSubstance
        HP:0004324 categories biolink:PhenotypicFeature
        """
)
async def test_solve_multiple_reverse_edges():
    """
    Test that we can solve a query graph
    where we have to traverse two reverse edges
    """

    QGRAPH = query_graph_from_string(
        """
        n0(( id MONDO:0005148 ))
        n1-- biolink:treats -->n0
        n1(( category biolink:ChemicalSubstance ))
        n2-- biolink:has_biomarker -->n1
        n2(( category biolink:PhenotypicFeature ))
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    validate_message(
        {
            "knowledge_graph":
                """
                CHEBI:6801 biolink:treats MONDO:0005148
                HP:0004324 biolink:has_biomarker CHEBI:6801
                """,
            "results": [
                """
                node_bindings:
                    n0 MONDO:0005148
                    n1 CHEBI:6801
                    n2 HP:0004324
                edge_bindings:
                    n1n0 CHEBI:6801-MONDO:0005148
                    n2n1 HP:0004324-CHEBI:6801
                """
            ]
        },
        output["message"],
    )


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    kp_data={
        "kp0":
        """
            CHEBI:6801(( category biolink:ChemicalSubstance ))
            MONDO:0005148(( category biolink:Disease ))
            CHEBI:6801-- predicate biolink:not_a_real_predicate -->MONDO:0005148
        """
    },
    normalizer_data="""
        CHEBI:6801 categories biolink:ChemicalSubstance
        MONDO:0005148 categories biolink:Disease
        """
)
async def test_solve_not_real_predicate():
    """
    Test that we can solve a query graph with
    predicates that we don't recognize
    """

    QGRAPH = query_graph_from_string(
        """
        n0(( id CHEBI:6801 ))
        n1(( category biolink:Disease ))
        n0-- biolink:not_a_real_predicate -->n1
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    validate_message(
        {
            "knowledge_graph":
                """
                CHEBI:6801 biolink:not_a_real_predicate MONDO:0005148
                """,
            "results": [
                """
                node_bindings:
                    n0 CHEBI:6801
                    n1 MONDO:0005148
                edge_bindings:
                    n0n1 CHEBI:6801-MONDO:0005148
                """
            ]
        },
        output["message"],
    )


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    kp_data={
        "kp0":
        """
            PR:000014578(( category biolink:GeneOrGeneProduct ))
            PR:000014578-- predicate biolink:related_to -->PR:000005855
            PR:000005855(( category biolink:GeneOrGeneProduct ))
        """,
    },
    normalizer_data="""
        PR:000014578 categories biolink:Protein
        PR:000005855 categories biolink:Protein
        """
)
async def test_convert_protein_to_gene_product():
    """
    Test that when given a protein we can use knowledge sources
    that provide GeneOrGeneProduct
    """

    QGRAPH = query_graph_from_string(
        """
        n0(( id PR:000014578 ))
        n1(( category biolink:Protein ))
        n0-- biolink:related_to -->n1
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    validate_message(
        {
            "knowledge_graph":
                """
                PR:000014578 biolink:related_to PR:000005855
                """,
            "results": [
                """
                node_bindings:
                    n0 PR:000014578
                    n1 PR:000005855
                edge_bindings:
                    n0n1 PR:000014578-PR:000005855
                """
            ]
        },
        output["message"],
    )


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    kp_data={
        "kp0":
        """
            MONDO:1(( category biolink:Disease ))
            MONDO:1-- predicate biolink:treated_by -->MESH:1
            MESH:1(( category biolink:ChemicalSubstance ))
        """,
        "kp1":
        """
            MONDO:1(( category biolink:ChemicalSubstance ))
            MONDO:1-- predicate biolink:ameliorates -->HP:1
            HP:1(( category biolink:PhenotypicFeature ))
        """,
    },
    normalizer_data="""
        MONDO:1 categories biolink:NamedThing
        MESH:1 categories biolink:ChemicalSubstance
        HP:1 categories biolink:PhenotypicFeature
        """
)
async def test_solve_double_subclass():
    """
    Test that when given a node with a general type that we subclass
    it and contact all KPs available for information about that node
    """

    QGRAPH = query_graph_from_string(
        """
        n0(( id MONDO:1 ))
        n1(( category biolink:NamedThing ))
        n0-- biolink:related_to -->n1
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    validate_message(
        {
            "knowledge_graph":
                """
                MONDO:1 biolink:treated_by MESH:1
                MONDO:1 biolink:ameliorates HP:1
                """,
            "results": [
                """
                node_bindings:
                    n0 MONDO:1
                    n1 MESH:1
                edge_bindings:
                    n0n1 MONDO:1-MESH:1
                """,
                """
                node_bindings:
                    n0 MONDO:1
                    n1 HP:1
                edge_bindings:
                    n0n1 MONDO:1-HP:1
                """
            ]
        },
        output["message"],
    )

    # Check that the knowledge graph node has the correct category
    assert set(output["message"]["knowledge_graph"]["nodes"]["MONDO:1"]["categories"]) == \
        {"biolink:ChemicalSubstance", "biolink:Disease"}


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    kp_data={
        "kp0":
        """
            MONDO:1(( category biolink:Disease ))
            MONDO:1-- predicate biolink:treated_by -->CHEBI:1
            MONDO:1-- predicate biolink:treated_by -->CHEBI:2
            CHEBI:1(( category biolink:ChemicalSubstance ))
            CHEBI:2(( category biolink:ChemicalSubstance ))
        """
    },
    normalizer_data="""
        MONDO:1 categories biolink:Disease
        CHEBI:1 categories biolink:ChemicalSubstance
        CHEBI:2 categories biolink:ChemicalSubstance
        """
)
async def test_pinned_to_pinned():
    """
    Test that we can solve a query to check if a pinned node is
    connected to another pinned node
    """

    QGRAPH = query_graph_from_string(
        """
        n0(( id MONDO:1 ))
        n1(( id CHEBI:1 ))
        n0-- biolink:related_to -->n1
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    validate_message(
        {
            "knowledge_graph":
                """
                MONDO:1 biolink:treated_by CHEBI:1
                """,
            "results": [
                """
                node_bindings:
                    n0 MONDO:1
                    n1 CHEBI:1
                edge_bindings:
                    n0n1 MONDO:1-CHEBI:1
                """,
            ]
        },
        output["message"],
    )


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    kp_data={
        "kp0":
        """
            CHEBI:1(( category biolink:ChemicalSubstance ))
            CHEBI:1-- predicate biolink:increases_uptake_of -->CHEBI:1
        """
    },
    normalizer_data="""
        CHEBI:1 categories biolink:ChemicalSubstance
        """
)
async def test_self_edge():
    """
    Test that we can solve a query with a self-edge
    """

    QGRAPH = query_graph_from_string(
        """
        n0(( id CHEBI:1 ))
        n0-- biolink:related_to -->n0
        """
    )

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    validate_message(
        {
            "knowledge_graph":
                """
                CHEBI:1 biolink:increases_uptake_of CHEBI:1
                """,
            "results": [
                """
                node_bindings:
                    n0 CHEBI:1
                edge_bindings:
                    n0n0 CHEBI:1-CHEBI:1
                """,
            ]
        },
        output["message"],
    )


@pytest.mark.asyncio
async def test_exception_response():
    """
    Test that an exception's response is a 500 error
    includes the correct CORS headers,
    and is a valid TRAPI message with a log included
    """
    # Referring to nodes that don't exist will induce a 500 error
    qgraph = {"nodes": {}, "edges": {
        "n0n1": {"subject": "n0", "object": "n1"}}}

    response = await client.post(
        "/query",
        json={"message": {"query_graph": qgraph}},
        headers={"origin": "http://localhost:80"}
    )

    assert response.status_code == 500
    assert "access-control-allow-origin" in response.headers
    assert len(response.json()["message"]["logs"])

@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
)
async def test_constraint_error():
    """
    Test that we throw an error and exit if we encounter
    any constraints (not implemented yet)
    """
    QGRAPH = query_graph_from_string(
        """
        n0(( id CHEBI:6801 ))
        n0-- biolink:treats -->n1
        n1(( category biolink:Disease ))
        """
    )

    QGRAPH["nodes"]["n0"]["constraints"] = [
        {
            "name" : "Chromosome band",
            "id" : "NCIT:C13432",
            "operator" : "==",
            "value" : "11q13.*",
        }
    ]

    # Create query
    q = {"message" : {"query_graph" : QGRAPH}}

    # Run
    response = await client.post("/query", json=q)
    output = response.json()

    # Check that we stored the error
    assert 'Unable to process query due to constraints' in output["message"]["logs"][0]['message']