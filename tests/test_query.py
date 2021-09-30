"""Test Strider."""
import json

import fakeredis
from fastapi.responses import Response
import pytest

from tests.helpers.context import with_translator_overlay, with_response_overlay
from tests.helpers.logger import setup_logger
from tests.helpers.utils import query_graph_from_string, validate_message

from strider.config import settings
from strider.server import APP, lookup

# Switch prefix path before importing server
settings.kpregistry_url = "http://registry"
settings.normalizer_url = "http://normalizer"


@pytest.fixture()
def redis():
    """Create a Redis client."""
    return fakeredis.FakeRedis(
        encoding="utf-8",
        decode_responses=True,
    )


setup_logger()


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
async def test_solve_ex1(redis):
    """Test solving the ex1 query graph"""
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:Disease ))
        n2(( categories[] biolink:PhenotypicFeature ))
        n0-- biolink:treats -->n1
        n1-- biolink:has_phenotype -->n2
        """
    )

    # Create query
    q = {
        "message" : {"query_graph" : QGRAPH},
        "log_level": "ERROR",
    }

    # Run
    output = await lookup(q, redis)

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
            MONDO:0005148(( category biolink:Disease ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
            MONDO:0005148-- predicate biolink:has_phenotype -->CHEBI:6801
        """
    }
)
async def test_mixed_canonical(redis):
    """Test qedge with mixed canonical and non-canonical predicates."""
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:treats biolink:phenotype_of -->n1
        """
    )

    # Create query
    q = {
        "message" : {"query_graph" : QGRAPH},
        "log_level": "ERROR",
    }

    # Run
    output = await lookup(q, redis)

    assert len(output["message"]["results"]) == 2


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    {
        "ctd":
        """
            CHEBI:6801(( category biolink:ChemicalSubstance ))
            MONDO:0005148(( category biolink:Disease ))
            CHEBI:6801-- predicate biolink:genetically_interacts_with -->MONDO:0005148
        """
    }
)
async def test_symmetric_noncanonical(redis):
    """Test qedge with the symmetric, non-canonical predicate genetically_interacts_with."""
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:genetically_interacts_with -->n1
        """
    )

    # Create query
    q = {
        "message" : {"query_graph" : QGRAPH},
        "log_level": "ERROR",
    }

    # Run
    output = await lookup(q, redis)

    assert len(output["message"]["results"]) == 1


@pytest.mark.asyncio
@with_translator_overlay(
    settings.kpregistry_url,
    settings.normalizer_url,
    {
        "ctd":
        """
            CHEBI:6801(( category biolink:SmallMolecule ))
            MONDO:0005148(( category biolink:Disease ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
        """,
    }
)
# Add attributes to ctd response
@with_response_overlay(
    "http://ctd/query",
    Response(
        status_code=200,
        content=json.dumps({"message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": ["CHEBI:6801"]},
                    "n1": {"categories": ["biolink:Disease"]},
                },
                "edges": {
                    "n0n1": {
                        "subject": "n0",
                        "predicate": "biolink:treats",
                        "object": "n1",
                    },
                },
            },
            "knowledge_graph": {
                "nodes": {
                    "CHEBI:XXX": {},
                    "MONDO:0005148": {
                        "attributes": [
                            {
                                "attribute_type_id": "test_constraint",
                                "value": "foo",
                            },
                        ],
                    },
                },
                "edges": {
                    "n0n1": {
                        "subject": "CHEBI:XXX",
                        "predicate": "biolink:treats",
                        "object": "MONDO:0005148",
                    },
                },
            },
            "results": [
                {
                    "node_bindings": {
                        "n0": [{
                            "id": "CHEBI:XXX",
                            "qnode_id": "CHEBI:6801",
                        }],
                        "n1": [{"id": "MONDO:0005148"}],
                    },
                    "edge_bindings": {
                        "n0n1": [{"id": "n0n1"}],
                    },
                },
            ],
        }}),
    )
)
async def test_disambiguation(redis):
    """
    Test disambiguating batch results with qnode_id.
    """
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0-- biolink:treats -->n1
        n1(( categories[] biolink:Disease ))
        """
    )

    # Create query
    q = {
        "message" : {"query_graph" : QGRAPH},
        "log_level": "ERROR",
    }

    # Run
    output = await lookup(q, redis)
    assert len(output["message"]["results"]) == 1

    validate_message(
        {
            "knowledge_graph":
                """
                CHEBI:XXX biolink:treats MONDO:0005148
                """,
            "results": [
                """
                node_bindings:
                    n0 CHEBI:XXX
                    n1 MONDO:0005148
                edge_bindings:
                    n0n1 CHEBI:XXX-MONDO:0005148
                """,
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
            CHEBI:6801(( category biolink:SmallMolecule ))
            MONDO:0005148(( category biolink:Disease ))
            CHEBI:6801-- predicate biolink:treats -->MONDO:0005148
        """,
    }
)
# Add attributes to ctd response
@with_response_overlay(
    "http://ctd/query",
    Response(
        status_code=200,
        content=json.dumps({"message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": ["CHEBI:6801"]},
                    "n1": {"categories": ["biolink:Disease"]},
                },
                "edges": {
                    "n0n1": {
                        "subject": "n0",
                        "predicate": "biolink:treats",
                        "object": "n1",
                    },
                },
            },
            "knowledge_graph": {
                "nodes": {
                    "CHEBI:XXX": {},
                    "MONDO:0005148": {
                        "attributes": [
                            {
                                "attribute_type_id": "test_constraint",
                                "value": "foo",
                            },
                        ],
                    },
                },
                "edges": {
                    "n0n1": {
                        "subject": "CHEBI:XXX",
                        "predicate": "biolink:treats",
                        "object": "MONDO:0005148",
                    },
                },
            },
            "results": [
                {
                    "node_bindings": {
                        "n0": [{"id": "CHEBI:XXX"}],
                        "n1": [{"id": "MONDO:0005148"}],
                    },
                    "edge_bindings": {
                        "n0n1": [{"id": "n0n1"}],
                    },
                },
            ],
        }}),
    )
)
async def test_trivial_unbatching(redis):
    """Test trivial unbatching with batch size one."""
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0-- biolink:treats -->n1
        n1(( categories[] biolink:Disease ))
        """
    )

    # Create query
    q = {
        "message" : {"query_graph" : QGRAPH},
        "log_level": "ERROR",
    }

    # Run
    output = await lookup(q, redis)
    assert len(output["message"]["results"]) == 1

    validate_message(
        {
            "knowledge_graph":
                """
                CHEBI:XXX biolink:treats MONDO:0005148
                """,
            "results": [
                """
                node_bindings:
                    n0 CHEBI:XXX
                    n1 MONDO:0005148
                edge_bindings:
                    n0n1 CHEBI:XXX-MONDO:0005148
                """,
            ],
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
            MONDO:0008114(( category biolink:Disease ))
            MONDO:0008114-- predicate biolink:related_to -->MESH:C035133
            MESH:C035133(( category biolink:Gene ))
        """,
        "kp1":
        """
            MESH:C035133(( category biolink:Gene ))
            MESH:C035133-- predicate biolink:related_to -->HP:0007430
            HP:0007430(( category biolink:Protein ))
        """,
        "kp2":
        """
            HP:0007430(( category biolink:Protein ))
            HP:0007430-- predicate biolink:related_to -->CHEBI:6801
            CHEBI:6801(( category biolink:Disease ))
            HGNC:6284(( category biolink:Gene ))
            HP:0007430-- predicate biolink:related_to -->HGNC:6284
        """
    },
    normalizer_data="""
        MONDO:0008114 categories biolink:Disease
        HP:0007430 categories biolink:Protein
        MESH:C035133 categories biolink:Gene
        HGNC:6284 categories biolink:Gene
        CHEBI:6801 categories biolink:Disease
        """
)
async def test_gene_protein_conflation(redis):
    """Test conflation of biolink:Gene and biolink:Protein categories"""
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] MONDO:0008114 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:Protein ))
        n2(( categories[] biolink:Gene ))
        n3(( categories[] biolink:Disease ))
        n0-- biolink:related_to -->n1
        n1-- biolink:related_to -->n2
        n2-- biolink:related_to -->n3
        """
    )

    q = {
        "message" : {"query_graph" : QGRAPH},
        "log_level" : "ERROR"
    }

    output = await lookup(q, redis)

    validate_message({
        "knowledge_graph":
            """
            MONDO:0008114 biolink:related_to MESH:C035133
            MESH:C035133 biolink:related_to HP:0007430
            HP:0007430 biolink:related_to CHEBI:6801
            """,
        "results": [
            """
            node_bindings:
                n0 MONDO:0008114
                n1 MESH:C035133
                n2 HP:0007430
                n3 CHEBI:6801
            edge_bindings:
                n0n1 MONDO:0008114-MESH:C035133
                n1n2 MESH:C035133-HP:0007430
                n2n3 HP:0007430-CHEBI:6801
            """
        ],
    },
        output["message"])
