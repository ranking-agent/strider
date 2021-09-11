"""Test Strider."""
import fakeredis
import pytest

from tests.helpers.context import with_translator_overlay
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
