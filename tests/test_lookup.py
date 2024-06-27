"""Test Strider."""

import json

from fastapi.responses import Response
import pytest
from pytest_httpx import HTTPXMock
import redis.asyncio
from reasoner_pydantic import Response as PydanticResponse

from tests.helpers.logger import setup_logger
from tests.helpers.utils import query_graph_from_string
from tests.helpers.redisMock import redisMock
import tests.helpers.mock_responses as mock_responses

from strider.config import settings
from strider.server import lookup

# Switch prefix path before importing server
settings.normalizer_url = "http://normalizer"


setup_logger()


@pytest.mark.asyncio
async def test_mixed_canonical(monkeypatch, mocker):
    """Test qedge with mixed canonical and non-canonical predicates."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.throttle.ThrottledServer._query",
        return_value=PydanticResponse.parse_obj({"message": {}}),
    )
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
        "message": {"query_graph": QGRAPH},
        "log_level": "ERROR",
    }

    # Run
    await lookup(q)

    query.assert_called_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["CHEBI:6801"],
                            "categories": ["biolink:ChemicalSubstance"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                        "n1": {
                            "categories": ["biolink:Disease"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                    },
                    "edges": {
                        "n0n1": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats", "biolink:phenotype_of"],
                            "attribute_constraints": [],
                            "qualifier_constraints": [],
                        },
                    },
                },
            },
        },
        False,
        ["infores:kp1"],
        True,
    )


@pytest.mark.asyncio
async def test_symmetric_noncanonical(monkeypatch, mocker):
    """Test qedge with the symmetric, non-canonical predicate genetically_interacts_with."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.throttle.ThrottledServer._query",
        return_value=PydanticResponse.parse_obj({"message": {}}),
    )
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
        "message": {"query_graph": QGRAPH},
        "log_level": "INFO",
    }

    # Run
    await lookup(q)

    query.assert_called_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["CHEBI:6801"],
                            "categories": ["biolink:ChemicalSubstance"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                        "n1": {
                            "categories": ["biolink:Disease"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                    },
                    "edges": {
                        "n0n1": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:genetically_interacts_with"],
                            "attribute_constraints": [],
                            "qualifier_constraints": [],
                        },
                    },
                },
            },
        },
        False,
        ["infores:kp1"],
        True,
    )


@pytest.mark.asyncio
async def test_disambiguation(monkeypatch, httpx_mock: HTTPXMock):
    """
    Test disambiguating batch results with qnode_id.
    """
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(
        url="http://kp1/query", json=mock_responses.disambiguation_response
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0-- biolink:treats -->n1
        n1(( categories[] biolink:Disease ))
        """
    )

    # Create query
    q = {
        "message": {"query_graph": QGRAPH},
    }

    # Run
    output = await lookup(q)
    print(output)
    assert len(output["message"]["results"]) == 1


@pytest.mark.asyncio
async def test_trivial_unbatching(monkeypatch, httpx_mock: HTTPXMock):
    """Test trivial unbatching with batch size one."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    httpx_mock.add_response(
        url="http://kp1/query", json=mock_responses.unbatching_response
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0-- biolink:treats -->n1
        n1(( categories[] biolink:Disease ))
        """
    )

    # Create query
    q = {
        "message": {"query_graph": QGRAPH},
        "log_level": "DEBUG",
    }

    # Run
    output = await lookup(q)
    assert len(output["message"]["results"]) == 1


@pytest.mark.asyncio
async def test_protein_gene_conflation(monkeypatch, mocker):
    """Test conflation of biolink:Gene and biolink:Protein categories.
    e0 checks that Gene is added to Protein nodes."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.throttle.ThrottledServer._query",
        return_value=PydanticResponse.parse_obj({"message": {}}),
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] MONDO:0008114 ))
        n0(( categories[] biolink:Disease ))
        n1(( categories[] biolink:Protein ))
        n0-- biolink:related_to -->n1
        """
    )

    # Create query
    q = {"message": {"query_graph": QGRAPH}, "log_level": "INFO"}

    # Run query
    await lookup(q)

    query.assert_called_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["MONDO:0008114"],
                            "categories": ["biolink:Disease"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                        "n1": {
                            "categories": ["biolink:Protein", "biolink:Gene"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                    },
                    "edges": {
                        "n0n1": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                            "attribute_constraints": [],
                            "qualifier_constraints": [],
                        },
                    },
                },
            },
        },
        False,
        ["infores:kp2"],
        True,
    )


@pytest.mark.asyncio
async def test_gene_protein_conflation(monkeypatch, mocker):
    """Test conflation of biolink:Gene and biolink:Protein categories.
    e0 checks to make sure that Protein is added to Gene nodes."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.throttle.ThrottledServer._query",
        return_value=PydanticResponse.parse_obj({"message": {}}),
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( categories[] biolink:Gene ))
        n1(( ids[] MONDO:0008114 ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:related_to -->n1
        """
    )

    # Create query
    q = {"message": {"query_graph": QGRAPH}, "log_level": "INFO"}

    # Run query
    await lookup(q)

    query.assert_called_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "categories": ["biolink:Gene", "biolink:Protein"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                        "n1": {
                            "ids": ["MONDO:0008114"],
                            "categories": ["biolink:Disease"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                    },
                    "edges": {
                        "n0n1": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:related_to"],
                            "attribute_constraints": [],
                            "qualifier_constraints": [],
                        },
                    },
                },
            },
        },
        False,
        ["infores:kp2"],
        True,
    )


@pytest.mark.asyncio
async def test_node_set(monkeypatch, mocker):
    """Test that is_set is handled correctly."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.throttle.ThrottledServer._query",
        return_value=PydanticResponse.parse_obj({"message": {}}),
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:treats -->n1
        """
    )
    QGRAPH["nodes"]["n1"]["set_interpretation"] = "ALL"

    # Create query
    q = {
        "message": {"query_graph": QGRAPH},
        "log_level": "WARNING",
    }

    # Run
    await lookup(q)

    query.assert_called_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["CHEBI:6801"],
                            "categories": ["biolink:ChemicalSubstance"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                        "n1": {
                            "categories": ["biolink:Disease"],
                            "set_interpretation": "ALL",
                            "constraints": [],
                        },
                    },
                    "edges": {
                        "n0n1": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                            "attribute_constraints": [],
                            "qualifier_constraints": [],
                        },
                    },
                },
            },
        },
        False,
        ["infores:kp1"],
        True,
    )


@pytest.mark.asyncio
async def test_bypass_cache_is_sent_along_to_kps(monkeypatch, mocker):
    """Test that is_set is handled correctly."""
    monkeypatch.setattr(redis.asyncio, "Redis", redisMock)
    query = mocker.patch(
        "strider.throttle.ThrottledServer._query",
        return_value=PydanticResponse.parse_obj({"message": {}}),
    )
    QGRAPH = query_graph_from_string(
        """
        n0(( ids[] CHEBI:6801 ))
        n0(( categories[] biolink:ChemicalSubstance ))
        n1(( categories[] biolink:Disease ))
        n0-- biolink:treats -->n1
        """
    )

    # Create query
    q = {
        "message": {"query_graph": QGRAPH},
        "bypass_cache": True,
        "log_level": "WARNING",
    }

    # Run
    await lookup(q)

    query.assert_called_with(
        {
            "message": {
                "query_graph": {
                    "nodes": {
                        "n0": {
                            "ids": ["CHEBI:6801"],
                            "categories": ["biolink:ChemicalSubstance"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                        "n1": {
                            "categories": ["biolink:Disease"],
                            "set_interpretation": "BATCH",
                            "constraints": [],
                        },
                    },
                    "edges": {
                        "n0n1": {
                            "subject": "n0",
                            "object": "n1",
                            "predicates": ["biolink:treats"],
                            "attribute_constraints": [],
                            "qualifier_constraints": [],
                        },
                    },
                },
            },
        },
        True,
        ["infores:kp1"],
        True,
    )
