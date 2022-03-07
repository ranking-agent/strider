"""Compatibility utilities."""
import asyncio
from collections import namedtuple
from functools import cache
from json.decoder import JSONDecodeError
import logging

import httpx
import pydantic
from reasoner_pydantic import Message
from reasoner_pydantic.message import Query, Response

from .trapi_throttle.throttle import ThrottledServer
from .util import (
    StriderRequestError,
    elide_curies,
    remove_null_values,
    log_response,
    log_request,
)
from .trapi import apply_curie_map, get_curies
from .config import settings


class KnowledgePortal:
    """Knowledge portal."""

    def __init__(
        self,
        synonymizer: "Synonymizer" = None,
        logger: logging.Logger = None,
    ):
        """Initialize."""
        if not logger:
            logger = logging.getLogger(__name__)
        if not synonymizer:
            synonymizer = Synonymizer(logger=logger)
        self.logger = logger
        self.synonymizer = synonymizer
        self.tservers: dict[str, ThrottledServer] = dict()

    async def make_throttled_request(
        self,
        kp_id: str,
        request: dict,
        logger: logging.Logger,
        timeout: float = 60.0,
    ):
        """
        Make post request and write errors to log if present
        """
        try:
            return await self.tservers[kp_id].query(request, timeout=timeout)
        except asyncio.TimeoutError as e:
            logger.warning(
                {
                    "message": f"{kp_id} took >{timeout} seconds to respond",
                    "error": str(e),
                    "request": elide_curies(request),
                }
            )
        except httpx.ReadTimeout as e:
            logger.warning(
                {
                    "message": f"{kp_id} took >60 seconds to respond",
                    "error": str(e),
                    "request": log_request(e.request),
                }
            )
        except httpx.RequestError as e:
            # Log error
            logger.warning(
                {
                    "message": f"Request Error contacting {kp_id}",
                    "error": str(e),
                    "request": log_request(e.request),
                }
            )
        except httpx.HTTPStatusError as e:
            # Log error with response
            logger.warning(
                {
                    "message": f"Response Error contacting {kp_id}",
                    "error": str(e),
                    "request": log_request(e.request),
                    "response": log_response(e.response),
                }
            )
        except JSONDecodeError as e:
            # Log error with response
            logger.warning(
                {
                    "message": f"Received bad JSON data from {kp_id}",
                    "request": e.request,
                    "response": e.response.text,
                    "error": str(e),
                }
            )
        except pydantic.ValidationError as e:
            logger.warning(
                {
                    "message": f"Received non-TRAPI compliant response from {kp_id}",
                    "error": str(e),
                }
            )
        except Exception as e:
            logger.warning(
                {
                    "message": f"Something went wrong while querying {kp_id}",
                    "error": str(e),
                }
            )
        raise StriderRequestError

    async def map_prefixes(
        self,
        message: Message,
        prefixes: dict[str, list[str]],
        logger: logging.Logger = None,
    ) -> Message:
        """Map prefixes."""
        if not logger:
            logger = self.logger
        curies = get_curies(message)
        await self.synonymizer.load_curies(*curies)
        curie_map = self.synonymizer.map(curies, prefixes, logger)
        return apply_curie_map(message, curie_map)

    async def fetch(
        self,
        kp_id: str,
        request: dict,
    ):
        """Wrap fetch with CURIE mapping(s)."""
        request = remove_null_values(request)

        try:
            response = await self.make_throttled_request(
                kp_id,
                request,
                self.logger,
            )
        except StriderRequestError:
            # Continue processing with an empty response object
            response = {
                "message": {
                    "query_graph": request["message"]["query_graph"],
                    "knowledge_graph": {"nodes": {}, "edges": {}},
                    "results": [],
                }
            }

        message = response["message"]
        if message.get("query_graph") is None:
            message = {
                "query_graph": request["message"]["query_graph"],
                "knowledge_graph": {"nodes": {}, "edges": {}},
                "results": [],
            }
        if message.get("knowledge_graph") is None:
            message["knowledge_graph"] = {"nodes": {}, "edges": {}}
        if message.get("results") is None:
            message["results"] = []

        add_source(message)

        return message


def add_source(message: Message):
    """Add provenance annotation to kedges.
    Sources from which we retrieve data add their own prov, we add prov for aragorn."""
    for kedge in message["knowledge_graph"]["edges"].values():
        kedge["attributes"] = (kedge.get("attributes", None) or []) + [
            dict(
                attribute_type_id="biolink:aggregator_knowledge_source",
                value="infores:aragorn",
                value_type_id="biolink:InformationResource",
                attribute_source="infores:aragorn"  
            )
        ]


Entity = namedtuple("Entity", ["categories", "identifiers"])


class Synonymizer:
    """CURIE synonymizer."""

    def __init__(
        self,
        logger: logging.Logger,
    ):
        """Initialize."""
        self.logger = logger
        self._data = dict()

    async def load_message(self, message: Message):
        """Load map for concepts in message."""
        curies = get_curies(message)
        await self.load_curies(*curies)

    async def load_curies(self, *curies: list[str]):
        """Load CURIES into map."""
        # get all curie synonyms
        url_base = f"{settings.normalizer_url}/get_normalized_nodes"
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(
                    url_base,
                    json={"curies": list(curies)},
                )
                response.raise_for_status()
        except httpx.RequestError as e:
            # Log error
            self.logger.warning(
                {
                    "message": "RequestError contacting normalizer. Results may be incomplete",
                    "request": log_request(e.request),
                    "error": str(e),
                }
            )
            return
        except httpx.HTTPStatusError as e:
            # Log error with response
            self.logger.warning(
                {
                    "message": "Error contacting normalizer. Results may be incomplete",
                    "request": log_request(e.request),
                    "response": e.response.text,
                    "error": str(e),
                }
            )
            return

        entities = [
            Entity(
                entity["type"],
                [synonym["identifier"] for synonym in entity["equivalent_identifiers"]],
            )
            for entity in response.json().values()
            if entity
        ]
        self._data |= {
            curie: entity for entity in entities for curie in entity.identifiers
        }

    def __getitem__(self, curie: str):
        """Get preferred curie."""
        return self._data[curie]

    def map(
        self,
        curies: list[str],
        prefixes: dict[str, list[str]],
        logger: logging.Logger = None,
    ):
        """Generate CURIE map."""
        if not logger:
            logger = self.logger
        return {
            curie: self.map_curie(curie, self._data, prefixes, logger)
            for curie in curies
        }

    def map_curie(
        self,
        curie: str,
        data: dict[str, Entity],
        prefixes: dict[str, list[str]],
        logger: logging.Logger = None,
    ) -> str:
        """Map a single CURIE to the list of preferred equivalent CURIES.

        1. Find the most-preferred prefix for which the provided CURIE has synonyms.
        2. Return all synonymous CURIEs that have that prefix.
        """
        try:
            categories, identifiers = data[curie]
        except KeyError:
            return [curie]
        # Gather the preferred prefixes for each category, deduplicating while retaining order
        prefixes = list(
            dict.fromkeys(
                prefix
                for category in categories
                for prefix in prefixes.get(category, [])
            )
        )
        if not prefixes:
            # There are no preferred prefixes for these categories - use the prefixes that Biolink prefers
            logger.debug(
                "[{}] Cannot not find preferred prefixes for at least one of: {}".format(
                    getattr(logger, "context", ""),
                    categories,
                )
            )
            prefixes = identifiers[0].split(":")[0]

        # Find CURIEs beginning with the most-preferred prefix
        for prefix in prefixes:
            curies = [_curie for _curie in identifiers if _curie.startswith(prefix)]
            if curies:
                return curies

        # There is no equivalent CURIE with any of the acceptable prefixes - return the original CURIE
        logger.debug(
            "[{}] Cannot find identifier in {} with a preferred prefix in {}".format(
                getattr(logger, "context", ""),
                identifiers,
                prefixes,
            ),
        )
        return [curie]
