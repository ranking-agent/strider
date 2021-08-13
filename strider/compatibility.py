"""Compatibility utilities."""
from collections import namedtuple
from functools import cache
from json.decoder import JSONDecodeError
import logging

import httpx
import pydantic
from reasoner_pydantic import Message
from reasoner_pydantic.message import Query, Response
from trapi_throttle.throttle import ThrottledServer
from trapi_throttle.utils import log_request

from .util import StriderRequestError, remove_null_values, log_response, log_request
from .trapi import apply_curie_map, get_curies
from .config import settings


class KnowledgePortal():
    """Knowledge portal."""

    def __init__(
            self,
            logger: logging.Logger = None,
    ):
        """Initialize."""
        if not logger:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.synonymizer = Synonymizer(self.logger)
        self.tservers: dict[str, ThrottledServer] = dict()

    async def make_throttled_request(self, kp_id, request, logger, log_name):
        """
        Make post request and write errors to log if present
        """
        try:
            return await self.tservers[kp_id].query(request)
        except httpx.ReadTimeout as e:
            logger.error({
                "message": f"{log_name} took >60 seconds to respond",
                "error": str(e),
                "request": log_request(e.request),
            })
        except httpx.RequestError as e:
            # Log error
            logger.error({
                "message": f"Request Error contacting {log_name}",
                "error": str(e),
                "request": log_request(e.request),
            })
        except httpx.HTTPStatusError as e:
            # Log error with response
            logger.error({
                "message": f"Response Error contacting {log_name}",
                "error": str(e),
                "request": log_request(e.request),
                "response": log_response(e.response),
            })
        except JSONDecodeError as e:
            # Log error with response
            logger.error({
                "message": f"Received bad JSON data from {log_name}",
                "request": e.request,
                "response": e.response.text,
                "error": str(e),
            })
        except pydantic.ValidationError as e:
            logger.error({
                "message": "Received non-TRAPI compliant response from KP",
                "error": str(e),
            })
        except Exception as e:
            logger.error({
                "message": "Something went wrong while querying KP",
                "error": str(e),
            })
        raise StriderRequestError

    async def map_prefixes(
            self,
            message: Message,
            prefixes: dict[str, list[str]],
    ) -> Message:
        """Map prefixes."""
        await self.synonymizer.load_message(message)
        curie_map = self.synonymizer.map(prefixes)
        return apply_curie_map(message, curie_map)

    async def fetch(
            self,
            kp_id: str,
            request: dict,
            input_prefixes: dict = None,
            output_prefixes: dict = None,
    ):
        """Wrap fetch with CURIE mapping(s)."""
        request['message'] = await self.map_prefixes(request['message'], input_prefixes)
        request = remove_null_values(request)

        trapi_request = Query.parse_obj(request).dict(by_alias=True, exclude_unset=True)
        try:
            response = await self.make_throttled_request(kp_id, trapi_request, self.logger, "KP")
        except StriderRequestError:
            # Continue processing with an empty response object
            response = {
                "message" : {
                    "query_graph": request["message"]["query_graph"],
                    "knowledge_graph": {"nodes": {}, "edges": {}},
                    "results": [],
                }
            }

        try:
            # Parse with reasoner_pydantic to validate
            response = Response.parse_obj(response).dict(
                exclude_unset=True,
                exclude_none=True,
            )
        except pydantic.ValidationError as e:
            self.logger.error({
                "message": "Received non-TRAPI compliant response from KP",
                "response": response,
                "error": str(e),
            })
            # Continue processing with an empty response object
            response = {
                "message" : {
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

        message = await self.map_prefixes(message, output_prefixes)

        add_source(message)

        return message


def add_source(message: Message):
    """Add provenance annotation to kedges.
       Sources from which we retrieve data add their own prov, we add prov for aragorn."""
    for kedge in message["knowledge_graph"]["edges"].values():
        kedge["attributes"] = kedge.get("attributes", []) + [dict(
            attribute_type_id="biolink:aggregator_knowledge_source",
            value="infores:aragorn",
        )]


Entity = namedtuple("Entity", ["categories", "identifiers"])


class Synonymizer():
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
                    json = {"curies": list(curies)},
                )
                response.raise_for_status()
        except httpx.RequestError as e:
            # Log error
            self.logger.warning({
                "message": "RequestError contacting normalizer. Results may be incomplete",
                "request": log_request(e.request),
                "error": str(e),
            })
            return
        except httpx.HTTPStatusError as e:
            # Log error with response
            self.logger.warning({
                "message": "Error contacting normalizer. Results may be incomplete",
                "request": log_request(e.request),
                "response": e.response.text,
                "error": str(e),
            })
            return

        entities = [
            Entity(
                entity["type"] + (
                    ["biolink:ChemicalSubstance"]
                    if "biolink:SmallMolecule" in entity["type"] else
                    []
                ),
                [
                    synonym["identifier"]
                    for synonym in entity["equivalent_identifiers"]
                ],
            )
            for entity in response.json().values() if entity
        ]
        self._data |= {
            curie: entity
            for entity in entities
            for curie in entity.identifiers
        }

    def __getitem__(self, curie: str):
        """Get preferred curie."""
        return self._data[curie]

    def map(self, prefixes: dict[str, list[str]]):
        """Generate CURIE map."""
        return CURIEMap(self, prefixes, self.logger)


class CURIEMap():
    """CURIE map."""

    def __init__(
            self,
            lookup: Synonymizer,
            prefixes: dict[str, list[str]],
            logger: logging.Logger
    ):
        """Initialize."""
        self.prefixes = prefixes
        self.lookup = lookup
        self.logger = logger

    @cache
    def __getitem__(self, curie):
        """Get preferred curie."""
        categories, identifiers = self.lookup[curie]
        try:
            prefixes = next(
                self.prefixes[category]
                for category in categories
                if category in self.prefixes
            )
        except StopIteration:
            # no preferred prefixes for these categories
            self.logger.warning(
                f"Could not find preferred prefixes for at least one of: {categories}")
            return [curie]

        # Iterate through prefixes until we find
        # one with CURIEs
        prefix_identifiers = [
            curie
            for prefix in prefixes
            for curie in identifiers
            if curie.startswith(prefix)
        ]
        if len(prefix_identifiers) > 0:
            return prefix_identifiers

        # no preferred curie with these prefixes
        self.logger.warning(
            "Cannot find identifier in {} with a preferred prefix in {}".format(
                identifiers,
                prefixes,
            ),
        )
        return [curie]

    def get(self, *args):
        """Get item."""
        try:
            return self.__getitem__(args[0])
        except KeyError:
            if len(args) > 1:
                return args[1]
            raise
