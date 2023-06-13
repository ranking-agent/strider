import asyncio
import httpx
from json.decoder import JSONDecodeError
import logging
import pydantic
from reasoner_pydantic import (
    Message,
    QueryGraph,
    KnowledgeGraph,
    Results,
    AuxiliaryGraphs,
    RetrievalSource,
)

from .trapi_throttle.throttle import ThrottledServer
from .util import (
    StriderRequestError,
    elide_curies,
    remove_null_values,
    log_response,
    log_request,
)
from .trapi import apply_curie_map, get_curies
from .synonymizer import Synonymizer


class KPError(Exception):
    """Exception in a KP request."""


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
                    "message": f"Knowledge_Provider: Something went wrong while querying {kp_id}",
                    "error": str(e),
                    "traceback": e.with_traceback(),
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
        if len(curies):
            await self.synonymizer.load_curies(*curies)
            curie_map = self.synonymizer.map(curies, prefixes, logger)
            apply_curie_map(message, curie_map)

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

        message = response.message
        if message.query_graph is None:
            message = Message(
                query_graph=QueryGraph.parse_obj(request["message"]["query_graph"]),
                knowledge_graph=KnowledgeGraph.parse_obj({"nodes": {}, "edges": {}}),
                results=Results.parse_obj([]),
                auxiliary_graphs=AuxiliaryGraphs.parse_obj({}),
            )
        if message.knowledge_graph is None:
            message.knowledge_graph = KnowledgeGraph.parse_obj({"nodes": {}, "edges": {}})
        if message.results is None:
            message.results = Results.parse_obj([])
        if message.auxiliary_graphs is None:
            message.auxiliary_graphs = AuxiliaryGraphs.parse_obj({})

        add_source(message, kp_id)

        return message


class KnowledgeProvider:
    """Knowledge provider."""

    def __init__(self, details, portal, id, *args, **kwargs):
        """Initialize."""
        self.details = details
        self.portal = portal
        # self.portal: KnowledgePortal = portal
        self.id = id

    async def solve_onehop(self, request):
        """Solve one-hop query."""
        return await self.portal.fetch(
            self.id,
            {"message": {"query_graph": request}},
        )


def add_source(message: Message, kp_id):
    """Add provenance annotation to kedges.
    Sources from which we retrieve data add their own prov, we add prov for aragorn."""
    for kedge in message.knowledge_graph.edges.values():
        # create copy of kedge
        new_kedge = kedge.copy()
        new_kedge.sources.add(
            RetrievalSource.parse_obj({
                "resource_id": "infores:aragorn",
                "resource_role": "aggregator_knowledge_source",
                "upstream_resource_ids": [kp_id],
            })
        )
        # update existing kedge
        kedge.update(new_kedge)
    for result in message.results:
        for analysis in result.analyses:
            analysis.resource_id = "infores:aragorn"
            # remove scores generated by KP
            analysis.score = None
            analysis.scoring_method = None