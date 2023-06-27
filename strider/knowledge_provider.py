import asyncio
import httpx
from json.decoder import JSONDecodeError
import pydantic
from reasoner_pydantic import (
    Response,
    Message,
    QueryGraph,
    KnowledgeGraph,
    Results,
    AuxiliaryGraphs,
    RetrievalSource,
)

from .throttle import ThrottledServer
from .utils import (
    WBMT,
    elide_curies,
    get_curies,
    remove_null_values,
    log_response,
    log_request,
)
from .trapi import apply_curie_map
from .normalizer import Normalizer
from .config import settings


class KnowledgeProvider:
    """Knowledge provider."""

    def __init__(self, kp_id, kp, logger, *args, **kwargs):
        """Initialize."""
        self.id = kp_id
        self.kp = kp
        self.logger = logger
        self.preferred_prefixes = WBMT.entity_prefix_mapping
        self.throttle = ThrottledServer(
            kp_id,
            url=kp["url"],
            logger=logger,
            preproc=self.get_processor(self.kp["details"]["preferred_prefixes"]),
            postproc=self.get_processor(self.preferred_prefixes),
            *args,
            *kwargs,
        )
        self.normalizer = Normalizer(logger=logger)

    def get_processor(self, preferred_prefixes):
        """Get processor."""

        async def processor(request):
            """Map message CURIE prefixes."""
            await self.map_prefixes(request.message, preferred_prefixes)

        return processor

    async def map_prefixes(
        self,
        message: Message,
        prefixes: dict[str, list[str]],
    ) -> Message:
        """Map prefixes."""
        curies = get_curies(message)
        if len(curies):
            await self.normalizer.load_curies(*curies)
            curie_map = self.normalizer.map(curies, prefixes)
            apply_curie_map(message, curie_map)

    async def solve_onehop(self, request):
        """Solve one-hop query."""
        request = remove_null_values(request)
        response = None
        try:
            response = await self.throttle.query({"message": {"query_graph": request}})
        except asyncio.TimeoutError as e:
            self.logger.warning(
                {
                    "message": f"{self.id} took > {settings.kp_timeout} seconds to respond",
                    "error": str(e),
                    "request": elide_curies(request),
                }
            )
        except httpx.ReadTimeout as e:
            self.logger.warning(
                {
                    "message": f"{self.id} took > {settings.kp_timeout} seconds to respond",
                    "error": str(e),
                    "request": log_request(e.request),
                }
            )
        except httpx.RequestError as e:
            # Log error
            self.logger.warning(
                {
                    "message": f"Request Error contacting {self.id}",
                    "error": str(e),
                    "request": log_request(e.request),
                }
            )
        except httpx.HTTPStatusError as e:
            # Log error with response
            self.logger.warning(
                {
                    "message": f"Response Error contacting {self.id}",
                    "error": str(e),
                    "request": log_request(e.request),
                    "response": log_response(e.response),
                }
            )
        except JSONDecodeError as e:
            # Log error with response
            self.logger.warning(
                {
                    "message": f"Received bad JSON data from {self.id}",
                    "request": e.request,
                    "response": e.response.text,
                    "error": str(e),
                }
            )
        except pydantic.ValidationError as e:
            self.logger.warning(
                {
                    "message": f"Received non-TRAPI compliant response from {self.id}",
                    "error": str(e),
                }
            )
        except Exception as e:
            self.logger.warning(
                {
                    "message": f"Knowledge_Provider: Something went wrong while querying {self.id}",
                    "error": str(e),
                    "traceback": e.with_traceback(),
                }
            )

        if response is None:
            response = Response.parse_obj({"message": {}})

        message = response.message
        if message.query_graph is None:
            message = Message(
                query_graph=QueryGraph.parse_obj(request),
                knowledge_graph=KnowledgeGraph.parse_obj({"nodes": {}, "edges": {}}),
                results=Results.parse_obj([]),
                auxiliary_graphs=AuxiliaryGraphs.parse_obj({}),
            )
        if message.knowledge_graph is None:
            message.knowledge_graph = KnowledgeGraph.parse_obj(
                {"nodes": {}, "edges": {}}
            )
        if message.results is None:
            message.results = Results.parse_obj([])
        if message.auxiliary_graphs is None:
            message.auxiliary_graphs = AuxiliaryGraphs.parse_obj({})

        add_source(message, self.id)

        return message


def add_source(message: Message, kp_id):
    """Add provenance annotation to kedges.
    Sources from which we retrieve data add their own prov, we add prov for aragorn."""
    for kedge in message.knowledge_graph.edges.values():
        # create copy of kedge
        new_kedge = kedge.copy()
        new_kedge.sources.add(
            RetrievalSource.parse_obj(
                {
                    "resource_id": "infores:aragorn",
                    "resource_role": "aggregator_knowledge_source",
                    "upstream_resource_ids": [kp_id],
                }
            )
        )
        # update existing kedge
        kedge.update(new_kedge)
    for result in message.results:
        for analysis in result.analyses:
            analysis.resource_id = "infores:aragorn"
            # remove scores generated by KP
            analysis.score = None
            analysis.scoring_method = None
