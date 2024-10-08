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
import traceback

from .throttle import ThrottledServer
from .utils import (
    WBMT,
    elide_curies,
    get_curies,
    remove_null_values,
    log_response,
    log_request,
)
from .trapi import apply_curie_map, filter_message, convert_subclasses_to_aux_graphs
from .normalizer import Normalizer
from .config import settings


class KnowledgeProvider:
    """Knowledge provider."""

    def __init__(
        self,
        kp_id,
        kp,
        logger,
        parameters: dict = {},
        information_content_threshold: int = settings.information_content_threshold,
        *args,
        **kwargs,
    ):
        """Initialize."""
        # Use kp timeout given in the message, otherwise use env variable
        kp_timeout = parameters.get("kp_timeout")
        self.timeout = kp_timeout if type(kp_timeout) is int else settings.kp_timeout
        self.id = kp_id
        self.logger = logger
        self.throttle = ThrottledServer(
            kp_id,
            url=kp["url"],
            logger=logger,
            preproc=self.get_preprocessor(kp["details"]["preferred_prefixes"]),
            postproc=self.get_postprocessor(WBMT.entity_prefix_mapping),
            parameters=parameters,
            kp_timeout=self.timeout,
            *args,
            *kwargs,
        )
        self.normalizer = Normalizer(logger=logger)
        self.information_content_threshold = information_content_threshold

    def get_preprocessor(self, preferred_prefixes):
        """Get pre processor."""

        async def processor(request):
            """Map message CURIE prefixes."""
            await self.map_prefixes(request.message, preferred_prefixes)

        return processor

    def get_postprocessor(self, preferred_prefixes):
        """Get post processor."""

        async def processor(response, last_hop: bool):
            """Map message CURIE prefixes."""
            await self.map_prefixes(response.message, preferred_prefixes)
            filter_message(
                response.message,
                self.normalizer.curie_map,
                self.logger,
                self.information_content_threshold,
                last_hop,
            )
            convert_subclasses_to_aux_graphs(
                response.message,
                self.id,
                self.logger,
            )

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
            apply_curie_map(message, curie_map, self.id, self.logger)

    async def get_mcq_uuid(self, curies: list[str]) -> str:
        """Given a list of curies, get the MCQ uuid from NN."""
        uuid = await self.normalizer.get_mcq_uuid(curies)
        return uuid

    async def solve_onehop(
        self, request, bypass_cache: bool, call_stack: list, last_hop: bool
    ):
        """Solve one-hop query."""
        request = remove_null_values(request.dict())
        response = None
        try:
            response = await self.throttle.query(
                {"message": request},
                bypass_cache,
                call_stack,
                last_hop,
            )
        except asyncio.TimeoutError as e:
            self.logger.warning(
                {
                    "message": f"{self.id} took > {self.timeout} seconds to respond",
                    "error": str(e),
                    "request": elide_curies(request),
                }
            )
        except httpx.ReadTimeout as e:
            self.logger.warning(
                {
                    "message": f"{self.id} took > {self.timeout} seconds to respond",
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
                    "traceback": traceback.format_exc(),
                }
            )

        if response is None:
            response = Response.parse_obj({"message": {}})

        message = response.message
        if message.query_graph is None:
            message = Message(
                query_graph=QueryGraph.parse_obj(request["query_graph"]),
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
