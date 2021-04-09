"""Fetcher 2.0.

 .oooooo..o     .             o8o        .o8
d8P'    `Y8   .o8             `"'       "888
Y88bo.      .o888oo oooo d8b oooo   .oooo888   .ooooo.  oooo d8b
 `"Y8888o.    888   `888""8P `888  d88' `888  d88' `88b `888""8P
     `"Y88b   888    888      888  888   888  888ooo888  888
oo     .d8P   888 .  888      888  888   888  888    .o  888
8""88888P'    "888" d888b    o888o `Y8bod88P" `Y8bod8P' d888b

"""
import asyncio
from collections.abc import Iterable
import logging
import json
from datetime import datetime
from typing import Optional
import jsonpickle

from reasoner_pydantic import QueryGraph, Result, Response

from .query_planner import generate_plans, Step, NoAnswersError
from .compatibility import KnowledgePortal
from .trapi import merge_messages, merge_results, \
    fill_categories_predicates
from .worker import Worker
from .caching import async_locking_cache
from .storage import RedisGraph, RedisList, RedisLogHandler
from .kp_registry import Registry
from .config import settings
from .util import ensure_list, standardize_graph_lists, \
    extract_predicate_direction, WrappedBMT

# Initialize registry
registry = Registry(settings.kpregistry_url)

# Initialize BMT
WBMT = WrappedBMT()


class ReasonerLogEntryFormatter(logging.Formatter):
    """ Format to match Reasoner API LogEntry """

    def format(self, record):
        # If given a string, convert to dict
        if isinstance(record.msg, str):
            record.msg = dict(message=record.msg)

        iso_timestamp = datetime.utcfromtimestamp(
            record.created
        ).isoformat()

        # If given a code, set a code
        code = None
        if 'code' in record.msg:
            code = record.msg['code']
            del record.msg['code']

        # Extra fields go in the message
        record.msg['line_number'] = record.lineno
        if record.exc_info:
            record.msg['exception_info'] = self.formatException(
                record.exc_info
            )

        return dict(
            code=code,
            message=jsonpickle.encode(
                record.msg,
            ),
            level=record.levelname,
            timestamp=iso_timestamp,
        )


class StriderWorker(Worker):
    """Async worker to process query"""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        self.plan: dict[Step, list]
        self.preferred_prefixes: dict[str, list[str]]
        self.qgraph: RedisGraph
        self.kgraph: RedisGraph
        self.logger: logging.Logger
        self.results: list[Result] = []
        self.portal: KnowledgePortal = None
        super().__init__(*args, **kwargs)

    # pylint: disable=arguments-differ
    async def setup(
            self,
            qid: str,
            log_level: int,
    ):
        """Set up."""
        # Set up DB results objects
        self.kgraph = RedisGraph(f"{qid}:kgraph")
        self.results = RedisList(f"{qid}:results")

        # Set up logger
        handler = RedisLogHandler(f"{qid}:log")
        handler.setFormatter(
            ReasonerLogEntryFormatter()
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.addHandler(handler)

        self.logger.debug("Initialized strider worker")

        if not self.portal:
            self.portal = KnowledgePortal(self.logger)

        # Pull query graph from Redis
        qgraph = RedisGraph(f"{qid}:qgraph").get()

        # Use BMT for preferred prefixes
        self.preferred_prefixes = WBMT.entity_prefix_mapping

        # fix qgraph
        self.qgraph = (await self.portal.map_prefixes(
            {"query_graph": qgraph},
            self.preferred_prefixes,
        ))["query_graph"]

        # Fill in missing categories and predicates using normalizer
        await fill_categories_predicates(self.qgraph, self.logger)
        standardize_graph_lists(self.qgraph)

        # Replace biolink:Proten with biolink:GeneOrGeneProduct
        for node in self.qgraph["nodes"].values():
            categories = node.get("category", [])
            for category in categories:
                if category == "biolink:Protein":
                    category = "biolink:GeneOrGeneProduct"

    async def generate_plan(self):
        """
        Use the self.qgraph object to generate a plan and store it
        in the self.plan object.

        Also adds a partial result to the queue so that the
        run method can be called.
        """
        self.logger.debug("Generating plan")
        # Generate traversal plan
        plans = await generate_plans(
            self.qgraph,
            kp_registry=registry,
            logger=self.logger)

        if len(plans) == 0:
            self.logger.error("Could not find a plan to traverse query graph")
            raise NoAnswersError()

        self.plan = plans[0]

        self.logger.debug({"plan": self.plan})

        # add partial results for each curie that we are given
        for qnode_id, qnode in self.qgraph["nodes"].items():

            # Remove ID from query graph because we will
            # add it back manually later in the get_kp_request_body
            # function
            curies = qnode.pop("id", None)
            if not curies:
                continue

            for curie in curies:
                result = {
                    "node_bindings": {
                        qnode_id: [{
                            "id": curie,
                            "category": qnode["category"][0],
                        }]
                    },
                    "edge_bindings": {},
                }
                await self.put(result)

    def next_step(
            self,
            bound_edges: Iterable[str],
    ):
        """Get next step in plan."""
        return next(
            step
            for step in self.plan
            if step.edge not in bound_edges
        )

    @ async_locking_cache
    async def execute_step(
            self,
            step: Step,
            curie: str,
            category: Optional[str],
    ):
        """Fetch results for step."""

        self.logger.debug({
            "description": "Executing step: ",
            "step": self.plan[step],
        })

        responses = await asyncio.gather(*(
            self.portal.fetch(
                kp["url"],
                get_kp_request_body(
                    self.qgraph,
                    curie,
                    step,
                    kp,
                ),
                kp["preferred_prefixes"],
                self.preferred_prefixes,
            )
            for kp in self.plan[step]
            if kp["source_category"] in category
            or kp["target_category"] in category
        ))
        return merge_messages(responses)

    async def on_message(
            self,
            result: Result,
    ):
        """Process partial result."""
        # find the next step in the plan
        try:
            step = self.next_step(result["edge_bindings"])
        except StopIteration:
            # Mission accomplished!
            self.results.append(result)
            return

        # execute step
        self.logger.debug({
            "description": "Recieved results from KPs",
            "data": result,
            "step": step,
        })

        category_list = ensure_list(
            result["node_bindings"][step.source][0].get("category", [])
        )

        response = await self.execute_step(
            step,
            result["node_bindings"][step.source][0]["id"],
            category=tuple(category_list),
        )

        # process kgraph
        self.kgraph.nodes.merge(response["knowledge_graph"]["nodes"])
        self.kgraph.edges.merge(response["knowledge_graph"]["edges"])

        # process results
        for new_result in response["results"]:
            # queue the results for further processing
            for nbs in new_result["node_bindings"].values():
                for nb in nbs:
                    nb["category"] = response["knowledge_graph"]["nodes"][nb["id"]]["category"]
            await self.put(merge_results([result, new_result]))


def get_kp_request_body(
        qgraph: QueryGraph,
        curie: str,
        step: Step,
        kp: dict,
) -> Response:
    """Get request to send to KP."""

    predicate, reverse = \
        extract_predicate_direction(kp["edge_predicate"])

    # Build request edges
    request_edge = qgraph["edges"][step.edge].copy()
    request_source = qgraph["nodes"][step.source].copy()
    request_target = qgraph["nodes"][step.target].copy()

    # Update request properties to match what the KP expects
    request_edge["predicate"] = predicate
    request_source["category"] = kp["source_category"]
    request_target["category"] = kp["target_category"]

    # If we have a reversed predicate (<-predicate-)
    # then we look up from object to subject
    if reverse:
        request_edge["subject"] = step.target
        request_edge["object"] = step.source
    else:
        request_edge["subject"] = step.source
        request_edge["object"] = step.target

    # Fill in the current curie
    request_source["id"] = curie

    # Build request
    request_qgraph = {
        "nodes": {
            step.source: request_source,
            step.target: request_target,
        },
        "edges": {
            step.edge: request_edge
        },
    }

    return {"message": {"query_graph": request_qgraph}}
