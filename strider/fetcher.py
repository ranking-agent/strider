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
import json
import logging
from datetime import datetime
from typing import Optional

from reasoner_pydantic import QueryGraph, Result, Response
from redis import Redis

from .query_planner import generate_plans, Step, NoAnswersError
from .compatibility import KnowledgePortal
from .trapi import filter_by_qgraph, merge_messages, merge_results, \
    fill_categories_predicates
from .worker import Worker
from .caching import async_locking_cache
from .storage import RedisGraph, RedisList, RedisLogHandler
from .kp_registry import Registry
from .config import settings
from .util import ensure_list, standardize_graph_lists, \
    extract_predicate_direction, WrappedBMT, transform_keys


# Initialize BMT
WBMT = WrappedBMT()

SELF_EDGE_SUFFIX = ".self"

class ReasonerLogEntryFormatter(logging.Formatter):
    """ Format to match Reasoner API LogEntry """

    def format(self, record):
        log_entry = {}

        # If given a string use that as the message
        if isinstance(record.msg, str):
            log_entry["message"] = record.msg

        # If given a dict, just use that as the log entry
        # Make sure everything is serializeable
        if isinstance(record.msg, dict):
            log_entry |= record.msg

        # Add timestamp
        iso_timestamp = datetime.utcfromtimestamp(
            record.created
        ).isoformat()
        log_entry["timestamp"] = iso_timestamp

        # Add level
        log_entry["level"] = record.levelname

        return log_entry


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
            redis_client: Optional[Redis] = None,
    ):
        """Set up."""
        # Set up DB results objects
        self.kgraph = RedisGraph(f"{qid}:kgraph", redis_client)
        self.results = RedisList(f"{qid}:results", redis_client)

        # Set up logger
        handler = RedisLogHandler(f"{qid}:log", redis_client)
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
        qgraph = RedisGraph(f"{qid}:qgraph", redis_client).get()

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

        # Replace biolink:Protein with biolink:GeneOrGeneProduct
        for node in self.qgraph["nodes"].values():
            if not node.get("category"):
                continue
            node["category"] = [
                "biolink:GeneOrGeneProduct"
                if category == "biolink:Protein" else category
                for category in node["category"]
            ]

        # Check for constraints
        for node in self.qgraph["nodes"].values():
            if node["constraints"]:
                raise ValueError("Unable to process query due to constraints")
        for edge in self.qgraph["edges"].values():
            if edge["constraints"]:
                raise ValueError("Unable to process query due to constraints")

    async def generate_plan(self):
        """
        Use the self.qgraph object to generate a plan and store it
        in the self.plan object.

        Also adds a partial result to the queue so that the
        run method can be called.
        """

        # Initialize registry
        registry = Registry(settings.kpregistry_url, self.logger)

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

        self.logger.debug({
            "plan": transform_keys(
                self.plan,
                lambda step: f"{step.source}-{step.edge}-{step.target}"
            )
        })

        # Initialize results

        first_step = next(iter(self.plan.keys()))
        pinned_node_id = first_step.source
        pinned_node = self.qgraph["nodes"][pinned_node_id]

        # Remove ID from pinned node because we will
        # add it back manually later in the get_kp_request_body
        # function
        curies = pinned_node.pop("id")

        # Add partial results for each curie we are given
        for curie in curies:
            result = {
                "node_bindings": {
                    pinned_node_id: [{
                        "id": curie,
                        "category": pinned_node["category"][0],
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

        # Valid categories for the next KP that we contact
        # include descendants of any types that we received
        all_valid_categories = []
        for c in category:
            all_valid_categories.extend(WBMT.get_descendants(c))

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
            if kp["source_category"] in all_valid_categories
            or kp["target_category"] in all_valid_categories
        ))

        for response in responses:
            # Fix self edges to point to the correct node
            # (without .self suffix)
            for result in response["results"]:
                for qg_id in list(result["node_bindings"].keys()):
                    if qg_id.endswith(SELF_EDGE_SUFFIX):
                        nb_list = result["node_bindings"].pop(qg_id)
                        result["node_bindings"][qg_id[:-len(SELF_EDGE_SUFFIX)]] = nb_list

            standardize_graph_lists(
                response["knowledge_graph"],
                node_fields = ["category"],
                edge_fields = [],
            )
            filter_by_qgraph(response, self.qgraph)

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
            "step": f"{step.source}-{step.edge}-{step.target}",
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

    # Fill in the current curie
    request_source["id"] = curie

    source = step.source
    target = step.target

    # If this is a self edge add a suffix to the target
    # to make sure that we send two nodes to the KP
    if source == target:
        target += SELF_EDGE_SUFFIX

    # If we have a reversed predicate (<-predicate-)
    # then we look up from object to subject
    if reverse:
        request_edge["subject"] = target
        request_edge["object"] = source
    else:
        request_edge["subject"] = source
        request_edge["object"] = target

    # Remove ID from the target if present
    # because KPs can't handle it properly
    request_target.pop("id", None)

    # Build request
    request_qgraph = {
        "nodes": {
            source: request_source,
            target: request_target,
        },
        "edges": {
            step.edge: request_edge
        },
    }

    return {"message": {"query_graph": request_qgraph}}
