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
from itertools import chain
import json
import logging
from datetime import datetime
from typing import Optional

import httpx
from reasoner_pydantic import QueryGraph, Result, Response
from reasoner_pydantic.results import NodeBinding
from redis import Redis

from .query_planner import generate_plans, Step, NoAnswersError
from .compatibility import KnowledgePortal, Synonymizer
from .trapi import filter_by_qgraph, fix_qgraph, merge_messages, merge_results, \
    fill_categories_predicates
from .worker import Worker
from .caching import async_locking_cache
from .storage import RedisGraph, RedisList, RedisLogHandler
from .kp_registry import Registry
from .config import settings
from .util import ensure_list, standardize_graph_lists, \
    extract_predicate_direction, WBMT, transform_keys

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

        # Update qgraph identifiers
        synonymizer = Synonymizer(self.logger)
        await synonymizer.load_message({"query_graph" : qgraph})
        curie_map = synonymizer.map(self.preferred_prefixes)
        self.qgraph = fix_qgraph(qgraph, curie_map, primary = True)

        # Fill in missing categories and predicates using normalizer
        await fill_categories_predicates(self.qgraph, self.logger)
        standardize_graph_lists(self.qgraph)

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
        plans, kps = await generate_plans(
            self.qgraph,
            kp_registry=registry,
            logger=self.logger)
        self.kps = kps

        if len(plans) == 0:
            self.logger.error("Could not find a plan to traverse query graph")
            raise NoAnswersError()

        self.plan = plans[0]

        # extract KP preferred prefixes from plan
        self.kp_preferred_prefixes = dict()
        for _kps in kps.values():
            for kp in _kps:
                url = kp["url"][:-5] + "meta_knowledge_graph"
                # url = kp["url"][:-5] + "metadata"
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(url)
                except httpx.ConnectError as err:
                    self.logger.warning(
                        "Unable to get meta knowledge graph for KP %s: %s",
                        kp["url"],
                        str(err),
                    )
                    self.kp_preferred_prefixes[kp["url"]] = dict()
                    continue
                meta_kg = response.json()
                self.kp_preferred_prefixes[kp["url"]] = {
                    category: data["id_prefixes"]
                    for category, data in meta_kg["nodes"].items()
                }

        self.logger.debug({
            "plan": self.plan,
        })

        # Initialize results

        first_qedge_id = self.plan[0]
        # first_step = next(iter(self.plan.keys()))
        # pinned_node_id = first_step.source
        first_edge = self.qgraph["edges"][first_qedge_id]
        first_node_id, first_node = next(
            (node_id, node)
            for node_id in (
                first_edge["subject"],
                first_edge["object"],
            )
            if (node := self.qgraph["nodes"][node_id]).get("ids", None)
        )

        # Remove ID from pinned node because we will
        # add it back manually later in the get_kp_request_body
        # function
        curies = first_node.pop("ids")

        # Add partial results for each curie we are given
        for curie in curies:
            result = {
                "node_bindings": {
                    first_node_id: [{
                        "id": curie,
                        "category": first_node["categories"][0],
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
            qedge_id
            for qedge_id in self.plan
            if qedge_id not in bound_edges
        )

    @ async_locking_cache
    async def execute_step(
            self,
            step: Step,
            node_bindings: tuple[tuple[str]],
    ):
        """Fetch results for step."""

        self.logger.debug({
            "description": "Executing step: ",
            "step": step,
        })

        responses = await asyncio.gather(*chain(
            (
                self.portal.fetch(
                    kp["url"],
                    get_kp_request_body(
                        self.qgraph,
                        node_bindings,
                        step,
                        kp,
                    ),
                    self.kp_preferred_prefixes[kp["url"]],
                    self.preferred_prefixes,
                )
                for kp in self.kps[step]
            ), (
                self.portal.fetch(
                    kp["url"],
                    request,
                    self.kp_preferred_prefixes[kp["url"]],
                    self.preferred_prefixes,
                )
                for kp in self.kps[step]
                if (request := get_kp_request_body(
                    self.qgraph,
                    node_bindings,
                    step,
                    kp,
                    invert=True,
                ))
            ),
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
                node_fields = ["categories"],
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
            qedge_id = self.next_step(result["edge_bindings"])
        except StopIteration:
            # Mission accomplished!
            self.results.append(result)
            return

        # execute step
        self.logger.debug({
            "description": "Recieved results from KPs",
            "data": result,
            "step": qedge_id,
        })

        response = await self.execute_step(
            qedge_id,
            tuple(
                (qnode_id, bindings[0]["id"])
                for qnode_id, bindings in result["node_bindings"].items()
            ),
        )

        # process kgraph
        self.kgraph.nodes.merge(response["knowledge_graph"]["nodes"])
        self.kgraph.edges.merge(response["knowledge_graph"]["edges"])

        # process results
        for new_result in response["results"]:
            # queue the results for further processing
            for nbs in new_result["node_bindings"].values():
                for nb in nbs:
                    nb["category"] = response["knowledge_graph"]["nodes"][nb["id"]].get("categories", None)
            await self.put(merge_results([result, new_result]))


def get_kp_request_body(
        qgraph: QueryGraph,
        node_bindings: tuple[tuple[str]],
        qedge_id: str,
        kp: dict,
        invert: bool = False,
) -> Response:
    """Get request to send to KP."""

    # Build request edges
    request_edge = qgraph["edges"][qedge_id].copy()
    subject_id = request_edge["subject"]
    request_subject = qgraph["nodes"][subject_id].copy()
    object_id = request_edge["object"]
    request_object = qgraph["nodes"][object_id].copy()

    # If this is a self edge add a suffix to the target
    # to make sure that we send two nodes to the KP
    if subject_id == object_id:
        object_id += SELF_EDGE_SUFFIX

    if invert:
        predicates = [
            inverse
            for predicate in request_edge.pop("predicates")
            if (inverse := WBMT.predicate_inverse(predicate)) is not None
        ]
        if not predicates:
            # logger.warning("No inverse predicates: %s", str(predicates))
            return None
        request_edge = {
            "subject": request_edge.pop("object"),
            "object": request_edge.pop("subject"),
            "predicates": predicates,
            **request_edge,
        }

    # Build request
    request_qgraph = {
        "nodes": {
            subject_id: request_subject,
            object_id: request_object,
        },
        "edges": {
            qedge_id: request_edge
        },
    }

    for qnode_key, bound_id in node_bindings:
        if qnode_key not in request_qgraph["nodes"]:
            continue
        request_qgraph["nodes"][qnode_key]["ids"] = [bound_id]

    return {"message": {"query_graph": request_qgraph}}
