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
import copy
from itertools import chain
from json.decoder import JSONDecodeError
import logging
from datetime import datetime
from typing import Any, Optional

import aiostream
import httpx
from reasoner_pydantic import QueryGraph, Result, Response
from redis import Redis
from trapi_throttle.throttle import ThrottledServer

from .graph import Graph
from .compatibility import KnowledgePortal, Synonymizer
from .trapi import canonicalize_qgraph, filter_by_qgraph, map_qgraph_curies, merge_messages, merge_results, \
    fill_categories_predicates
from .caching import async_locking_cache
from .query_planner import generate_plan
from .storage import RedisGraph, RedisList, RedisLogHandler
from .kp_registry import Registry
from .config import settings
from .util import KnowledgeProvider, WBMT


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


class Binder():
    """Binder."""

    def __init__(
            self,
            qid: str,
            log_level: int = logging.DEBUG,
            name: Optional[str] = None,
            redis_client: Optional[Redis] = None,
    ):
        """Initialize."""
        self.name = name

        # Set up logger
        handler = RedisLogHandler(f"{qid}:log", redis_client)
        handler.setFormatter(
            ReasonerLogEntryFormatter()
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.addHandler(handler)
        self.logger.debug("Initialized strider worker")

        self.portal = KnowledgePortal(self.logger)

        # Set up DB results objects
        self.kgraph = RedisGraph(f"{qid}:kgraph", redis_client)
        self.results = RedisList(f"{qid}:results", redis_client)

        self.preferred_prefixes = WBMT.entity_prefix_mapping

    async def lookup(
            self,
            qgraph: Graph = None,
            use_cache: bool = True,
    ):
        """Expand from query graph node."""
        # if this is a leaf node, we're done
        if qgraph is None:
            qgraph = Graph(self.qgraph)
        if not qgraph["edges"]:
            yield {"nodes": dict(), "edges": dict()}, {"node_bindings": dict(), "edge_bindings": dict()}
            return
        self.logger.debug(f"Lookup for qgraph: {qgraph}")

        try:
            qedge_id, qedge = next(
                (qedge_id, qedge)
                for qedge_id, qedge in qgraph["edges"].items()
                if any(
                    qnode.get("ids", [])
                    for qnode in (
                        qgraph["nodes"][qedge["subject"]],
                        qgraph["nodes"][qedge["object"]],
                    )
                )
            )
        except StopIteration:
            raise RuntimeError("Cannot find qedge with pinned endpoint")

        qedge = qgraph["edges"][qedge_id]
        onehop = {
            "nodes": {
                key: value
                for key, value in qgraph["nodes"].items()
                if key in (qedge["subject"], qedge["object"])
            },
            "edges": {
                qedge_id: qedge
            }
        }

        generators = [
            self.generate_from_kp(qgraph, onehop, self.kps[kp_id], use_cache=use_cache)
            for kp_id in self.plan[qedge_id]
        ]
        async with aiostream.stream.merge(*generators).stream() as streamer:
            async for output in streamer:
                yield output

    async def generate_from_kp(
        self,
        qgraph,
        onehop_qgraph,
        kp: KnowledgeProvider,
        **kwargs,
    ):
        """Generate one-hop results from KP."""
        print("sending", onehop_qgraph)
        onehop_response = await kp.solve_onehop(
            onehop_qgraph,
            # **kwargs,
        )
        onehop_kgraph = onehop_response["knowledge_graph"]
        onehop_results = onehop_response["results"]
        qedge_id = next(iter(qgraph["edges"].keys()))
        generators = []
        for result in onehop_results:
            # add edge to results and kgraph
            result_kgraph = {
                "nodes": {
                    binding["id"]: onehop_kgraph["nodes"][binding["id"]]
                    for _, bindings in result["node_bindings"].items()
                    for binding in bindings
                },
                "edges": {
                    binding["id"]: onehop_kgraph["edges"][binding["id"]]
                    for _, bindings in result["edge_bindings"].items()
                    for binding in bindings
                },
            }

            # now solve the smaller question
            subqgraph = copy.deepcopy(qgraph)
            # remove edge
            subqgraph["edges"].pop(qedge_id)
            # pin node
            for qnode_id, bindings in result["node_bindings"].items():
                subqgraph["nodes"][qnode_id]["ids"] = [
                    binding["id"]
                    for binding in bindings
                ]
            # remove orphaned nodes
            subqgraph.remove_orphaned()

            generators.append(self.generate_from_result(subqgraph, result_kgraph, result))

        async with aiostream.stream.merge(*generators).stream() as streamer:
            async for result in streamer:
                yield result

    async def generate_from_result(
        self,
        qgraph,
        kgraph,
        result,
        **kwargs,
    ):
        # LOGGER.debug(
        #     "Expanding from result %s...",
        #     result,
        # )
        async for subkgraph, subresult in self.lookup(
            qgraph,
            **kwargs,
        ):
            # combine one-hop with subquery results
            subresult = {
                "node_bindings": {
                    **subresult["node_bindings"],
                    **result["node_bindings"],
                },
                "edge_bindings": {
                    **subresult["edge_bindings"],
                    **result["edge_bindings"],
                },
            }
            subkgraph["nodes"].update(kgraph["nodes"])
            subkgraph["edges"].update(kgraph["edges"])
            yield subkgraph, subresult

    async def get_results(
        self,
        qgraph: dict[str, Any],
        use_cache=True,
        max_results=None,
    ):
        """Get results and kgraph."""
        qgraph = copy.deepcopy(qgraph)
        qgraph = Graph(qgraph)
        # normalize_qgraph(qgraph)

        kgraph = {"nodes": dict(), "edges": dict()}
        results = []
        counter = 0
        async for kgraph_, result_ in self.lookup(qgraph, use_cache=use_cache):
            # aggregate results
            kgraph["nodes"].update(kgraph_["nodes"])
            kgraph["edges"].update(kgraph_["edges"])
            results.append(result_)
            counter += 1
            if max_results is not None and counter >= max_results:
                break

        for kedge in kgraph["edges"].values():
            kedge["attributes"] = [{
                "attribute_type_id": "biolink:knowledge_source",
                "value": f"infores:{self.name}",
            }]

        return kgraph, results

    async def __aenter__(self):
        """Enter context."""
        for tserver in self.portal.tservers.values():
            await tserver.__aenter__()
        return self

    async def __aexit__(self, *args):
        """Exit context."""
        for tserver in self.portal.tservers.values():
            await tserver.__aexit__()

    # pylint: disable=arguments-differ
    async def setup(
            self,
            qgraph: dict,
    ):
        """Set up."""
        # Update qgraph identifiers
        self.qgraph = qgraph
        synonymizer = Synonymizer(self.logger)
        await synonymizer.load_message({"query_graph": self.qgraph})
        curie_map = synonymizer.map(self.preferred_prefixes)
        self.qgraph = map_qgraph_curies(self.qgraph, curie_map, primary=True)
        self.qgraph = canonicalize_qgraph(self.qgraph)

        # Fill in missing categories and predicates using normalizer
        await fill_categories_predicates(self.qgraph, self.logger)

        # Check for constraints
        for node in self.qgraph["nodes"].values():
            if node.get("constraints"):
                raise ValueError("Unable to process query due to constraints")
        for edge in self.qgraph["edges"].values():
            if edge.get("constraints"):
                raise ValueError("Unable to process query due to constraints")

        # Initialize registry
        registry = Registry(settings.kpregistry_url, self.logger)

        self.logger.debug("Generating plan")
        # Generate traversal plan
        self.plan, kps = await generate_plan(
            self.qgraph,
            kp_registry=registry,
            logger=self.logger,
        )

        # extract KP preferred prefixes from plan
        self.kp_preferred_prefixes = dict()
        self.portal.tservers = dict()
        for kp_id, kp in kps.items():
            url = kp["url"][:-5] + "meta_knowledge_graph"
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        url,
                        timeout=10,
                    )
                meta_kg = response.json()
                self.kp_preferred_prefixes[kp_id] = {
                    category: data["id_prefixes"]
                    for category, data in meta_kg["nodes"].items()
                }
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as err:
                self.logger.warning(
                    "Unable to get meta knowledge graph from KP {}: {}".format(
                        kp_id,
                        str(err),
                    ),
                )
                self.kp_preferred_prefixes[kp_id] = dict()
            except JSONDecodeError as err:
                self.logger.warning(
                    "Unable to parse meta knowledge graph from KP {}: {}".format(
                        kp["id"],
                        str(err),
                    ),
                )
                self.kp_preferred_prefixes[kp_id] = dict()
            self.portal.tservers[kp_id] = ThrottledServer(
                kp_id,
                url=kp["url"],
                request_qty=1,
                request_duration=1,
            )
        self.kps = {
            kp_id: KnowledgeProvider(
                details,
                self.portal,
                kp_id,
                self.kp_preferred_prefixes[kp_id],
                self.preferred_prefixes,
            )
            for kp_id, details in kps.items()
        }


def get_kp_request_qgraph(
        qgraph: QueryGraph,
        node_bindings: tuple[tuple[str]],
        qedge_id: str,
) -> Response:
    """Get request to send to KP."""

    # Build request edges
    request_edge = qgraph["edges"][qedge_id].copy()
    request_edge.pop("provided_by", None)
    subject_id = request_edge["subject"]
    request_subject = qgraph["nodes"][subject_id].copy()
    object_id = request_edge["object"]
    request_object = qgraph["nodes"][object_id].copy()

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

    return request_qgraph
