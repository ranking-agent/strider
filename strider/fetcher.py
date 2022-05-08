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
from collections import defaultdict
from collections.abc import Iterable
import copy
from itertools import chain
from json.decoder import JSONDecodeError
import logging
from datetime import datetime

from strider.constraints import enforce_constraints
from typing import Any, Callable, Optional

import aiostream
import httpx
import pydantic
from reasoner_pydantic import QueryGraph, Result, Response, MetaKnowledgeGraph, Message
from redis import Redis

from .trapi_throttle.throttle import ThrottledServer
from .graph import Graph
from .compatibility import KnowledgePortal, Synonymizer
from .trapi import (
    get_canonical_qgraphs,
    filter_by_qgraph,
    get_curies,
    map_qgraph_curies,
    fill_categories_predicates,
)
from .query_planner import generate_plan, get_next_qedge
from .storage import RedisGraph, RedisList, RedisLogHandler
from .kp_registry import Registry
from .config import settings
from .util import KnowledgeProvider, WBMT, batch, setup_logging


class ReasonerLogEntryFormatter(logging.Formatter):
    """Format to match Reasoner API LogEntry"""

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
        iso_timestamp = datetime.utcfromtimestamp(record.created).isoformat()
        log_entry["timestamp"] = iso_timestamp

        # Add level
        log_entry["level"] = record.levelname

        return log_entry

setup_logging()
_logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s: %(levelname)s/%(name)s]: %(message)s")
sh.setFormatter(formatter)
_logger.addHandler(sh)


class Binder:
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
        handler.setFormatter(ReasonerLogEntryFormatter())
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.addHandler(handler)

        self.synonymizer = Synonymizer(self.logger)
        self.portal = KnowledgePortal(self.synonymizer, self.logger)

        # Set up DB results objects
        self.kgraph = RedisGraph(f"{qid}:kgraph", redis_client)
        self.results = RedisList(f"{qid}:results", redis_client)

        self.preferred_prefixes = WBMT.entity_prefix_mapping

        self.lookup_id = 0
        self.generate_from_kp_id = 0
        self.generate_from_result_id = 0

    async def lookup(
        self,
        qgraph: Graph = None,
        callstack: list = None
    ):
        """Expand from query graph node."""
        # if this is a leaf node, we're done
        pid = self.lookup_id
        self.lookup_id = self.lookup_id + 1
        if not callstack:
            callstack = [f"lookup_id: {pid}"]
        else:
            callstack.append(f"lookup_id: {pid}")
        if qgraph is None:
            qgraph = Graph(self.qgraph)
        if not qgraph["edges"]:
            yield {"nodes": dict(), "edges": dict()}, {
                "node_bindings": dict(),
                "edge_bindings": dict(),
            }
            self.logger.debug("Finishing lookup")
            return
        self.logger.debug(f"Lookup for qgraph: {qgraph}")

        try:
            qedge_id, qedge = get_next_qedge(qgraph)
        except StopIteration:
            raise RuntimeError("Cannot find qedge with pinned endpoint")

        qedge = qgraph["edges"][qedge_id]
        onehop = {
            "nodes": {
                key: value
                for key, value in qgraph["nodes"].items()
                if key in (qedge["subject"], qedge["object"])
            },
            "edges": {qedge_id: qedge},
        }

        self.logger.debug(
            {
                "state": "intializing generate_from_kp",
                "lookup_id": pid,
                "callstack": callstack
            }
        )

        generators = [
            self.generate_from_kp(qgraph, onehop, self.kps[kp_id], callstack=copy.deepcopy(callstack))
            for kp_id in self.plan[qedge_id]
        ]

        self.logger.debug(
            {
                "state": " calling generators lookup",
                "qgraph": qgraph,
                "onehop": onehop,
                "kps": [
                    self.kps[kp_id].id for kp_id in self.plan[qedge_id]
                ],
                "lookup_id": pid,
                "callstack": callstack
            }
        )

        async with aiostream.stream.merge(*generators).stream() as streamer:
            self.logger.debug(
                {
                    "state": "merging streamer from lookup",
                    "lookup_id": pid,
                    "callstack": callstack
                }
            )
            async for output in streamer:
                self.logger.debug(
                    {
                        "state": "yielding lookup",
                        "lookup_id": pid,
                        "callstack": callstack
                    }
                )
                yield output
            self.logger.debug(
                {
                    "state": "finishing streamer from lookup",
                    "lookup_id": pid,
                    "callstack": callstack
                }
            )
        self.logger.debug(
            {
                "state": "finishing lookup",
                "lookup_id": pid,
                "callstack": callstack
            }
        )

    async def generate_from_kp(
        self,
        qgraph,
        onehop_qgraph,
        kp: KnowledgeProvider,
        callstack: list = None
    ):
        pid = self.generate_from_kp_id
        self.generate_from_kp_id = self.generate_from_kp_id + 1
        if not callstack:
            callstack = [f"generate_from_kp_id: {pid}"]
        else:
            callstack.append(f"generate_from_kp_id: {pid}")
        self.logger.debug(
            {
                "state": "Starting generate_from_kp",
                "generate_from_KP_id": pid,
                "callstack": callstack
            }
        )
        """Generate one-hop results from KP."""
        onehop_response = await kp.solve_onehop(
            onehop_qgraph,
        )
        self.logger.debug(
            {
                "state": "Generated one hop response",
                "generate_from_KP_id": pid,
                "callstack": callstack
            }
        )
        onehop_response = enforce_constraints(onehop_response)
        onehop_kgraph = onehop_response["knowledge_graph"]
        onehop_results = onehop_response["results"]
        qedge_id = next(iter(onehop_qgraph["edges"].keys()))
        generators = []

        self.logger.debug(
            {
                "state": "Get next edge",
                "generate_from_KP_id": pid,
                "callstack": callstack
            }
        )

        if onehop_results:
            self.logger.debug(
                {
                    "state": "entered if",
                    "generate_from_KP_id": pid,
                    "callstack": callstack
                }
            )
            subqgraph = copy.deepcopy(qgraph)
            # remove edge
            subqgraph["edges"].pop(qedge_id)
            # remove orphaned nodes
            subqgraph.remove_orphaned()
        else:
            self.logger.debug(
                {
                    "state": "did not enter",
                    "generate_from_KP_id": pid,
                    "callstack": callstack
                }
            )
        for batch_results in batch(onehop_results, 1_000_000):
            self.logger.debug(
                {
                    "state": "Batched results",
                    "generate_from_KP_id": pid,
                    "callstack": callstack
                }
            )
            result_map = defaultdict(list)
            for result in batch_results:
                self.logger.debug(
                    {
                        "state": "single result in batch",
                        "generate_from_KP_id": pid,
                        "callstack": callstack
                    }
                )
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

                # pin nodes
                for qnode_id, bindings in result["node_bindings"].items():
                    if qnode_id not in subqgraph["nodes"]:
                        continue
                    subqgraph["nodes"][qnode_id]["ids"] = (
                        subqgraph["nodes"][qnode_id].get("ids") or []
                    ) + [binding["id"] for binding in bindings]
                qnode_ids = set(subqgraph["nodes"].keys()) & set(
                    result["node_bindings"].keys()
                )
                key_fcn = lambda res: tuple(
                    (
                        qnode_id,
                        tuple(
                            binding["id"] for binding in bindings  # probably only one
                        ),
                    )
                    for qnode_id, bindings in res["node_bindings"].items()
                    if qnode_id in qnode_ids
                )
                result_map[key_fcn(result)].append((result, result_kgraph))

            self.logger.debug(
                {
                    "state": "Initiating generators",
                    "generate_from_KP_id": pid,
                    "callstack": callstack
                }
            )

            generators.append(
                self.generate_from_result(
                    copy.deepcopy(subqgraph),
                    lambda result: result_map[key_fcn(result)],
                    callstack=copy.deepcopy(callstack)
                )
            )
        
        self.logger.debug(
            {
                "state": "calling generators from generate_from_kp",
                "generate_from_KP_id": pid,
                "callstack": callstack
            }
        )

        async with aiostream.stream.merge(*generators).stream() as streamer:
            self.logger.debug(
                {
                    "state": "merging streamer from generate_from_kp",
                    "generate_from_KP_id": pid,
                    "callstack": callstack
                }
            )
            async for result in streamer:
                self.logger.debug(
                    {
                        "state": "Yielding from generate_from_kp",
                        "generate_from_KP_id": pid,
                        "callstack": callstack
                    }
                )
                yield result
            self.logger.debug(
                {
                    "state": "finishing streamer generate_from_kp",
                    "generate_from_KP_id": pid,
                    "callstack": callstack
                }
            )
        
        self.logger.debug(
            {
                "state": "finishing generate_from_kp",
                "generate_from_KP_id": pid,
                "callstack": callstack
            }
        )
        return

    async def generate_from_result(
        self,
        qgraph,
        get_results: Callable[[dict], Iterable[tuple[dict, dict]]],
        callstack: list = None
    ):
        # LOGGER.debug(
        #     "Expanding from result %s...",
        #     result,
        # )
        pid = self.generate_from_result_id
        self.generate_from_result_id = self.generate_from_result_id + 1
        if not callstack:
            callstack = [f"generate_from_result_id: {pid}"]
        else:
            callstack.append(f"generate_from_result_id: {pid}")
        self.logger.debug(
            {
                "state": "Starting generate_from_result",
                "generate_from_result_id": pid
            }
        )
        async for subkgraph, subresult in self.lookup(
            qgraph, callstack=copy.deepcopy(callstack)
        ):
            self.logger.debug(
                {
                    "state": "Generating with subk",
                    "generate_from_result_id": pid
                }
            )
            for result, kgraph in get_results(subresult):
                # combine one-hop with subquery results
                self.logger.debug(
                    {
                        "state": "generating with subresult",
                        "generate_from_result_id": pid
                    }
                )
                new_subresult = {
                    "node_bindings": {
                        **subresult["node_bindings"],
                        **result["node_bindings"],
                    },
                    "edge_bindings": {
                        **subresult["edge_bindings"],
                        **result["edge_bindings"],
                    },
                }
                new_subkgraph = copy.deepcopy(subkgraph)
                new_subkgraph["nodes"].update(kgraph["nodes"])
                new_subkgraph["edges"].update(kgraph["edges"])
                self.logger.debug(
                    {
                        "state": "Yielding generate_from_result",
                        "generate_from_result_id": pid
                    }
                )
                yield new_subkgraph, new_subresult

    async def __aenter__(self):
        """Enter context."""
        for tserver in self.portal.tservers.values():
            await tserver.__aenter__()
        return self

    async def __aexit__(self, *args):
        """Exit context."""
        for tserver in self.portal.tservers.values():
            await tserver.__aexit__()

    def get_processor(self, preferred_prefixes):
        """Get processor."""

        async def processor(request, logger: logging.Logger = None):
            """Map message CURIE prefixes."""
            if logger is None:
                logger = self.logger
            await self.portal.map_prefixes(request.message, preferred_prefixes)

        return processor

    # pylint: disable=arguments-differ
    async def setup(
        self,
        qgraph: dict,
    ):
        """Set up."""

        # Update qgraph identifiers
        message = Message.parse_obj({"query_graph": qgraph})
        curies = get_curies(message)
        await self.synonymizer.load_curies(*curies)
        curie_map = self.synonymizer.map(curies, self.preferred_prefixes)
        map_qgraph_curies(message.query_graph, curie_map, primary=True)

        self.qgraph = message.query_graph.dict()

        # Fill in missing categories and predicates using normalizer
        await fill_categories_predicates(self.qgraph, self.logger)

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
                response.raise_for_status()
                meta_kg = response.json()
                MetaKnowledgeGraph.parse_obj(meta_kg)
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
            except httpx.HTTPStatusError as e:
                self.logger.warning(
                    "Received error response from /meta_knowledge_graph for KP {}: {}".format(
                        kp_id,
                        e.response.text,
                    ),
                )
                self.kp_preferred_prefixes[kp_id] = dict()
            except JSONDecodeError as err:
                self.logger.warning(
                    "Unable to parse meta knowledge graph from KP {}: {}".format(
                        kp_id,
                        str(err),
                    ),
                )
                self.kp_preferred_prefixes[kp_id] = dict()
            except pydantic.ValidationError as err:
                self.logger.warning(
                    "Meta knowledge graph from KP {} is non-compliant: {}".format(
                        kp_id,
                        str(err),
                    ),
                )
                self.kp_preferred_prefixes[kp_id] = dict()
            except Exception as err:
                self.logger.warning(
                    "Something went wrong while parsing meta knowledge graph from KP {}: {}".format(
                        kp_id,
                        str(err),
                    ),
                )
                self.kp_preferred_prefixes[kp_id] = dict()

            self.portal.tservers[kp_id] = ThrottledServer(
                kp_id,
                url=kp["url"],
                request_qty=1,
                request_duration=1,
                preproc=self.get_processor(self.kp_preferred_prefixes[kp_id]),
                postproc=self.get_processor(self.preferred_prefixes),
                logger=self.logger,
            )
        self.kps = {
            kp_id: KnowledgeProvider(
                details,
                self.portal,
                kp_id,
            )
            for kp_id, details in kps.items()
        }
