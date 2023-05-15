"""Fetcher 2.0.

 .oooooo..o     .             o8o        .o8
d8P'    `Y8   .o8             `"'       "888
Y88bo.      .o888oo oooo d8b oooo   .oooo888   .ooooo.  oooo d8b
 `"Y8888o.    888   `888""8P `888  d88' `888  d88' `88b `888""8P
     `"Y88b   888    888      888  888   888  888ooo888  888
oo     .d8P   888 .  888      888  888   888  888    .o  888
8""88888P'    "888" d888b    o888o `Y8bod88P" `Y8bod8P' d888b

"""
from collections import defaultdict
from collections.abc import Iterable
import copy
import json
import logging

from strider.constraints import enforce_constraints
from typing import Callable, List

import aiostream
from kp_registry import Registry
from reasoner_pydantic import Message

from .trapi_throttle.throttle import ThrottledServer
from .graph import Graph
from .compatibility import KnowledgePortal, Synonymizer
from .trapi import (
    get_curies,
    map_qgraph_curies,
    fill_categories_predicates,
)
from .query_planner import generate_plan, get_next_qedge
from .config import settings
from .util import (
    KnowledgeProvider,
    WBMT,
    batch,
    elide_curies,
)
from .caching import get_kp_onehop, save_kp_onehop


class Binder:
    """Binder."""

    def __init__(self, logger):
        """Initialize."""
        self.logger = logger
        self.synonymizer = Synonymizer(self.logger)
        self.portal = KnowledgePortal(self.synonymizer, self.logger)

        self.preferred_prefixes = WBMT.entity_prefix_mapping

    async def lookup(
        self,
        qgraph: Graph = None,
        call_stack: List = [],
    ):
        """Expand from query graph node."""
        # if this is a leaf node, we're done
        if qgraph is None:
            qgraph = Graph(self.qgraph)
        if not qgraph["edges"]:
            self.logger.info(f"Finished call stack: {(', ').join(call_stack)}")
            yield {"nodes": dict(), "edges": dict()}, {
                "node_bindings": dict(),
                "analyses": [],
            }
            return

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

        generators = [
            self.generate_from_kp(
                qgraph, onehop, self.kps[kp_id], copy.deepcopy(call_stack)
            )
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
        call_stack: List,
    ):
        """Generate one-hop results from KP."""
        # keep track of call stack for each kp plan branch
        call_stack.append(kp.id)
        self.logger.info(f"Current call stack: {(', ').join(call_stack)}")
        onehop_response = await get_kp_onehop(kp.id, onehop_qgraph)
        if onehop_response is not None:
            self.logger.info(f"[{kp.id}]: Got onehop from cache")
        if onehop_response is None and not settings.offline_mode:
            # onehop not in cache, have to go get response
            self.logger.info(
                f"[{kp.id}] Need to get results for: {json.dumps(elide_curies(onehop_qgraph))}"
            )
            onehop_response = await kp.solve_onehop(
                onehop_qgraph,
            )
            await save_kp_onehop(kp.id, onehop_qgraph, onehop_response)
        if onehop_response is None and settings.offline_mode:
            self.logger.info(
                f"[{kp.id}] Didn't get anything back from cache in offline mode."
            )
            # Offline mode and query wasn't cached, just continue
            onehop_results = []
        else:
            onehop_response = enforce_constraints(onehop_response)
            onehop_kgraph = onehop_response["knowledge_graph"]
            onehop_results = onehop_response["results"]
            qedge_id = next(iter(onehop_qgraph["edges"].keys()))
        generators = []

        if onehop_results:
            subqgraph = copy.deepcopy(qgraph)
            # remove edge
            subqgraph["edges"].pop(qedge_id)
            # remove orphaned nodes
            subqgraph.remove_orphaned()
        else:
            self.logger.info(
                f"Ending call stack with no results: {(', ').join(call_stack)}"
            )
        for batch_results in batch(onehop_results, 1_000_000):
            result_map = defaultdict(list)
            for result in batch_results:
                # add edge to results and kgraph
                result_kgraph = {
                    "nodes": {
                        binding["id"]: onehop_kgraph["nodes"][binding["id"]]
                        for _, bindings in result["node_bindings"].items()
                        for binding in bindings
                    },
                    "edges": {
                        binding["id"]: onehop_kgraph["edges"][binding["id"]]
                        for analysis in result.get("analyses", [])
                        for _, bindings in analysis["edge_bindings"].items()
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

            generators.append(
                self.generate_from_result(
                    copy.deepcopy(subqgraph),
                    lambda result: result_map[key_fcn(result)],
                    call_stack,
                )
            )

        async with aiostream.stream.merge(*generators).stream() as streamer:
            async for result in streamer:
                yield result

    async def generate_from_result(
        self,
        qgraph,
        get_results: Callable[[dict], Iterable[tuple[dict, dict]]],
        call_stack: List,
    ):
        async for subkgraph, subresult in self.lookup(
            qgraph,
            call_stack,
        ):
            for result, kgraph in get_results(subresult):
                # combine one-hop with subquery results
                new_subresult = {
                    "node_bindings": {
                        **subresult["node_bindings"],
                        **result["node_bindings"],
                    },
                    "analyses": [
                        *subresult["analyses"],
                        *result["analyses"],
                        # reconsider
                    ],
                }
                new_subkgraph = copy.deepcopy(subkgraph)
                new_subkgraph["nodes"].update(kgraph["nodes"])
                new_subkgraph["edges"].update(kgraph["edges"])
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
        registry: Registry,
    ):
        """Set up."""

        # Update qgraph identifiers
        message = Message.parse_obj({"query_graph": qgraph})
        curies = get_curies(message)
        if len(curies):
            await self.synonymizer.load_curies(*curies)
            curie_map = self.synonymizer.map(curies, self.preferred_prefixes)
            map_qgraph_curies(message.query_graph, curie_map, primary=True)

        self.qgraph = message.query_graph.dict()

        # Fill in missing categories and predicates using normalizer
        await fill_categories_predicates(self.qgraph, self.logger)

        # Generate traversal plan
        self.plan, kps = await generate_plan(
            self.qgraph,
            logger=self.logger,
            registry=registry,
        )
        self.logger.info(f"Generated query plan: {self.plan}")

        # extract KP preferred prefixes from plan
        self.kp_preferred_prefixes = dict()
        self.portal.tservers = dict()
        for kp_id, kp in kps.items():
            try:
                self.kp_preferred_prefixes[kp_id] = kp["details"]["preferred_prefixes"]
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
