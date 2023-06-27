"""Fetcher 2.0."""
from collections import defaultdict
from collections.abc import Iterable
import copy
import json
import logging

from strider.constraints import enforce_constraints
from typing import Callable, List

import aiostream
from kp_registry import Registry
from reasoner_pydantic import (
    Message,
    QueryGraph,
    KnowledgeGraph,
    AuxiliaryGraphs,
    Result,
)

from .graph import Graph
from .normalizer import Normalizer
from .knowledge_provider import KnowledgeProvider
from .trapi import (
    map_qgraph_curies,
    fill_categories_predicates,
)
from .query_planner import generate_plan, get_next_qedge
from .config import settings
from .utils import (
    WBMT,
    batch,
    elide_curies,
    get_curies,
)
from .caching import get_kp_onehop, save_kp_onehop


class Fetcher:
    """Strider lookup engine.

    The lookup process involves three recursive async generators:
        lookup -> generate_from_kp -> generate_from_result -> lookup
    Steps:
    - Given a query, develop a kp plan for each edge. KPs are found based on their meta_knowledge_graphs
    - Grab the weightiest edge from the query and send that onehop to kps according to the plan
    - Once a response is received, each result is stored in a map so that once all hops have been completed,
        a full result can be returned for merging.
    """

    def __init__(self, logger):
        """Initialize."""
        self.logger = logger
        self.normalizer = Normalizer(self.logger)
        self.kps = dict()

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
            yield QueryGraph.parse_obj(
                {"nodes": dict(), "edges": dict()}
            ), Result.parse_obj(
                {
                    "node_bindings": dict(),
                    "analyses": [],
                }
            ), AuxiliaryGraphs.parse_obj(
                {}
            )
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
            self.logger.info(
                f"[{kp.id}]: Got onehop with {len(onehop_response['results'])} results from cache"
            )
            onehop_response = Message.parse_obj(onehop_response, normalize=False)
        if onehop_response is None and not settings.offline_mode:
            # onehop not in cache, have to go get response
            self.logger.info(
                f"[{kp.id}] Need to get results for: {json.dumps(elide_curies(onehop_qgraph))}"
            )
            onehop_response = await kp.solve_onehop(
                onehop_qgraph,
            )
            await save_kp_onehop(kp.id, onehop_qgraph, onehop_response.dict())
        if onehop_response is None and settings.offline_mode:
            self.logger.info(
                f"[{kp.id}] Didn't get anything back from cache in offline mode."
            )
            # Offline mode and query wasn't cached, just continue
            onehop_results = []
        else:
            onehop_response = enforce_constraints(onehop_response)
            onehop_kgraph = onehop_response.knowledge_graph
            onehop_results = onehop_response.results
            onehop_auxgraphs = onehop_response.auxiliary_graphs
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
                result_kgraph = KnowledgeGraph.parse_obj(
                    {
                        "nodes": {
                            binding.id: onehop_kgraph.nodes[binding.id]
                            for _, bindings in result.node_bindings.items()
                            for binding in bindings
                        },
                        "edges": {
                            binding.id: onehop_kgraph.edges[binding.id]
                            for analysis in result.analyses or []
                            for _, bindings in analysis.edge_bindings.items()
                            for binding in bindings
                        },
                    }
                )

                # collect all auxiliary graph ids from results and edges
                aux_graphs = [
                    aux_graph_id
                    for analysis in result.analyses or []
                    for aux_graph_id in analysis.support_graphs or []
                ]

                aux_graphs.extend([
                    aux_graph_id
                    for analysis in result.analyses or []
                    for _, bindings in analysis.edge_bindings.items()
                    for binding in bindings
                    for attribute in onehop_kgraph.edges[binding.id].attributes or []
                    if attribute.attribute_type_id == "biolink:support_graphs"
                    for aux_graph_id in attribute.value
                ])

                result_auxgraph = AuxiliaryGraphs.parse_obj(
                    {
                        aux_graph_id: onehop_auxgraphs[aux_graph_id]
                        for aux_graph_id in aux_graphs
                    }
                )

                # pin nodes
                for qnode_id, bindings in result.node_bindings.items():
                    if qnode_id not in subqgraph["nodes"]:
                        continue
                    subqgraph["nodes"][qnode_id]["ids"] = (
                        subqgraph["nodes"][qnode_id].get("ids") or []
                    ) + [binding.id for binding in bindings]
                qnode_ids = set(subqgraph["nodes"].keys()) & set(
                    result.node_bindings.keys()
                )
                key_fcn = lambda res: tuple(
                    (
                        qnode_id,
                        tuple(binding.id for binding in bindings),  # probably only one
                    )
                    for qnode_id, bindings in res.node_bindings.items()
                    if qnode_id in qnode_ids
                )
                result_map[key_fcn(result)].append(
                    (result, result_kgraph, result_auxgraph)
                )

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
        async for subkgraph, subresult, subauxgraph in self.lookup(
            qgraph,
            call_stack,
        ):
            for result, kgraph, auxgraph in get_results(subresult):
                # combine one-hop with subquery results
                # Need to create a new result with all node bindings combined
                new_subresult = Result.parse_obj(
                    {
                        "node_bindings": {
                            **subresult.node_bindings,
                            **result.node_bindings,
                        },
                        "analyses": [
                            *subresult.analyses,
                            *result.analyses,
                            # reconsider
                        ],
                    }
                )
                new_subkgraph = copy.deepcopy(subkgraph)
                new_subkgraph.nodes.update(kgraph.nodes)
                new_subkgraph.edges.update(kgraph.edges)
                new_auxgraph = copy.deepcopy(subauxgraph)
                new_auxgraph.update(auxgraph)
                yield new_subkgraph, new_subresult, new_auxgraph

    async def __aenter__(self):
        """Enter context."""
        for kp in self.kps.values():
            await kp.throttle.__aenter__()
        return self

    async def __aexit__(self, *args):
        """Exit context."""
        for kp in self.kps.values():
            await kp.throttle.__aexit__()

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
            await self.normalizer.load_curies(*curies)
            curie_map = self.normalizer.map(curies, self.preferred_prefixes)
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

            self.kps[kp_id] = KnowledgeProvider(
                kp_id,
                kp,
                self.logger,
            )
