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
import os

from reasoner_pydantic import QueryGraph, Result

from .kp_registry import Registry
from .query_planner import generate_plan, Step
from .compatibility import KnowledgePortal
from .trapi import merge_messages, merge_results
from .worker import Worker
from .caching import async_locking_cache

LOGGER = logging.getLogger(__name__)


class StriderWorker(Worker):
    """Strider async worker."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        self.kgraph: dict[str, dict] = {
            "nodes": dict(),
            "edges": dict(),
        }
        self.plan: list[Step] = None
        self.preferred_prefixes: dict[str, list[str]] = None
        self.qgraph: QueryGraph = None
        self.registry: Registry = Registry("http://registry")
        self.results: list[Result] = []
        self.portal: KnowledgePortal = KnowledgePortal()
        super().__init__(*args, **kwargs)

    async def setup(
            self,
            qgraph: QueryGraph,
    ):
        """Set up."""
        # get preferred prefixes
        prefixes_json = os.getenv("PREFIXES", None)
        if prefixes_json:
            with open(prefixes_json, "r") as stream:
                self.preferred_prefixes = json.load(stream)
        else:
            self.preferred_prefixes = None

        # fix qgraph
        self.qgraph = (await self.portal.map_prefixes(
            {"query_graph": qgraph},
            self.preferred_prefixes,
        ))["query_graph"]

        # generate traversal plan
        plans = await generate_plan(self.qgraph, self.registry)
        self.plan = plans[-1]

        # add first partial result
        for qnode_id, qnode in self.qgraph["nodes"].items():
            if "id" not in qnode or qnode["id"] is None:
                continue
            result = {
                "node_bindings": {
                    qnode_id: [{"id": qnode["id"]}]
                },
                "edge_bindings": {},
            }
            await self.put(result)
            break

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

    @async_locking_cache
    async def execute_step(
            self,
            step: Step,
            curie: str,
    ):
        """Fetch results for step."""
        curie = await self.portal.map_curie(
            curie,
            list(self.plan[step].values())[0]["preferred_prefixes"]
        )
        responses = await asyncio.gather(*(
            self.portal.fetch(
                details["url"],
                details["request_template"](curie),
                details["preferred_prefixes"],
                self.preferred_prefixes,
            )
            for details in self.plan[step].values()
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
        LOGGER.debug(result)
        LOGGER.debug(step)
        response = await self.execute_step(
            step,
            result["node_bindings"][step.source][0]["id"],
        )

        # process kgraph
        self.kgraph["nodes"] |= response["knowledge_graph"]["nodes"]
        self.kgraph["edges"] |= response["knowledge_graph"]["edges"]

        # process results
        for new_result in response["results"]:
            # queue the results for further processing
            await self.put(merge_results([result, new_result]))
