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
from datetime import datetime
import jsonpickle

from reasoner_pydantic import QueryGraph, Result

from .query_planner import generate_plan, Step, NoAnswersError
from .compatibility import KnowledgePortal
from .trapi import merge_messages, merge_results
from .worker import Worker
from .caching import async_locking_cache
from .storage import RedisGraph, RedisList, RedisLogHandler
from .kp_registry import Registry
from .config import settings

# Initialize registry
registry = Registry(settings.kpregistry_url)


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
        self.plan: dict[Step, list] = None
        self.preferred_prefixes: dict[str, list[str]] = None
        self.qgraph: QueryGraph = None
        self.results: list[Result] = []
        self.portal: KnowledgePortal = KnowledgePortal()
        super().__init__(*args, **kwargs)

    async def setup(
            self,
            qid: str,
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
        self.logger.addHandler(handler)

        self.logger.debug("Initialized strider worker")

        # Pull query graph from Redis
        qgraph = RedisGraph(f"{qid}:qgraph").get()

        # get preferred prefixes
        with open(settings.prefixes_path, "r") as stream:
            self.preferred_prefixes = json.load(stream)

        # fix qgraph
        self.qgraph = (await self.portal.map_prefixes(
            {"query_graph": qgraph},
            self.preferred_prefixes,
        ))["query_graph"]

        self.logger.debug("Generating plan")
        try:
            # Generate traversal plan
            self.plan = await generate_plan(self.qgraph, registry)
            self.logger.debug({"plan": self.plan})
        except NoAnswersError:
            self.logger.error({"code": "QueryNotTraversable"})

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

    @ async_locking_cache
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

        self.logger.debug({
            "description": "Executing step: ",
            "step": self.plan[step],
        })

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
        self.logger.debug({
            "description": "Recieved results from KPs",
            "data": result,
            "step": step,
        })

        try:
            response = await self.execute_step(
                step,
                result["node_bindings"][step.source][0]["id"],
            )
        except Exception:
            self.logger.exception('Failed to execute step')
            return

        # process kgraph
        self.kgraph.nodes.merge(response["knowledge_graph"]["nodes"])
        self.kgraph.edges.merge(response["knowledge_graph"]["edges"])

        # process results
        for new_result in response["results"]:
            # queue the results for further processing
            await self.put(merge_results([result, new_result]))
