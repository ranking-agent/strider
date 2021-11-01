"""Strider TRAPI Throttle."""
import asyncio
from asyncio.queues import QueueEmpty
from asyncio.tasks import Task
import copy
import datetime
from functools import wraps
import itertools
from json.decoder import JSONDecodeError
import logging
import traceback
from typing import Callable, Optional, Union

import httpx
import pydantic
from reasoner_pydantic import Response as ReasonerResponse
import uuid

from .trapi import BatchingError, get_curies, remove_curies, filter_by_curie_mapping
from .utils import get_keys_with_value, log_response
from ..trapi import canonicalize_qgraph
from ..util import elide_curies, log_request
from ..caching import async_locking_query_cache


class KPInformation(pydantic.main.BaseModel):
    url: pydantic.AnyHttpUrl
    request_qty: int
    request_duration: datetime.timedelta


def log_errors(fcn):
    @wraps(fcn)
    async def wrapper(*args, **kwargs):
        try:
            return await fcn(*args, **kwargs)
        except Exception as err:
            traceback.print_exc()
            raise

    return wrapper


async def anull(arg, *args, **kwargs):
    """Do nothing, asynchronously."""
    return arg


class ThrottledServer:
    """Throttled server."""

    def __init__(
        self,
        id: str,
        url: str,
        request_qty: int,
        request_duration: float,
        use_cache: bool = True,
        *args,
        max_batch_size: Optional[int] = None,
        timeout: float = 60.0,
        preproc: Callable = anull,
        postproc: Callable = anull,
        logger: logging.Logger = None,
        **kwargs,
    ):
        """Initialize."""
        self.id = id
        self.worker: Optional[Task] = None
        self.request_queue = asyncio.PriorityQueue()
        self.counter = itertools.count()
        self.url = url
        self.request_qty = request_qty
        self.request_duration = datetime.timedelta(seconds=request_duration)
        self.use_cache = use_cache
        self.timeout = timeout
        self.max_batch_size = max_batch_size
        self.preproc = preproc
        self.postproc = postproc
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

        # locking cache needs to be here so each KP instance has its own cache.
        # https://stackoverflow.com/a/14946506
        if use_cache:
            self.query = async_locking_query_cache(self._query)
        else:
            self.query = self._query

    @log_errors
    async def process_batch(
        self,
    ):
        """Set up a subscriber to process batching."""
        # Initialize the TAT
        #
        # TAT = Theoretical Arrival Time
        # When the next request should be sent
        # to adhere to the rate limit.
        #
        # This is an implementation of the GCRA algorithm
        # More information can be found here:
        # https://dev.to/astagi/rate-limiting-using-python-and-redis-58gk
        if self.request_qty > 0:
            interval = self.request_duration / self.request_qty
            tat = datetime.datetime.utcnow() + interval

        while True:
            # Get everything in the stream or wait for something to show up
            priority, (
                request_id,
                payload,
                response_queue,
            ) = await self.request_queue.get()
            priorities = {request_id: priority}
            request_value_mapping = {request_id: payload}
            response_queues = {request_id: response_queue}
            while True:
                if (
                    self.max_batch_size is not None
                    and len(request_value_mapping) == self.max_batch_size
                ):
                    break
                try:
                    priority, (
                        request_id,
                        payload,
                        response_queue,
                    ) = self.request_queue.get_nowait()
                except QueueEmpty:
                    break
                priorities[request_id] = priority
                request_value_mapping[request_id] = payload
                response_queues[request_id] = response_queue

            # Extract a curie mapping from each request
            request_curie_mapping = {
                request_id: get_curies(request_value["message"]["query_graph"])
                for request_id, request_value in request_value_mapping.items()
            }

            # Find requests that are the same (those that we can merge)
            # This disregards non-matching IDs because the IDs have been
            # removed with the extract_curie method
            stripped_qgraphs = {
                request_id: remove_curies(request["message"]["query_graph"])
                for request_id, request in request_value_mapping.items()
            }
            first_value = next(iter(stripped_qgraphs.values()))

            batch_request_ids = get_keys_with_value(
                stripped_qgraphs,
                first_value,
            )

            # Re-queue the un-selected requests
            for request_id in request_value_mapping:
                if request_id not in batch_request_ids:
                    await self.request_queue.put(
                        (
                            priorities[request_id],
                            (
                                request_id,
                                request_value_mapping[request_id],
                                response_queues[request_id],
                            ),
                        )
                    )

            request_value_mapping = {
                k: v for k, v in request_value_mapping.items() if k in batch_request_ids
            }

            # Filter curie mapping to only include matching requests
            request_curie_mapping = {
                k: v for k, v in request_curie_mapping.items() if k in batch_request_ids
            }

            # Pull first value from request_value_mapping
            # to use as a template for our merged request
            merged_request_value = copy.deepcopy(
                next(iter(request_value_mapping.values()))
            )

            # Remove qnode ids
            for qnode in merged_request_value["message"]["query_graph"][
                "nodes"
            ].values():
                qnode.pop("ids", None)

            # Update merged request using curie mapping
            for curie_mapping in request_curie_mapping.values():
                for node_id, node_curies in curie_mapping.items():
                    node = merged_request_value["message"]["query_graph"]["nodes"][
                        node_id
                    ]
                    if "ids" not in node:
                        node["ids"] = []
                    node["ids"].extend(node_curies)
            for qnode in merged_request_value["message"]["query_graph"][
                "nodes"
            ].values():
                if qnode.get("ids"):
                    qnode["ids"] = list(set(qnode["ids"]))

            response_values = dict()
            try:
                # Make request
                self.logger.info(
                    "[{id}] Sending request made of {subrequests} subrequests ({curies} curies)".format(
                        id=self.id,
                        subrequests=len(request_curie_mapping),
                        curies=" x ".join(
                            str(len(qnode.get("ids", []) or []))
                            for qnode in merged_request_value["message"]["query_graph"][
                                "nodes"
                            ].values()
                        ),
                    )
                )
                self.logger.context = self.id
                merged_request_value = await self.preproc(
                    merged_request_value, self.logger
                )
                merged_request_value["submitter"] = "infores:aragorn"
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.url,
                        json=merged_request_value,
                        timeout=self.timeout,
                    )
                if response.status_code == 429:
                    # reset TAT
                    interval = self.request_duration / self.request_qty
                    tat = datetime.datetime.utcnow() + interval
                    # re-queue requests
                    for request_id in request_value_mapping:
                        await self.request_queue.put(
                            (
                                priorities[request_id],
                                (
                                    request_id,
                                    request_value_mapping[request_id],
                                    response_queues[request_id],
                                ),
                            )
                        )
                    # try again later
                    continue

                response.raise_for_status()

                # Parse with reasoner_pydantic to validate
                response_body = ReasonerResponse.parse_obj(response.json()).dict()
                response_body = await self.postproc(response_body)
                message = response_body["message"]
                results = message.get("results") or []
                self.logger.info(
                    "[{}] Received response with {} results in {} seconds".format(
                        self.id,
                        len(results),
                        response.elapsed.total_seconds(),
                    )
                )

                if len(request_curie_mapping) == 1:
                    request_id = next(iter(request_curie_mapping))
                    response_values[request_id] = {
                        "message": {
                            "query_graph": request_value_mapping[request_id]["message"][
                                "query_graph"
                            ],
                            "knowledge_graph": message.get(
                                "knowledge_graph", {"nodes": {}, "edges": {}}
                            ),
                            "results": message.get("results", []),
                        }
                    }
                else:
                    # Split using the request_curie_mapping
                    for request_id, curie_mapping in request_curie_mapping.items():
                        try:
                            kgraph, results = filter_by_curie_mapping(
                                message, curie_mapping, kp_id=self.id
                            )
                            response_values[request_id] = {
                                "message": {
                                    "query_graph": request_value_mapping[request_id][
                                        "message"
                                    ]["query_graph"],
                                    "knowledge_graph": kgraph,
                                    "results": results,
                                }
                            }
                        except BatchingError as err:
                            # the response is probably malformed
                            response_values[request_id] = err
            except (
                asyncio.exceptions.TimeoutError,
                httpx.RequestError,
                httpx.HTTPStatusError,
                JSONDecodeError,
                pydantic.ValidationError,
            ) as e:
                for request_id, curie_mapping in request_curie_mapping.items():
                    response_values[request_id] = {
                        "message": {
                            "query_graph": request_value_mapping[request_id]["message"][
                                "query_graph"
                            ],
                            "knowledge_graph": {
                                "nodes": {},
                                "edges": {},
                            },
                            "results": [],
                        },
                    }
                if isinstance(e, asyncio.TimeoutError):
                    self.logger.warning(
                        {
                            "message": f"{self.id} took >60 seconds to respond",
                            "error": str(e),
                            "request": elide_curies(merged_request_value),
                        }
                    )
                elif isinstance(e, httpx.ReadTimeout):
                    self.logger.warning(
                        {
                            "message": f"{self.id} took >60 seconds to respond",
                            "error": str(e),
                            "request": log_request(e.request),
                        }
                    )
                elif isinstance(e, httpx.RequestError):
                    # Log error
                    self.logger.warning(
                        {
                            "message": f"Request Error contacting {self.id}",
                            "error": str(e),
                            "request": log_request(e.request),
                        }
                    )
                elif isinstance(e, httpx.HTTPStatusError):
                    # Log error with response
                    self.logger.warning(
                        {
                            "message": f"Response Error contacting {self.id}",
                            "error": str(e),
                            "request": log_request(e.request),
                            "response": log_response(e.response),
                        }
                    )
                elif isinstance(e, JSONDecodeError):
                    # Log error with response
                    self.logger.warning(
                        {
                            "message": f"Received bad JSON data from {self.id}",
                            "request": e.request,
                            "response": e.response.text,
                            "error": str(e),
                        }
                    )
                elif isinstance(e, pydantic.ValidationError):
                    self.logger.warning(
                        {
                            "message": f"Received non-TRAPI compliant response from {self.id}",
                            "error": str(e),
                        }
                    )
                else:
                    self.logger.warning(
                        {
                            "message": f"Something went wrong while querying {self.id}",
                            "error": str(e),
                        }
                    )

            for request_id, response_value in response_values.items():
                # Write finished value to DB
                await response_queues[request_id].put(response_value)

            # if request_qty == 0 we don't enforce the rate limit
            if self.request_qty > 0:
                time_remaining_seconds = (
                    tat - datetime.datetime.utcnow()
                ).total_seconds()

                # Wait for TAT
                if time_remaining_seconds > 0:
                    await asyncio.sleep(time_remaining_seconds)

                # Update TAT
                tat = datetime.datetime.utcnow() + interval

    async def __aenter__(
        self,
    ):
        """Set KP info and start processing task."""
        loop = asyncio.get_event_loop()
        self.worker = loop.create_task(self.process_batch())

        return self

    async def __aexit__(
        self,
        *args,
    ):
        """Cancel KP processing task."""
        task: Task = self.worker
        self.worker = None

        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _query(
            self,
            query: dict,
            priority: float = 0,  # lowest goes first
            timeout: Optional[float] = 60.0,
    ) -> dict:
        """Queue up a query for batching and return when completed"""
        if self.worker is None:
            raise RuntimeError(
                "Cannot send a request until a worker is running - enter the context"
            )

        response_queue = asyncio.Queue()

        qgraphs = canonicalize_qgraph(query["message"]["query_graph"])

        for qgraph in qgraphs:
            subquery = {
                "message": {
                    "query_graph": qgraph,
                },
            }

            # Queue query for processing
            request_id = str(uuid.uuid1())
            await self.request_queue.put(
                (
                    (priority, next(self.counter)),
                    (request_id, subquery, response_queue),
                )
            )

        outputs = []
        for _ in qgraphs:
            # Wait for response
            output: Union[dict, Exception] = await asyncio.wait_for(
                response_queue.get(),
                timeout=None,
            )

            if isinstance(output, Exception):
                raise output

            outputs.append(output)
        output = {
            "message": {
                "query_graph": query["message"]["query_graph"],
                "knowledge_graph": {
                    "nodes": {
                        key: value
                        for output in outputs
                        for key, value in output["message"]["knowledge_graph"][
                            "nodes"
                        ].items()
                    },
                    "edges": {
                        key: value
                        for output in outputs
                        for key, value in output["message"]["knowledge_graph"][
                            "edges"
                        ].items()
                    },
                },
                "results": [
                    result
                    for output in outputs
                    for result in output["message"]["results"]
                ],
            }
        }

        return output


class DuplicateError(Exception):
    """Duplicate KP."""


class Throttle:
    """TRAPI Throttle."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        self.servers: dict[str, ThrottledServer] = dict()

    async def register_kp(
        self,
        kp_id: str,
        kp_info: dict,
    ):
        """Set KP info and start processing task."""
        if kp_id in self.servers:
            raise DuplicateError(f"{kp_id} already exists")
        self.servers[kp_id] = ThrottledServer(kp_id, **kp_info)
        await self.servers[kp_id].__aenter__()

    async def unregister_kp(
        self,
        kp_id: str,
    ):
        """Cancel KP processing task."""
        await self.servers.pop(kp_id).__aexit__()

    async def query(
        self,
        kp_id: str,
        query: dict,
    ) -> dict:
        """Queue up a query for batching and return when completed"""
        return await self.servers[kp_id].query(query)
