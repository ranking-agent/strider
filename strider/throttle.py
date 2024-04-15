"""Strider TRAPI Throttle."""

import asyncio
from asyncio.queues import QueueEmpty
from asyncio.tasks import Task
import copy
import json
import datetime
from functools import wraps
import itertools
from json.decoder import JSONDecodeError
import logging
import traceback
from typing import Callable, Optional, Union

import httpx
import pydantic
from reasoner_pydantic import (
    Response as ReasonerResponse,
    Query,
    Message,
    KnowledgeGraph,
)
from reasoner_pydantic.utils import HashableSequence, HashableMapping
import uuid

from .throttle_utils import (
    get_keys_with_value,
    log_response,
    get_curies,
    get_max_num_curies,
    remove_curies,
    filter_by_curie_mapping,
)
from .trapi import get_canonical_qgraphs
from .utils import elide_curies, log_request, remove_null_values
from .caching import async_locking_cache
from .config import settings


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
        *args,
        preproc: Callable = anull,
        postproc: Callable = anull,
        logger: logging.Logger = None,
        parameters: dict = {},
        **kwargs,
    ):
        """Initialize."""
        self.id = id
        self.worker: Optional[Task] = None
        self.request_queue = asyncio.Queue()
        self.url = url
        self.preproc = preproc
        self.postproc = postproc
        self.use_cache = settings.use_cache
        self.parameters = parameters
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

        # locking cache needs to be here so each KP instance has its own cache.
        # https://stackoverflow.com/a/14946506
        if self.use_cache:
            self.query = async_locking_cache(self._query)
        else:
            self.query = self._query

    @log_errors
    async def process_batch(
        self,
    ):
        """Set up a subscriber to process batching."""

        while True:
            # Get everything in the stream or wait for something to show up
            (
                (
                    request_id,
                    payload,
                    response_queue,
                    bypass_cache,
                    call_stack,
                    last_hop,
                ),
            ) = await self.request_queue.get()
            request_value_mapping = {request_id: payload}
            response_queues = {request_id: response_queue}
            while True:
                # check if we've reached the max batch size
                max_curies = get_max_num_curies(request_value_mapping.values())
                if max_curies >= self.parameters["batch_size"]:
                    break
                try:
                    (
                        (
                            request_id,
                            payload,
                            response_queue,
                            bypass_cache,
                            call_stack,
                            last_hop,
                        ),
                    ) = self.request_queue.get_nowait()
                except QueueEmpty:
                    break
                # check if we've hit max batch size with the newly added request
                max_curies = get_max_num_curies(request_value_mapping.values())
                if max_curies >= self.parameters["batch_size"]:
                    # need to put this request back on the queue cause it didn't fit
                    # within batch size
                    await self.request_queue.put(
                        (
                            (
                                request_id,
                                payload,
                                response_queue,
                                bypass_cache,
                                call_stack,
                                last_hop,
                            ),
                        )
                    )
                    break
                request_value_mapping[request_id] = payload
                response_queues[request_id] = response_queue

            # Extract a curie mapping from each request
            request_curie_mapping = {
                request_id: get_curies(request_value.message.query_graph)
                for request_id, request_value in request_value_mapping.items()
            }

            # Find requests that are the same (those that we can merge)
            # This disregards non-matching IDs because the IDs have been
            # removed with the extract_curie method
            stripped_qgraphs = {
                request_id: remove_curies(request.message.query_graph)
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
                            (
                                request_id,
                                request_value_mapping[request_id],
                                response_queues[request_id],
                                bypass_cache,
                                call_stack,
                                last_hop,
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
            for qnode in merged_request_value.message.query_graph.nodes.values():
                qnode.ids = None

            # Update merged request using curie mapping
            for curie_mapping in request_curie_mapping.values():
                for node_id, node_curies in curie_mapping.items():
                    node = merged_request_value.message.query_graph.nodes[node_id]
                    if node.ids is None:
                        node.ids = node_curies.copy()
                    else:
                        node.ids.extend(node_curies)
            # TODO replace qnode.ids with a HashableSet so that this can be safely removed
            for qnode in merged_request_value.message.query_graph.nodes.values():
                if qnode.ids:
                    qnode.ids = list(set(qnode.ids))

            response_values = dict()
            try:
                self.logger.context = self.id
                await self.preproc(merged_request_value)
                # TODO rewrite this whole function to use pydantic model
                merged_request_value = merged_request_value.dict()
                merged_request_value["submitter"] = "infores:aragorn"
                merged_request_value["bypass_cache"] = bypass_cache
                merged_request_value = remove_null_values(merged_request_value)
                # Make request
                self.logger.info(
                    "[{callstack}] Sending request made of {subrequests} subrequests ({curies} curies)".format(
                        callstack=", ".join(call_stack),
                        subrequests=len(request_curie_mapping),
                        curies=" x ".join(
                            str(len(qnode.get("ids") or []))
                            for qnode in merged_request_value["message"]["query_graph"][
                                "nodes"
                            ].values()
                        ),
                    )
                )
                # Use kp timeout given in the message, otherwise use env variable
                kp_timeout = self.parameters.get("kp_timeout")
                kp_timeout = (
                    kp_timeout if type(kp_timeout) is int else settings.kp_timeout
                )
                async with httpx.AsyncClient(timeout=kp_timeout) as client:
                    response = await client.post(
                        self.url,
                        json=merged_request_value,
                    )

                response.raise_for_status()
                response_dict = response.json()

                msg = response_dict.get("message") or {}
                results = msg.get("results") or []
                num_results = len(results)
                self.logger.info(
                    "[{}] Received response with {} results in {} seconds".format(
                        (", ").join(call_stack),
                        num_results,
                        response.elapsed.total_seconds(),
                    )
                )
                if response.elapsed.total_seconds() > (settings.kp_timeout * 10):
                    self.logger.warning(
                        f"Response from {self.id} took way too long: {response.elapsed.total_seconds()} seconds"
                    )

                # Parse with reasoner_pydantic to validate
                response_body = ReasonerResponse.parse_obj(response_dict)
                await self.postproc(response_body, last_hop)
                new_num_results = len(response_body.message.results or [])
                if num_results != new_num_results:
                    self.logger.info(
                        f"[{(', ').join(call_stack)}] Postprocessing took out {num_results - new_num_results} results"
                    )
                message = response_body.message

                try:
                    if len(request_curie_mapping) == 1:
                        request_id = next(iter(request_curie_mapping))
                        # Make a copy
                        response_values[request_id] = ReasonerResponse(
                            message=Message()
                        )
                        response_values[request_id].message.query_graph = (
                            request_value_mapping[request_id].message.query_graph.copy()
                        )
                        response_values[request_id].message.knowledge_graph = (
                            message.knowledge_graph
                            or KnowledgeGraph(nodes={}, edges={})
                        ).copy()
                        response_values[request_id].message.results = (
                            message.results or HashableSequence(__root__=[])
                        ).copy()
                        response_values[request_id].message.auxiliary_graphs = (
                            message.auxiliary_graphs or HashableMapping(__root__={})
                        ).copy()
                    else:
                        # Split using the request_curie_mapping
                        for request_id, curie_mapping in request_curie_mapping.items():
                            filtered_msg = filter_by_curie_mapping(
                                message, curie_mapping, kp_id=self.id
                            )
                            filtered_msg.query_graph = request_value_mapping[
                                request_id
                            ].message.query_graph.copy()
                            response_values[request_id] = ReasonerResponse(
                                message=filtered_msg
                            )
                except Exception as err:
                    # Raise more descriptive error message of response message parsing
                    raise Exception(
                        "[{}] Failed to parse message response: {} with Error: {}".format(
                            (", ").join(call_stack),
                            response_dict,
                            traceback.format_exc(),
                        )
                    )
            except (
                asyncio.exceptions.TimeoutError,
                httpx.RequestError,
                httpx.HTTPStatusError,
                JSONDecodeError,
                pydantic.ValidationError,
                Exception,
            ) as e:
                for request_id, curie_mapping in request_curie_mapping.items():
                    response_values[request_id] = ReasonerResponse(
                        message=request_value_mapping[request_id].message.copy()
                    )
                if isinstance(e, asyncio.TimeoutError):
                    self.logger.warning(
                        {
                            "message": f"{self.id} took > {kp_timeout} seconds to respond",
                            "error": str(e),
                            "request": elide_curies(merged_request_value),
                        }
                    )
                elif isinstance(e, httpx.ReadTimeout):
                    self.logger.warning(
                        {
                            "message": f"{self.id} took > {kp_timeout} seconds to respond",
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
                            "request": json.dumps(merged_request_value),
                            "response": response.text,
                            "error": str(e),
                        }
                    )
                elif isinstance(e, pydantic.ValidationError):
                    self.logger.warning(
                        {
                            "message": f"Received non-TRAPI compliant response from {self.id}",
                            "error": f"{str(e)[:500]}...",
                            "request": elide_curies(merged_request_value),
                        }
                    )
                else:
                    self.logger.warning(
                        {
                            "message": f"Something went wrong while querying {self.id}",
                            "error": traceback.format_exc(),
                        }
                    )

            for request_id, response_value in response_values.items():
                # Write finished value to DB
                await response_queues[request_id].put(response_value)

    async def __aenter__(self):
        """Set KP info and start processing task."""
        loop = asyncio.get_event_loop()
        self.worker = loop.create_task(self.process_batch())

        return self

    async def __aexit__(self):
        """Cancel KP processing task."""
        task: Task = self.worker
        self.worker = None

        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

    @log_errors
    async def _query(
        self,
        query: Query,
        bypass_cache: bool,
        call_stack: list,
        last_hop: bool,
    ) -> ReasonerResponse:
        """Queue up a query for batching and return when completed"""
        if self.worker is None:
            raise RuntimeError(
                "Cannot send a request until a worker is running - enter the context"
            )

        # TODO figure out a way to remove this conversion
        query = Query.parse_obj(query)

        response_queue = asyncio.Queue()

        qgraphs = get_canonical_qgraphs(query.message.query_graph)

        for qgraph in qgraphs:
            subquery = Query(message=Message(query_graph=qgraph))

            # Queue query for processing
            request_id = str(uuid.uuid1())
            await self.request_queue.put(
                ((request_id, subquery, response_queue, bypass_cache, call_stack, last_hop),)
            )

        combined_output = ReasonerResponse.parse_obj(
            {
                "message": {
                    "knowledge_graph": {"nodes": {}, "edges": {}},
                    "results": [],
                }
            }
        )

        for _ in qgraphs:
            # Wait for response
            output: Union[ReasonerResponse, Exception] = await asyncio.wait_for(
                response_queue.get(),
                timeout=None,
            )

            if isinstance(output, Exception):
                raise output

            output.message.query_graph = None
            combined_output.message.update(output.message)

        combined_output.message.query_graph = query.message.query_graph

        return combined_output
