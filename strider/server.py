"""
 .oooooo..o     .             o8o        .o8
d8P'    `Y8   .o8             `"'       "888
Y88bo.      .o888oo oooo d8b oooo   .oooo888   .ooooo.  oooo d8b
 `"Y8888o.    888   `888""8P `888  d88' `888  d88' `88b `888""8P
     `"Y88b   888    888      888  888   888  888ooo888  888
oo     .d8P   888 .  888      888  888   888  888    .o  888
8""88888P'    "888" d888b    o888o `Y8bod88P" `Y8bod8P' d888b
"""

import copy
import datetime
import json
import os
import uuid
import logging
import warnings
import time
import traceback
import asyncio

from fastapi import Body, HTTPException, BackgroundTasks, Request, status
from fastapi.openapi.docs import (
    get_swagger_ui_html,
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel
from reasoner_pydantic.kgraph import KnowledgeGraph
from reasoner_pydantic.qgraph import QueryGraph
from reasoner_pydantic.utils import HashableMapping
from reasoner_pydantic.message import Result
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, Response

from kp_registry import Registry
from reasoner_pydantic import (
    Query,
    AsyncQuery,
    Message,
    Response as ReasonerResponse,
    Results,
    AuxiliaryGraphs,
)

from .caching import (
    get_kp_registry,
    save_kp_registry,
    get_registry_lock,
    remove_registry_lock,
    clear_cache,
)
from .fetcher import Fetcher
from .node_sets import collapse_sets
from .query_planner import NoAnswersError, generate_plan
from .config import settings
from .utils import add_cors_manually, setup_logging
from .openapi import TRAPI
from .logger import QueryLogger

setup_logging()
LOGGER = logging.getLogger(__name__)
registry = Registry()
backup_kps = {}

DESCRIPTION = """
<img src="/static/favicon.svg" width="200px">
<br /><br />
Translator Autonomous Relay Agent
"""

openapi_args = dict(
    title="Strider",
    description=DESCRIPTION,
    docs_url=None,
    version="4.8.4",
    terms_of_service=(
        "http://robokop.renci.org:7055/tos"
        "?service_long=Strider"
        "&provider_long=the%20Renaissance%20Computing%20Institute"
        "&provider_short=RENCI"
    ),
    translator_component="ARA",
    translator_teams=["Ranking Agent"],
    contact={
        "name": "Max Wang",
        "email": "max@covar.com",
        "x-id": "maximusunc",
        "x-role": "responsible developer",
    },
    trapi_operations=[
        "lookup",
        "filter_results_top_n",
    ],
)
if settings.openapi_server_url:
    openapi_args["servers"] = [
        {
            "url": settings.openapi_server_url,
            "x-maturity": settings.openapi_server_maturity,
            "x-location": settings.openapi_server_location,
        },
    ]
APP = TRAPI(**openapi_args)

CORS_OPTIONS = dict(
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP.add_middleware(
    CORSMiddleware,
    **CORS_OPTIONS,
)

if settings.profiler:
    # importing this file will engage profiler
    from .profiler import profiler_middleware

# Custom exception handler is necessary to ensure that
# we add CORS headers to errors and return a TRAPI response


async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        response = JSONResponse(
            {
                "message": {},
                "logs": [
                    {
                        "message": f"Exception in Strider: {repr(e)}",
                        "level": "ERROR",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "stack": traceback.format_exc(),
                    }
                ],
            },
            status_code=500,
        )
        add_cors_manually(APP, request, response, CORS_OPTIONS)
        return response


APP.middleware("http")(catch_exceptions_middleware)

if settings.jaeger_enabled == "True":
    LOGGER.info("Starting up Jaeger")

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import (
        SERVICE_NAME,
        Resource,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    # httpx connections need to be open a little longer by the otel decorators
    # but some libs display warnings of resource being unclosed.
    # these supresses such warnings.
    logging.captureWarnings(capture=True)
    warnings.filterwarnings("ignore", category=ResourceWarning)
    service_name = os.environ.get("OTEL_SERVICE_NAME", "STRIDER")
    jaeger_exporter = JaegerExporter(
        agent_host_name=settings.jaeger_host,
        agent_port=int(settings.jaeger_port),
    )
    resource = Resource(attributes={SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(jaeger_exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(
        APP, tracer_provider=provider, excluded_urls="docs,openapi.json"
    )
    HTTPXClientInstrumentor().instrument()


@APP.on_event("startup")
async def refresh_kp_registry():
    if not settings.offline_mode:
        await reload_kp_registry()


APP.mount("/static", StaticFiles(directory="static"), name="static")


@APP.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(req: Request) -> HTMLResponse:
    """Customize Swagger UI."""
    root_path = req.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + APP.openapi_url
    swagger_favicon_url = root_path + "/static/favicon.svg"
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title=APP.title + " - Swagger UI",
        oauth2_redirect_url=APP.swagger_ui_oauth2_redirect_url,
        swagger_favicon_url=swagger_favicon_url,
    )


async def reload_kp_registry():
    """Reload the kp registry and store in cache."""
    global backup_kps
    try:
        if await get_registry_lock():
            LOGGER.info("getting registry of kps")
            kps = await registry.retrieve_kps()
            await save_kp_registry(kps)
            LOGGER.info("kp registry refreshed.")
            await remove_registry_lock()
    except Exception:
        # failed to even get lock status, so need to fall back to in-memory registry
        backup_kps = await registry.retrieve_kps()


@APP.post("/refresh", status_code=status.HTTP_202_ACCEPTED, include_in_schema=False)
async def refresh_kps(background_tasks: BackgroundTasks):
    """Refresh registered KPs by consulting SmartAPI registry."""
    background_tasks.add_task(reload_kp_registry)
    return "Queued refresh. It will take a few seconds."


EXAMPLE = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["MONDO:0005148"], "categories": ["biolink:Disease"]},
                "n1": {"categories": ["biolink:PhenotypicFeature"]},
            },
            "edges": {
                "e01": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:has_phenotype"],
                }
            },
        }
    }
}

AEXAMPLE = {
    "callback": "",
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["MONDO:0005148"], "categories": ["biolink:Disease"]},
                "n1": {"categories": ["biolink:PhenotypicFeature"]},
            },
            "edges": {
                "e01": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:has_phenotype"],
                }
            },
        }
    },
}

MEXAMPLE = {
    "query1": {
        "callback": "",
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": ["MONDO:0005148"], "categories": ["biolink:Disease"]},
                    "n1": {"categories": ["biolink:PhenotypicFeature"]},
                },
                "edges": {
                    "e01": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": ["biolink:has_phenotype"],
                    }
                },
            }
        },
    },
    "query2": {
        "callback": "",
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": ["MONDO:0005148"], "categories": ["biolink:Disease"]},
                    "n1": {"categories": ["biolink:SmallMolecule"]},
                },
                "edges": {
                    "e01": {
                        "subject": "n0",
                        "object": "n1",
                        "predicates": ["biolink:treats"],
                    }
                },
            }
        },
    },
}


@APP.post("/query", response_model=ReasonerResponse)
async def sync_query(
    query: Query = Body(..., example=EXAMPLE),
) -> dict:
    """Handle synchronous query."""
    # parse requested workflow
    query_dict = query.dict()
    workflow = query_dict.get("workflow", None) or [{"id": "lookup"}]
    if not isinstance(workflow, list):
        raise HTTPException(400, "workflow must be a list")
    if not len(workflow) == 1:
        raise HTTPException(400, "workflow must contain exactly 1 operation")
    if "id" not in workflow[0]:
        raise HTTPException(400, "workflow must have property 'id'")
    if workflow[0]["id"] == "filter_results_top_n":
        max_results = workflow[0]["parameters"]["max_results"]
        if max_results < len(query_dict["message"]["results"]):
            query_dict["message"]["results"] = query_dict["message"]["results"][
                :max_results
            ]
        return query_dict
    if not workflow[0]["id"] == "lookup":
        raise HTTPException(400, "operations must have id 'lookup'")

    if (query_dict.get("set_interpretation", None) or "BATCH") == "MANY":
        raise HTTPException(422, "set_interpretation MANY not supported.")

    query_results = {}
    # Generate Query ID
    qid = str(uuid.uuid4())[:8]
    try:
        LOGGER.info(f"[{qid}] Starting sync query")
        # get max timeout
        timeout_seconds = (query_dict.get("parameters") or {}).get("timeout_seconds")
        # if timeout_seconds, less 10 seconds to account for sending back
        timeout = (
            timeout_seconds - 10
            if type(timeout_seconds) is int
            else settings.max_process_time
        )
        query_results = await asyncio.wait_for(lookup(query_dict, qid), timeout=timeout)
    except asyncio.TimeoutError:
        LOGGER.error(f"[{qid}] Sync query cancelled due to timeout.")
        query_results = {
            "message": {},
            "status_communication": {"strider_process_status": "timeout"},
        }
    except Exception as e:
        LOGGER.error(
            f"[{qid}] Sync query failed unexpectedly: {traceback.format_exc()}"
        )
        qid = "Exception"
        query_results = {
            "message": {},
            "status_communication": {"strider_process_status": "error"},
        }

    # Return results
    msg = query_results.get("message") or {}
    num_results = len(msg.get("results") or [])
    LOGGER.info(f"[{qid}] Returning sync query with {num_results} results")
    return JSONResponse(query_results)


@APP.post("/asyncquery")
async def async_query(
    background_tasks: BackgroundTasks,
    query: AsyncQuery = Body(..., example=AEXAMPLE),
):
    """Handle asynchronous query."""
    # parse requested workflow
    query_dict = query.dict()
    callback = query_dict["callback"]
    workflow = query_dict.get("workflow", None) or [{"id": "lookup"}]
    if not isinstance(workflow, list):
        raise HTTPException(400, "workflow must be a list")
    if not len(workflow) == 1:
        raise HTTPException(400, "workflow must contain exactly 1 operation")
    if "id" not in workflow[0]:
        raise HTTPException(400, "workflow must have property 'id'")
    if workflow[0]["id"] == "filter_results_top_n":
        max_results = workflow[0]["parameters"]["max_results"]
        if max_results < len(query_dict["message"]["results"]):
            query_dict["message"]["results"] = query_dict["message"]["results"][
                :max_results
            ]
        return query_dict
    if not workflow[0]["id"] == "lookup":
        raise HTTPException(400, "operations must have id 'lookup'")

    if (query_dict.get("set_interpretation", None) or "BATCH") == "MANY":
        raise HTTPException(422, "set_interpretation MANY not supported.")

    LOGGER.info(f"Doing async lookup for {callback}")
    background_tasks.add_task(async_lookup, callback, query_dict)

    return


@APP.post("/multiquery")
async def multi_query(
    background_tasks: BackgroundTasks,
    multiquery: dict[str, AsyncQuery] = Body(..., example=MEXAMPLE),
):
    """Handles multiple queries. Queries are sent back to a callback url as a dict with keys corresponding to keys of original query."""
    # Generate multiuery ID
    multiqid = str(uuid.uuid4())[:8]
    query_keys = list(multiquery.keys())
    callback = multiquery[query_keys[0]].dict().get("callback")
    if not callback:
        raise HTTPException(400, "callback url must be specified")
    queries = {}
    for query in query_keys:
        query_dict = multiquery[query].dict()
        query_callback = query_dict["callback"]
        workflow = query_dict.get("workflow", None) or [{"id": "lookup"}]
        if not isinstance(workflow, list):
            raise HTTPException(400, "workflow must be a list")
        if not len(workflow) == 1:
            raise HTTPException(400, "workflow must contain exactly 1 operation")
        if "id" not in workflow[0]:
            raise HTTPException(400, "workflow must have property 'id'")
        if not workflow[0]["id"] == "lookup":
            raise HTTPException(400, "operations must have id 'lookup'")
        if query_callback != callback:
            raise HTTPException(400, "callback url for all queries must be the same")
        queries[query] = query_dict

    LOGGER.info(
        f"[{multiqid}] Starting {len(query_keys)} multi lookup queries for {callback}"
    )
    background_tasks.add_task(multi_lookup, multiqid, callback, queries, query_keys)

    return


async def lookup(
    query_dict: dict,
    qid: str = None,
) -> dict:
    """Perform lookup operation."""
    global backup_kps
    lookup_start_time = time.time()
    message = query_dict.get("message", {})

    log_level = query_dict.get("log_level") or "INFO"
    # grab information content threshold from message if exists, otherwise grab from environment
    information_content_threshold = (
        query_dict.get("information_content_threshold")
        or settings.information_content_threshold
    )

    level_number = logging._nameToLevel[log_level]
    # Set up logger
    log_handler = QueryLogger().log_handler
    logger = logging.getLogger(f"strider.{qid}")
    logger.setLevel(level_number)
    logger.addHandler(log_handler)

    bypass_cache = query_dict.get("bypass_cache") or False
    parameters = query_dict.get("parameters") or {}

    fetcher = Fetcher(logger, bypass_cache, parameters)

    logger.info(f"Doing lookup for message: {json.dumps(message)}")
    try:
        await fetcher.setup(message, backup_kps, information_content_threshold)
    except NoAnswersError as e:
        logger.warning(f"Returning no results. {e}")
        return {
            "message": message,
            "logs": list(log_handler.contents()),
        }
    except Exception as e:
        raise e

    # Result container to make result merging much faster
    output_results = HashableMapping[str, Result]()

    output_kgraph = KnowledgeGraph.parse_obj(
        message.get("knowledge_graph") or {"nodes": {}, "edges": {}}
    )

    output_auxgraphs = AuxiliaryGraphs.parse_obj(message.get("auxiliary_graphs") or {})

    message_merging_time = 0

    async with fetcher:
        async for result_kgraph, result, result_auxgraph, sub_qid in fetcher.lookup(
            None
        ):
            # Update the kgraph
            start_merging = time.time()
            output_kgraph.update(result_kgraph)

            # Update the aux graphs
            output_auxgraphs.update(result_auxgraph)

            # Update the results
            # hashmap lookup is very quick
            sub_result_hash = hash(result)
            existing_result = output_results.get(sub_result_hash, None)
            if existing_result:
                # update existing result
                existing_result.update(result)
            else:
                # add new result to hashmap
                output_results[sub_result_hash] = result

            stop_merging = time.time()
            message_merging_time += stop_merging - start_merging

    results = Results.parse_obj([])
    for result in output_results.values():
        # copy so result analyses don't get combined somehow
        result = copy.deepcopy(result)
        if len(result.analyses) > 1:
            result.combine_analyses_by_resource_id()
        results.append(result)

    output_query = Query(
        message=Message(
            query_graph=fetcher.message.query_graph,
            knowledge_graph=output_kgraph,
            results=results,
            auxiliary_graphs=output_auxgraphs,
        ),
        log_level=log_level,
        workflow=query_dict.get("workflow"),
        bypass_cache=bypass_cache,
    )

    # Collapse sets
    collapse_sets(output_query, logger)

    output_query.logs = list(log_handler.contents())
    lookup_end_time = time.time()
    logger.info(
        {
            "total_lookup_time": (lookup_end_time - lookup_start_time),
            "total_merging": message_merging_time,
        }
    )
    return output_query.dict(exclude_none=True)


async def async_lookup(
    callback,
    query_dict: dict,
):
    """Perform lookup and send results to callback url"""
    qid = str(uuid.uuid4())[:8]
    query_results = {}
    try:
        # get max timeout
        timeout_seconds = (query_dict.get("parameters") or {}).get("timeout_seconds")
        # if timeout_seconds, less 10 seconds to account for sending back
        timeout = (
            timeout_seconds - 10
            if type(timeout_seconds) is int
            else settings.max_process_time
        )
        query_results = await asyncio.wait_for(lookup(query_dict, qid), timeout=timeout)
    except asyncio.TimeoutError:
        LOGGER.error(f"[{qid}]: Process cancelled due to timeout.")
        query_results = {
            "message": {},
            "status_communication": {"strider_process_status": "timeout"},
        }
    except Exception as e:
        LOGGER.error(f"[{qid}]: Query failed unexpectedly: {traceback.format_exc()}")
        query_results = {
            "message": {},
            "status_communication": {"strider_process_status": "error"},
        }
    try:
        msg = query_results.get("message") or {}
        num_results = len(msg.get("results") or [])
        LOGGER.info(
            f"[{qid}] Posting async query response with {num_results} results to {callback}"
        )
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=600.0)
            ) as client:
                res = await client.post(callback, json=query_results)
                res.raise_for_status()
                LOGGER.info(f"[{qid}] Posted to {callback} with code {res.status_code}")
        except Exception as e:
            LOGGER.error(f"[{qid}]: Callback to {callback} failed with: {e}")
    except Exception as e:
        LOGGER.error(e)


async def multi_lookup(multiqid, callback, queries: dict, query_keys: list):
    "Performs lookup for multiple queries and sends all results to callback url"
    start_time = time.time()

    async def single_lookup(query_key):
        qid = f"{multiqid}.{str(uuid.uuid4())[:8]}"
        query_result = {}
        try:
            # get max timeout
            timeout_seconds = (queries[query_key].get("parameters") or {}).get(
                "timeout_seconds"
            )
            # if timeout_seconds, less 10 seconds to account for sending back
            timeout = (
                timeout_seconds - 10
                if type(timeout_seconds) is int
                else settings.max_process_time
            )
            query_result = await asyncio.wait_for(
                lookup(queries[query_key], qid), timeout=timeout
            )
        except asyncio.TimeoutError:
            LOGGER.error(f"[{qid}]: Process cancelled due to timeout.")
            query_result = {
                "message": {},
                "status_communication": {"strider_process_status": "timeout"},
            }
        except Exception as e:
            LOGGER.error(
                f"[{qid}]: Query failed unexpectedly: {traceback.format_exc()}"
            )
            query_result = {
                "message": {},
                "status_communication": {"strider_process_status": "error"},
            }
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(timeout=600.0)
            ) as client:
                LOGGER.info(f"[{qid}]: Calling back to {callback}...")
                query_result["parameters"] = queries[query_key].get("parameters") or {}
                query_result["parameters"]["multiquery_uid"] = query_key
                callback_response = await client.post(callback, json=query_result)
                LOGGER.info(
                    f"[{qid}]: Called back to {callback}. Status={callback_response.status_code}"
                )
                callback_response.raise_for_status()
        except Exception as e:
            LOGGER.error(f"[{qid}]: Callback to {callback} failed with: {e}")
        return query_result

    await asyncio.gather(*map(single_lookup, query_keys), return_exceptions=True)

    query_results = {
        "message": {},
        "status_communication": {"strider_multiquery_status": "complete"},
    }

    LOGGER.info(f"[{multiqid}] All jobs complete.  Sending done signal to {callback}.")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=600.0)) as client:
            callback_response = await client.post(callback, json=query_results)
            LOGGER.info(
                f"[{multiqid}] Sent completion to {callback}. Status={callback_response.status_code}"
            )
    except Exception as e:
        LOGGER.error(
            f"[{multiqid}] Failed to send 'completed' response back to {callback} with error: {e}"
        )
    end_time = time.time()
    LOGGER.info(f"[{multiqid}] took {(end_time - start_time)} seconds")


@APP.post("/plan", response_model=dict[str, list[str]], include_in_schema=False)
async def generate_traversal_plan(
    query: Query,
):
    """Generate plans for traversing knowledge providers."""
    query_graph = query.message.query_graph

    plan, _ = await generate_plan(query_graph, backup_kps, LOGGER)

    return plan


@APP.get("/kps")
async def get_kps():
    """Return all kps in registry."""
    registry = await get_kp_registry()
    return sorted(
        list(
            kp["infores"]
            for kp in registry.values()
            if kp["maturity"] == settings.openapi_server_maturity
        )
    )


class ClearCacheRequest(BaseModel):
    pswd: str


@APP.post("/clear_cache", status_code=200, include_in_schema=False)
async def clear_redis_cache(request: ClearCacheRequest) -> dict:
    """Clear the redis cache."""
    if request.pswd == settings.redis_password:
        await clear_cache()
        return {"status": "success"}
    else:
        raise HTTPException(status_code=401, detail="Invalid Password")
