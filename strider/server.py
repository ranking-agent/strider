"""Simple ReasonerStdAPI server."""
import copy
import datetime
import os
import uuid
import json
import logging
import traceback
import pprint
from typing import Optional

from fastapi import Body, Depends, HTTPException, BackgroundTasks, Request
from fastapi.openapi.docs import (
    get_swagger_ui_html,
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
from redis import Redis
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, Response

from reasoner_pydantic import Query, AsyncQuery, Message, Response as ReasonerResponse

from .fetcher import Binder
from .node_sets import collapse_sets
from .query_planner import NoAnswersError, generate_plan
from .scoring import score_graph
from .storage import RedisGraph, RedisList, get_client as get_redis_client
from .config import settings
from .util import add_cors_manually
from .trapi import build_unique_kg_edge_ids, merge_kgraphs
from .trapi_openapi import TRAPI

LOGGER = logging.getLogger(__name__)

DESCRIPTION = """
<img src="/static/favicon.svg" width="200px">
<br /><br />
Translator Autonomous Relay Agent
"""

openapi_args = dict(
    title="Strider",
    description=DESCRIPTION,
    docs_url=None,
    version="3.16.1",
    terms_of_service=(
        "http://robokop.renci.org:7055/tos"
        "?service_long=Strider"
        "&provider_long=the%20Renaissance%20Computing%20Institute"
        "&provider_short=RENCI"
    ),
    translator_component="ARA",
    translator_teams=["Ranking Agent"],
    contact={
        "name": "Kenneth Morton",
        "email": "kenny@covar.com",
        "x-id": "kennethmorton",
        "x-role": "responsible developer",
    },
    trapi_operations=[
        "lookup",
        "filter_results_top_n",
    ],
    root_path=os.environ.get("ROOT_PATH", "/"),
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


@APP.on_event("startup")
async def print_config():
    pretty_config = pprint.pformat(settings.dict())
    LOGGER.info(f" App Configuration:\n {pretty_config}")


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


def get_finished_query(
    qid: str,
    redis_client: Redis,
) -> dict:
    qgraph = RedisGraph(f"{qid}:qgraph", redis_client)
    kgraph = RedisGraph(f"{qid}:kgraph", redis_client)
    results = RedisList(f"{qid}:results", redis_client)
    logs = RedisList(f"{qid}:log", redis_client)

    expiration_seconds = int(settings.store_results_for.total_seconds())

    qgraph.expire(expiration_seconds)
    kgraph.expire(expiration_seconds)
    results.expire(expiration_seconds)
    logs.expire(expiration_seconds)

    results = list(results.get())

    return dict(
        message=dict(
            query_graph=qgraph.get(),
            knowledge_graph=kgraph.get(),
            results=results,
        ),
        logs=list(logs.get()),
    )


@APP.post("/query_result", response_model=ReasonerResponse)
async def get_results(
    qid: str,
    redis_client: Redis = Depends(get_redis_client),
) -> dict:
    """Get results for a running or finished query"""
    return get_finished_query(qid, redis_client)


async def lookup(
    query_dict: dict,
    redis_client: Redis,
) -> dict:
    """Perform lookup operation."""
    # Generate Query ID
    qid = str(uuid.uuid4())[:8]

    qgraph = query_dict["message"]["query_graph"]

    log_level = query_dict["log_level"] or "ERROR"

    level_number = logging._nameToLevel[log_level]

    binder = Binder(
        qid,
        level_number,
        name="me",
        redis_client=redis_client,
    )

    output_query = Query.parse_obj(
        {
            "message": {
                "query_graph": qgraph,
                "knowledge_graph": {"nodes": {}, "edges": {}},
                "results": [],
            }
        }
    )

    try:
        await binder.setup(qgraph)
    except NoAnswersError:
        return {
            "message": {
                "query_graph": qgraph,
                "knowledge_graph": {"nodes": {}, "edges": {}},
                "results": [],
            },
            "logs": list(RedisList(f"{qid}:log", redis_client).get()),
        }

    async with binder:
        async for result_kgraph, result in binder.lookup(None):
            message_dict = {"knowledge_graph": result_kgraph, "results": [result]}
            # TODO figure out why this is required
            message_dict = copy.deepcopy(message_dict)

            # Update knowledge graph edge IDs so that we can merge
            build_unique_kg_edge_ids(message_dict)

            # Merge
            output_query.message.update(Message.parse_obj(message_dict))

    # Collapse sets
    message_dict = output_query.message.dict()
    collapse_sets(message_dict)
    output_query.message = Message.parse_obj(message_dict)

    output_query.logs = list(RedisList(f"{qid}:log", redis_client).get())
    return output_query.dict()


async def async_lookup(
    callback,
    query_dict: dict,
    redis_client: Redis,
):
    """Preform lookup and send results to callback url"""
    query_results = await lookup(query_dict, redis_client)
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=600.0)) as client:
        await client.post(callback, json=query_results)


@APP.post("/query", response_model=ReasonerResponse)
async def sync_query(
    query: Query = Body(..., example=EXAMPLE),
    redis_client: Redis = Depends(get_redis_client),
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

    query_results = await lookup(query_dict, redis_client)

    # Return results
    return JSONResponse(query_results)


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


@APP.post("/asyncquery", response_model=ReasonerResponse)
async def async_query(
    background_tasks: BackgroundTasks,
    query: AsyncQuery = Body(..., example=AEXAMPLE),
    redis_client: Redis = Depends(get_redis_client),
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

    background_tasks.add_task(async_lookup, callback, query_dict, redis_client)

    return


def parse_bindings(bindings):
    """Parse bindings into message format."""
    kgraph = {
        "nodes": dict(),
        "edges": dict(),
    }
    result = {
        "node_bindings": [],
        "edge_bindings": [],
    }
    for key, element in bindings.items():
        if key.startswith("_"):
            result[key[1:]] = element
            continue
        kid = element.pop("kid")
        qid = element.pop("qid")
        element.pop("kid_qid", None)
        if key.startswith("e_"):
            result["edge_bindings"].append(
                {
                    "qg_id": qid,
                    "kg_id": kid,
                }
            )
            kgraph["edges"][kid] = {
                "id": kid,
                **element,
            }
        else:
            result["node_bindings"].append(
                {
                    "qg_id": qid,
                    "kg_id": kid,
                }
            )
            kgraph["nodes"][kid] = {
                "id": kid,
                **element,
            }
    return result, kgraph


async def extract_results(query_id, since, limit, offset, database):
    """Extract results from database."""
    statement = f'SELECT * FROM "{query_id}"'
    if since is not None:
        statement += f" WHERE _timestamp >= {since}"
    statement += " ORDER BY _timestamp ASC"
    if limit is not None:
        statement += f" LIMIT {limit}"
    if offset:
        statement += f" OFFSET {offset}"
    rows = await database.execute(statement)
    return [
        tuple(json.loads(value) if isinstance(value, str) else value for value in row)
        for row in rows
    ]


@APP.post("/plan", response_model=dict[str, list[str]])
async def generate_traversal_plan(
    query: Query,
) -> list[list[str]]:
    """Generate plans for traversing knowledge providers."""
    query_graph = query.message.query_graph.dict()

    plan, _ = await generate_plan(query_graph)

    return plan


@APP.post("/score", response_model=Message)
async def score_results(
    query: Query,
) -> Message:
    """Score results."""
    message = query.message.dict()
    identifiers = {
        knode["id"]: knode.get("equivalent_identifiers", [])
        for knode in message["knowledge_graph"]["nodes"]
    }
    for result in message["results"]:
        graph = {
            "nodes": {
                nb["qg_id"]: {
                    "qid": nb["qg_id"],
                    "kid": nb["kg_id"],
                    "equivalent_identifiers": identifiers[nb["kg_id"]],
                }
                for nb in result["node_bindings"]
            },
            "edges": {
                eb["qg_id"]: {
                    "qid": eb["qg_id"],
                    "kid": eb["kg_id"],
                }
                for eb in result["edge_bindings"]
            },
        }
        result["score"] = await score_graph(
            graph,
            message["query_graph"],
        )
    return message
