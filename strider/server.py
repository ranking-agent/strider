"""Simple ReasonerStdAPI server."""
import datetime
import uuid
import asyncio
import itertools
import json
import logging
import os
import enum
from pathlib import Path
import pprint

from fastapi import Body, FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.openapi.docs import (
    get_swagger_ui_html,
)
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
from starlette.middleware.cors import CORSMiddleware
import yaml

from reasoner_pydantic import Query, Message, Response as ReasonerResponse

from .fetcher import StriderWorker
from .query_planner import generate_plans, NoAnswersError
from .scoring import score_graph
from .results import get_db, Database
from .storage import RedisGraph, RedisList
from .config import settings
from .util import add_cors_manually, standardize_graph_lists
from .trapi import fill_categories_predicates, add_descendants

LOGGER = logging.getLogger(__name__)

APP = FastAPI(
    docs_url=None,
    redoc_url=None,
)


def custom_openapi():
    """Build custom OpenAPI schema."""
    if APP.openapi_schema:
        return APP.openapi_schema

    extra_info_file = Path(__file__).parent / "openapi-info.yml"

    with open(extra_info_file) as stream:
        extra_info = yaml.load(stream, Loader=yaml.SafeLoader)

    openapi_schema = get_openapi(
        title="",
        version="",
        tags=[
            {
                "name": "translator",
            },
            {
                "name": "reasoner",
            }
        ],
        routes=APP.routes
    )

    openapi_schema["info"] |= extra_info

    APP.openapi_schema = openapi_schema
    return APP.openapi_schema

APP.openapi = custom_openapi

CORS_OPTIONS = dict(
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP.add_middleware(
    CORSMiddleware,
    **CORS_OPTIONS,
)

# Custom exception handler is necessary to ensure that
# we add CORS headers to errors and return a TRAPI response
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        response = JSONResponse(
            {
                "message" : {
                    "logs": [
                        {
                            "message" : f"Exception in Strider: {repr(e)}",
                            "level" : "ERROR",
                            "timestamp" : datetime.datetime.now().isoformat(),
                        }
                    ]
                }
            },
            status_code=500)
        add_cors_manually(APP, request, response, CORS_OPTIONS)
        return response

APP.middleware('http')(catch_exceptions_middleware)


@APP.on_event("startup")
async def print_config():
    pretty_config = pprint.pformat(
        settings.dict()
    )
    LOGGER.info(f" App Configuration:\n {pretty_config}")

EXAMPLE = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {
                    "id": "MONDO:0005148",
                    "category": "biolink:Disease"
                },
                "n1": {
                    "category": "biolink:PhenotypicFeature"
                }
            },
            "edges": {
                "e01": {
                    "subject": "n0",
                    "object": "n1",
                    "predicate": "biolink:has_phenotype"
                }
            }
        }
    }
}


def get_finished_query(
        qid: str,
) -> dict:
    qgraph = RedisGraph(f"{qid}:qgraph")
    kgraph = RedisGraph(f"{qid}:kgraph")
    results = RedisList(f"{qid}:results")
    logs = RedisList(f"{qid}:log")

    expiration_seconds = int(settings.store_results_for.total_seconds())

    qgraph.expire(expiration_seconds)
    kgraph.expire(expiration_seconds)
    results.expire(expiration_seconds)
    logs.expire(expiration_seconds)

    return dict(
        message=dict(
            query_graph=qgraph.get(),
            knowledge_graph=kgraph.get(),
            results=list(results.get()),
        ),
        logs=list(logs.get()),
    )


async def process_query(
        qid: str,
        log_level: str,
):
    # pylint: disable=protected-access
    level_number = logging._nameToLevel[log_level]

    # Set up workers
    strider = StriderWorker(num_workers=2)
    await strider.setup(qid, level_number)

    # Generate plan
    try:
        await strider.generate_plan()
    except NoAnswersError:
        # End early with no results
        # (but we should have log messages)
        return get_finished_query(qid)

    # Process
    await strider.run(qid, wait=True)

    # Pull results from redis
    # Also starts timer for expiring results
    return get_finished_query(qid)


@APP.post('/aquery')
async def async_query(
        background_tasks: BackgroundTasks,
        query: Query = Body(..., example=EXAMPLE),
) -> dict:
    """Start query processing."""
    # Generate Query ID
    qid = str(uuid.uuid4())[:8]

    query_graph = query.dict()['message']['query_graph']
    standardize_graph_lists(query_graph)

    # Save query graph to redis
    redis_query_graph = RedisGraph(f"{qid}:qgraph")
    redis_query_graph.set(query_graph)

    log_level = query.dict()["log_level"] or "ERROR"

    # Start processing
    background_tasks.add_task(process_query, qid, log_level)

    # Return ID
    return dict(id=qid)


@APP.post('/query_result', response_model=ReasonerResponse)
async def get_results(
        qid: str,
) -> dict:
    """ Get results for a running or finished query """
    return get_finished_query(qid)


@APP.post('/query', tags=['reasoner'], response_model=ReasonerResponse)
async def sync_query(
        query: Query = Body(..., example=EXAMPLE),
) -> dict:
    """Handle synchronous query."""
    # Generate Query ID
    qid = str(uuid.uuid4())[:8]

    query_graph = query.dict()['message']['query_graph']
    standardize_graph_lists(query_graph)

    # Save query graph to redis
    redis_query_graph = RedisGraph(f"{qid}:qgraph")
    redis_query_graph.set(query_graph)

    log_level = query.dict()["log_level"] or "ERROR"

    # Process query and wait for results
    query_results = await process_query(qid, log_level)

    # Return results
    return query_results


@APP.post('/ars')
async def handle_ars(
        data: dict,
):
    """Handle ARS message."""
    if data.get('model', None) != 'tr_ars.message':
        raise HTTPException(
            status_code=400,
            detail='Not a valid Translator message',
        )
    data = data['fields']
    if data.get('ref', None) is not None:
        raise HTTPException(
            status_code=400,
            detail='Not head message',
        )
    if data.get('data', None) is not None:
        data = json.loads(data['data'])
    elif data.get('url', None) is not None:
        data = httpx.get(data['url'], timeout=60).json()
    else:
        raise HTTPException(
            status_code=400,
            detail='Not a valid tr_ars.message',
        )

    content = await sync_answer(data)
    headers = {'tr_ars.message.status': 'A'}
    return JSONResponse(content=content, headers=headers)

APP.mount("/static", StaticFiles(directory="static"), name="static")


@APP.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Customize Swagger UI."""
    return get_swagger_ui_html(
        openapi_url=APP.openapi_url,
        title=APP.title + " - Swagger UI",
        oauth2_redirect_url=APP.swagger_ui_oauth2_redirect_url,
        swagger_favicon_url="/static/favicon.png",
    )


async def _get_results(
        query_id: str,
        since: float = None,
        limit: int = None,
        offset: int = 0,
        database=None,
):
    """Get results."""
    # get column names from results db
    columns = await database.get_columns(query_id)

    kgraph = {
        'nodes': dict(),
        'edges': dict(),
    }
    results = []
    for row in await extract_results(query_id, since, limit, offset, database):
        result, _kgraph = parse_bindings(dict(zip(columns, row)))
        results.append(result)
        kgraph['nodes'].update(_kgraph['nodes'])
        kgraph['edges'].update(_kgraph['edges'])
    # convert kgraph nodes and edges to list format
    kgraph = {
        'nodes': list(kgraph['nodes'].values()),
        'edges': list(kgraph['edges'].values()),
    }
    return {
        'knowledge_graph': kgraph,
        'results': results
    }


def parse_bindings(bindings):
    """Parse bindings into message format."""
    kgraph = {
        'nodes': dict(),
        'edges': dict(),
    }
    result = {
        'node_bindings': [],
        'edge_bindings': [],
    }
    for key, element in bindings.items():
        if key.startswith('_'):
            result[key[1:]] = element
            continue
        kid = element.pop('kid')
        qid = element.pop('qid')
        element.pop('kid_qid', None)
        if key.startswith('e_'):
            result['edge_bindings'].append({
                'qg_id': qid,
                'kg_id': kid,
            })
            kgraph['edges'][kid] = {
                'id': kid,
                **element,
            }
        else:
            result['node_bindings'].append({
                'qg_id': qid,
                'kg_id': kid,
            })
            kgraph['nodes'][kid] = {
                'id': kid,
                **element,
            }
    return result, kgraph


async def extract_results(query_id, since, limit, offset, database):
    """Extract results from database."""
    statement = f'SELECT * FROM "{query_id}"'
    if since is not None:
        statement += f' WHERE _timestamp >= {since}'
    statement += ' ORDER BY _timestamp ASC'
    if limit is not None:
        statement += f' LIMIT {limit}'
    if offset:
        statement += f' OFFSET {offset}'
    rows = await database.execute(statement)
    return [
        tuple(
            json.loads(value) if isinstance(value, str) else value
            for value in row
        )
        for row in rows
    ]


@APP.post('/plan', response_model=list[dict])
async def generate_traversal_plan(
        query: Query,
) -> list[dict]:
    """Generate plans for traversing knowledge providers."""
    query_graph = query.message.query_graph.dict()

    await fill_categories_predicates(query_graph, logging.getLogger())
    standardize_graph_lists(query_graph)
    plans = await generate_plans(query_graph)

    plans_stringified_keys = []
    for plan in plans:
        plans_stringified_keys.append(
            {
                f"{step.source}-{step.edge}-{step.target}": v
                for step, v in plan.items()
            }
        )
    return plans_stringified_keys


@APP.post('/score', response_model=Message)
async def score_results(
        query: Query,
) -> Message:
    """Score results."""
    message = query.message.dict()
    identifiers = {
        knode['id']: knode.get('equivalent_identifiers', [])
        for knode in message['knowledge_graph']['nodes']
    }
    for result in message['results']:
        graph = {
            'nodes': {
                nb['qg_id']: {
                    'qid': nb['qg_id'],
                    'kid': nb['kg_id'],
                    'equivalent_identifiers': identifiers[nb['kg_id']]
                }
                for nb in result['node_bindings']
            },
            'edges': {
                eb['qg_id']: {
                    'qid': eb['qg_id'],
                    'kid': eb['kg_id'],
                }
                for eb in result['edge_bindings']
            }
        }
        result['score'] = await score_graph(
            graph,
            message['query_graph'],
        )
    return message
