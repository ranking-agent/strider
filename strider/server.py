"""Simple ReasonerStdAPI server."""
import uuid
import asyncio
import itertools
import json
import logging
import os
from typing import Dict

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.openapi.docs import (
    get_swagger_ui_html,
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
from starlette.middleware.cors import CORSMiddleware

from reasoner_pydantic import Query, Message, \
        Response as ReasonerResponse

from .fetcher import StriderWorker
from .query_planner import generate_plan, NoAnswersError
from .scoring import score_graph
from .results import get_db, Database
from .util import setup_logging
from .storage import RedisGraph, RedisList

LOGGER = logging.getLogger(__name__)

APP = FastAPI(
    title='Strider/ARAGORN/Ranking Agent',
    description='Translator Autonomous Relay Agent',
    version='1.0.0',
    terms_of_service='N/A',
    docs_url=None,
    redoc_url=None,
)
APP.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_logging()

EXAMPLE = {
        "message" : {
            "query_graph": {
                "nodes": {
                    "n0": {
                        "id": "MONDO:0001056",
                        "category": "biolink:Disease"
                        },
                    "n1": {
                        "category": "biolink:Disease"
                        }
                    },
                "edges": {
                    "e01": {
                        "subject": "n0",
                        "object": "n1",
                        "predicate": "biolink:related_to"
                        }
                    }
                }
            }
        }

# How long we are storing results for
store_results_for = int(os.getenv(
    "STORE_RESULTS_FOR",
    1 * 24 * 60 * 60,
))

def get_finished_query(qid: str) -> ReasonerResponse:
    qgraph  = RedisGraph(f"{qid}:qgraph")
    kgraph  = RedisGraph(f"{qid}:kgraph")
    results = RedisList(f"{qid}:results")
    logs    = RedisList(f"{qid}:log")

    qgraph.expire(store_results_for)
    kgraph.expire(store_results_for)
    results.expire(store_results_for)
    logs.expire(store_results_for)

    return ReasonerResponse(
            message = Message(
                query_graph=qgraph.get(),
                knowledge_graph=kgraph.get(),
                results=list(results.get()),
                ),
            logs = list(logs.get()),
        )

async def process_query(qid):
    # Set up workers
    strider = StriderWorker(num_workers=2)

    # Process
    await strider.run(qid, wait=True)

    # Pull results from redis
    # Also starts timer for expiring results
    return get_finished_query(qid)

@APP.post('/aquery', tags=['query'])
async def async_query(
        query: Query = Body(..., example=EXAMPLE),
        ) -> dict:
    """Start query processing."""

    # Generate Query ID
    qid = str(uuid.uuid4())[:8]

    # Save query graph to redis
    qgraph = RedisGraph(f"{qid}:qgraph")
    qgraph.set(query.dict()['message']['query_graph'])

    # Start processing
    process_query(qid)

    # Return ID
    return dict(id=qid)

@APP.post('/query_result', response_model=Message)
async def get_results(qid: str) -> ReasonerResponse:
    print(get_finished_query(qid))
    return get_finished_query(qid)

@APP.post('/query', tags=['query'])
async def sync_query(
        query: Query = Body(..., example=EXAMPLE)
) -> ReasonerResponse:
    """Handle synchronous query."""
    # Generate Query ID
    qid = str(uuid.uuid4())[:8]

    # Save query graph to redis
    qgraph = RedisGraph(f"{qid}:qgraph")
    qgraph.set(
            query.dict()['message']['query_graph']
            )

    # Process query and wait for results
    query_results = await process_query(qid)

    # Return results
    return query_results

@APP.post('/ars')
async def handle_ars(
        data: Dict,
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


@APP.post('/plan', response_model=Dict, tags=['query'])
async def generate_traversal_plan(
        query: Query,
) -> Dict:
    """Generate a plan for traversing knowledge providers."""
    query_graph = query.message.query_graph.dict()

    query_graph = {
        'nodes': {
            qnode['id']: qnode
            for qnode in query_graph['nodes']
        },
        'edges': {
            qedge['id']: {
                **qedge,
                "type": qedge["type"] or "related_to",
            }
            for qedge in query_graph['edges']
        }
    }
    return await generate_plan(query_graph)


@APP.post('/score', response_model=Message, tags=['query'])
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
