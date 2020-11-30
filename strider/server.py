"""Simple ReasonerStdAPI server."""
import asyncio
import itertools
import json
import logging
from typing import Dict

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.openapi.docs import (
    get_swagger_ui_html,
)
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx
from starlette.middleware.cors import CORSMiddleware

from reasoner_pydantic import Query, Message

from .fetcher import StriderWorker
from .query_planner import generate_plan, NoAnswersError
from .scoring import score_graph
from .results import get_db, Database
from .util import setup_logging

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
    "message": {
        "query_graph": {
            "nodes": {
                "n1": {
                    "id": "CHEBI:136043",
                    "category": "biolink:ChemicalSubstance",
                },
                "n2": {
                    "category": "biolink:Disease",
                }
            },
            "edges": {
                "e12": {
                    "subject": "n1",
                    "object": "n2",
                    "predicate": "biolink:treats",
                }
            }
        },
    }
}

@APP.post('/query', response_model=Message, tags=['query'])
async def sync_query(
        query: Query = Body(..., example=EXAMPLE),
        support: bool = True,
) -> Message:
    """Handle synchronous query."""
    return await sync_answer(
        query.dict(),
        support=support,
    )


async def sync_answer(query: Dict, **kwargs):
    """Answer biomedical question, synchronously."""
    try:
        queue = asyncio.PriorityQueue()
        counter = itertools.count()

        # setup strider
        strider = StriderWorker(
            queue,
            num_workers=2,
            counter=counter,
        )

        await strider.run(query["message"]["query_graph"], wait=True)
        return {
            "query_graph": query["query_graph"],
            "results": strider.results,
            "knowledge_graph": strider.kgraph,
        }
    except NoAnswersError as err:
        LOGGER.warning(str(err))
        return query['message']
    async with Database('results.db') as database:
        return await _get_results(
            query_id=query_id,
            database=database,
        )


@APP.post('/aquery', response_model=str, tags=['query'])
async def async_query(
        query: Query,
        support: bool = True,
) -> str:
    """Handle asynchronous query."""
    query_id = await execute_query(
        query.message.query_graph.dict(),
        support=support,
        wait=False,
    )
    return query_id


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


@APP.get('/results', response_model=Message)
async def get_results(  # pylint: disable=too-many-arguments
        query_id: str,
        since: float = None,
        limit: int = None,
        offset: int = 0,
        database=Depends(get_db('results.db')),
) -> Message:
    """Get results for a query."""
    return await _get_results(query_id, since, limit, offset, database)


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
