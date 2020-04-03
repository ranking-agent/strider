"""Simple ReasonerStdAPI server."""
import json
import os
import sqlite3
from typing import Dict

import aioredis
import aiosqlite
from fastapi import Depends, FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware

from strider.models import Query, Message
from strider.setup_query import execute_query, generate_plan
from strider.scoring import score_graph

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')

APP = FastAPI(
    title='Strider/ARAGORN/Ranking Agent',
    description='Translator Autonomous Relay Agent',
    version='1.0.0',
)
APP.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_db():
    """Get SQLite connection."""
    async with aiosqlite.connect('results.db') as database:
        yield database


async def get_redis():
    """Get Redis connection."""
    redis = await aioredis.create_redis_pool(
        f'redis://{REDIS_HOST}',
        encoding='utf-8',
    )
    yield redis
    redis.close()


@APP.post('/query', response_model=str, tags=['query'])
async def answer_query(
        query: Query,
        support: bool = True,
) -> str:
    """Answer biomedical question."""
    query_id = await execute_query(query.message.query_graph.dict(), support=support)
    return query_id


@APP.get('/results', response_model=Message)
async def get_results(  # pylint: disable=too-many-arguments
        query_id: str,
        since: float = None,
        limit: int = None,
        offset: int = 0,
        database=Depends(get_db),
        redis=Depends(get_redis),
) -> Message:
    """Get results for a query."""
    qgraph = {
        'nodes': [],
        'edges': [],
    }
    for value in await redis.hvals(f'{query_id}_slots'):
        value = json.loads(value)
        if 'source_id' in value:
            qgraph['edges'].append(value)
        else:
            qgraph['nodes'].append(value)

    # get column names from results db
    async with database.execute(f'PRAGMA table_info("{query_id}")') as cursor:
        columns = [row[1] for row in await cursor.fetchall()]

    kgraph = {
        'nodes': dict(),
        'edges': dict(),
    }
    results = []
    for row in await extract_results(query_id, since, limit, offset, database):
        result, _kgraph = parse_bindings(dict(zip(columns, row)), qgraph)
        results.append(result)
        kgraph['nodes'].update(_kgraph['nodes'])
        kgraph['edges'].update(_kgraph['edges'])
    # convert kgraph nodes and edges to list format
    kgraph = {
        'nodes': list(kgraph['nodes'].values()),
        'edges': list(kgraph['edges'].values()),
    }
    return {
        'query_graph': qgraph,
        'knowledge_graph': kgraph,
        'results': results
    }


def parse_bindings(bindings, qgraph):
    """Parse bindings into message format."""
    kgraph = {
        'nodes': dict(),
        'edges': dict(),
    }
    result = {
        'node_bindings': [],
        'edge_bindings': [],
    }
    qedge_ids = [edge['id'] for edge in qgraph['edges']]
    for key, element in bindings.items():
        if key.startswith('_'):
            result[key[1:]] = element
            continue
        kid = element.pop('kid')
        qid = element.pop('qid')
        if qid in qedge_ids:
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
    try:
        cursor = await database.execute(
            statement,
        )
    except sqlite3.OperationalError as err:
        if 'no such table' in str(err):
            raise HTTPException(400, str(err))
        raise err
    return [
        tuple(json.loads(value) if isinstance(value, str) else value for value in row)
        for row in await cursor.fetchall()
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
    slots = {
        el['id']: el
        for el in message['query_graph']['nodes'] + message['query_graph']['edges']
    }
    knodes = {
        knode['id']: knode
        for knode in message['knowledge_graph']['nodes']
    }
    for result in message['results']:
        graph = {
            'nodes': {
                nb['qg_id']: {
                    'qid': nb['qg_id'],
                    'kid': nb['kg_id'],
                    'equivalent_identifiers': knodes[nb['kg_id']].get('equivalent_identifiers', [])
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
            slots,
        )
    return message
