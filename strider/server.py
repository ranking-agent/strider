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

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')

app = FastAPI(
    title='Strider/ARAGORN/Ranking Agent',
    description='Translator Autonomous Relay Agent',
    version='1.0.0',
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_db():
    """Get SQLite connection."""
    async with aiosqlite.connect('results.db') as db:
        yield db


async def get_redis():
    """Get Redis connection."""
    redis = await aioredis.create_redis_pool(
        f'redis://{REDIS_HOST}',
        encoding='utf-8',
    )
    yield redis
    redis.close()

# @app.post('/query', response_model=Message, tags=['query'])
# async def answer_query(query: Query) -> Message:
@app.post('/query', response_model=str, tags=['query'])
async def answer_query(
        query: Query,
        support: bool = True,
) -> str:
    """Answer biomedical question."""
    query_id = await execute_query(query.message.query_graph.dict(), support=support)
    return query_id


@app.get('/results', response_model=Message)
async def get_results(
        query_id: str,
        t0: float = None,
        limit: int = None,
        offset: int = 0,
        db=Depends(get_db),
        redis=Depends(get_redis),
) -> Message:
    """Get results for a query."""
    # get column names
    statement = f'PRAGMA table_info("{query_id}")'
    cursor = await db.execute(
        statement,
    )
    results = await cursor.fetchall()
    columns = [row[1] for row in results]
    slots = {
        key: json.loads(value)
        for key, value in (await redis.hgetall(f'{query_id}_slots')).items()
    }

    # get result rows
    statement = f'SELECT * FROM "{query_id}"'
    if t0 is not None:
        statement += f' WHERE _timestamp >= {t0}'
    statement += ' ORDER BY _timestamp ASC'
    if t0 is None:
        if limit is not None:
            statement += f' LIMIT {limit}'
        if offset:
            statement += f' OFFSET {offset}'
    try:
        cursor = await db.execute(
            statement,
        )
    except sqlite3.OperationalError as err:
        if 'no such table' in str(err):
            raise HTTPException(400, str(err))
        raise err
    _results = await cursor.fetchall()

    results = []
    for row in _results:
        node_bindings = []
        edge_bindings = []
        result = dict()
        for key, value in zip(columns, row):
            if key.startswith('_'):
                result[key[1:]] = value
            elif 'source_id' in slots[key]:
                edge_bindings.append({
                    'qg_id': key,
                    'kg_id': value,
                })
            else:
                node_bindings.append({
                    'qg_id': key,
                    'kg_id': value,
                })
        result.update({
            'node_bindings': node_bindings,
            'edge_bindings': edge_bindings,
        })
        results.append(result)
    qgraph = {
        'nodes': [],
        'edges': [],
    }
    for value in slots.values():
        if 'source_id' in value:
            qgraph['edges'].append(value)
        else:
            qgraph['nodes'].append(value)
    message = {
        'query_graph': qgraph,
        'results': results
    }
    return message


@app.post('/plan', response_model=Dict, tags=['query'])
async def generate_traversal_plan(
        query: Query,
) -> Dict:
    """Generate a plan for traversing knowledge providers."""
    query_graph = query.message.query_graph.dict()
    return await generate_plan(query_graph)
