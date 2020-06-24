"""Example: plan a query."""
import asyncio
import itertools
import json
import logging
import os
import time
import uuid

import aioredis
import uvloop

from strider.fetcher import Fetcher
from strider.query_planner import generate_plan
from strider.results import Database

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

LOGGER = logging.getLogger(__name__)
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')


async def execute_query(qgraph, **kwargs):
    """Execute a user query.

    1) Generate execution plan.
    2) Store execution plan in Redis.
    3) Set up results database.
    4) Add named nodes to job queue.
    """
    # generate query execution plan
    query_id = str(uuid.uuid4())
    qgraph = {
        'nodes': {
            qnode['id']: qnode
            for qnode in qgraph['nodes']
        },
        'edges': {
            qedge['id']: qedge
            for qedge in qgraph['edges']
        }
    }
    plan = await generate_plan(qgraph)

    # store plan in Redis
    redis = await aioredis.create_redis_pool(
        f'redis://{REDIS_HOST}'
    )
    await redis.delete(
        f'{query_id}_qgraph',
        f'{query_id}_plan',
        f'{query_id}_priorities',
        f'{query_id}_done',
        f'{query_id}_options',
    )
    await redis.set(f'{query_id}_starttime', time.time())
    for key, value in plan.items():
        await redis.hset(f'{query_id}_plan', key, json.dumps(value))
    await redis.set(f'{query_id}_qgraph', json.dumps(qgraph))
    for key, value in kwargs.items():
        await redis.hset(f'{query_id}_options', key, json.dumps(value))
    redis.close()

    # setup results DB
    slots = list(qgraph['nodes']) + list(qgraph['edges'])
    await setup_results(query_id, slots)

    # add a result for each named node
    # add a job for each named node
    queue = asyncio.PriorityQueue()
    counter = itertools.count()
    for node in qgraph['nodes'].values():
        if 'curie' not in node or node['curie'] is None:
            continue
        job_id = f'({node["curie"]}:{node["id"]})'
        job = {
            'query_id': query_id,
            'qid': node.pop('id'),
            'kid': node.pop('curie'),
            **node,
        }
        LOGGER.debug("Queueing result %s", job_id)
        queue.put_nowait((
            0,
            next(counter),
            job,
        ))

    # setup fetcher
    fetcher = Fetcher(
        queue,
        max_jobs=5,
        counter=counter,
        query_id=query_id,
        **kwargs,
    )
    await fetcher.run(qgraph)

    return query_id


async def setup_results(query_id, slots):
    """Set up results database."""
    column_names = ', '.join(
        [f'`{qid}`' for qid in slots]
        + ['_score', '_timestamp']
    )
    columns = ', '.join(
        [f'`{qid}` TEXT' for qid in slots]
        + ['_score REAL', '_timestamp REAL']
    )
    columns += f', UNIQUE({column_names})'
    statements = [
        f'DROP TABLE IF EXISTS `{query_id}`',
        f'CREATE TABLE `{query_id}` ({columns})',
    ]
    async with Database('results.db') as database:
        for statement in statements:
            await database.execute(statement)
