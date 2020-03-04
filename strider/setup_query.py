"""Example: plan a query."""
import asyncio
import json
import logging
import os
import sqlite3
import uuid

import aiormq
import aioredis
import uvloop

from strider.query_planner import generate_plan
from strider.rabbitmq import setup as setup_rabbitmq

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

LOGGER = logging.getLogger(__name__)
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_USER = os.getenv('RABBITMQ_USER', 'guest')
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD', 'guest')


async def execute_query(query_graph):
    """Execute a user query.

    1) Generate execution plan.
    2) Store execution plan in Redis.
    3) Set up results SQL database.
    4) Add named nodes to job queue.
    """
    # generate query execution plan
    query_id = str(uuid.uuid4())
    plan = await generate_plan(query_graph)
    slots = dict(**{
        node['id']: json.dumps(node)
        for node in query_graph['nodes']
    }, **{
        edge['id']: json.dumps(edge)
        for edge in query_graph['edges']
    })

    # store plan in Redis
    redis = await aioredis.create_redis_pool(
        f'redis://{REDIS_HOST}'
    )
    await redis.delete(
        f'{query_id}_slots',
        f'{query_id}_plan',
        f'{query_id}_priorities',
        f'{query_id}_done',
    )
    for key, value in plan.items():
        await redis.hset(f'{query_id}_plan', key, json.dumps(value))
    await redis.hmset_dict(f'{query_id}_slots', slots)
    redis.close()

    # setup results DB
    sql = sqlite3.connect('results.db')
    column_names = ', '.join([f'`{qid}`' for qid in slots] + ['_score', '_timestamp'])
    columns = ', '.join([f'`{qid}` TEXT' for qid in slots] + ['_score REAL', '_timestamp REAL'])
    columns += f', UNIQUE({column_names})'
    statements = [
        f'DROP TABLE IF EXISTS `{query_id}`',
        f'CREATE TABLE `{query_id}` ({columns})',
    ]
    with sql:
        for statement in statements:
            sql.execute(statement)

    # create a RabbitMQ connection
    seconds = 1
    while True:
        try:
            connection = await aiormq.connect(f'amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:5672/%2F')
            break
        except ConnectionError as err:
            if seconds >= 65:
                raise err
            LOGGER.debug('Failed to connect to RabbitMQ. Trying again in %d seconds', seconds)
            await asyncio.sleep(seconds)
            seconds *= 2
    channel = await connection.channel()
    await setup_rabbitmq()

    # add a result for each named node
    # add a job for each named node
    for node in query_graph['nodes']:
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
        await channel.basic_publish(
            routing_key='jobs',
            body=json.dumps(job).encode('utf-8'),
        )

    return query_id
