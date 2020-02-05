"""Example: plan a query."""
import asyncio
import json
import logging
import sqlite3
import uuid

import aiormq
import aioredis
import uvloop

from strider.query_planner import generate_plan

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

LOGGER = logging.getLogger(__name__)


async def execute_query(query_graph):
    """Execute a user query.

    1) Generate execution plan.
    2) Store execution plan in Redis.
    3) Set up results SQL database.
    4) Add named nodes to job queue.
    """
    # generate query execution plan
    query_id = str(uuid.uuid4())
    plan = generate_plan(query_graph)
    slots = [node['id'] for node in query_graph['nodes']] + [edge['id'] for edge in query_graph['edges']]

    # store plan in Redis
    redis = await aioredis.create_redis_pool(
        'redis://localhost'
    )
    await redis.delete(
        f'{query_id}_slots',
        f'{query_id}_plan',
        f'{query_id}_priorities',
        f'{query_id}_done',
    )
    for key, value in plan.items():
        await redis.hset(f'{query_id}_plan', key, json.dumps(value))
    await redis.sadd(f'{query_id}_slots', *slots)
    redis.close()

    # setup results DB
    sql = sqlite3.connect('results.db')
    column_names = ', '.join([f'`{qid}`' for qid in slots])
    columns = ', '.join([f'`{qid}` text' for qid in slots])
    columns += f', UNIQUE({column_names})'
    statements = [
        f'DROP TABLE IF EXISTS `{query_id}`',
        f'CREATE TABLE `{query_id}` ({columns})',
    ]
    with sql:
        for statement in statements:
            sql.execute(statement)

    # create a RabbitMQ connection
    connection = await aiormq.connect("amqp://guest:guest@localhost:5672/%2F")
    channel = await connection.channel()

    # add a result for each named node
    # add a job for each named node 
    for node in query_graph['nodes']:
        if 'curie' not in node or node['curie'] is None:
            continue
        job = {
            'query_id': query_id,
            'qid': node['id'],
            'kid': node['curie'],
        }
        job_id = f'({node["curie"]}:{node["id"]})'
        LOGGER.debug("Queueing job %s", job_id)
        await channel.basic_publish(
            routing_key='jobs',
            body=json.dumps(job).encode('utf-8'),
            properties=aiormq.spec.Basic.Properties(
                priority=255,
            ),
        )
        result = {
            'query_id': query_id,
            'nodes': [
                {
                    'kid': node['curie'],
                    'qid': node['id']
                }
            ],
            'edges': []
        }
        LOGGER.debug("Queueing result %s", job_id)
        await channel.basic_publish(
            routing_key='results',
            body=json.dumps(result).encode('utf-8'),
        )

    return query_id


if __name__ == '__main__':
    import yaml
    with open('examples/query_graph2.yml', 'r') as f:
        query_graph = yaml.load(f, Loader=yaml.SafeLoader)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(execute_query(query_graph))
