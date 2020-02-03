"""Example: plan a query."""
import asyncio
import json
import sqlite3

import aioredis
import uvloop
import yaml

from strider.query_planner import generate_plan

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


async def main():
    """Generate plan and put in Redis."""
    # get user query
    with open('examples/query_graph2.yml', 'r') as f:
        query_graph = yaml.load(f, Loader=yaml.SafeLoader)

    slots = [node['id'] for node in query_graph['nodes']] + [edge['id'] for edge in query_graph['edges']]

    execution_plan = 'alpha'

    plan = generate_plan(query_graph)
    print(plan)

    redis = await aioredis.create_redis_pool(
        'redis://localhost'
    )
    await redis.delete(f'{execution_plan}_slots', f'{execution_plan}_plan')
    for key, value in plan.items():
        await redis.hset(f'{execution_plan}_plan', key, json.dumps(value))
    await redis.sadd(f'{execution_plan}_slots', *slots)
    redis.close()

    sql = sqlite3.connect('results.db')
    column_names = ', '.join([f'`{qid}`' for qid in slots])
    columns = ', '.join([f'`{qid}` text' for qid in slots])
    columns += f', UNIQUE({column_names})'
    statements = [
        f'DROP TABLE IF EXISTS `{execution_plan}`',
        f'CREATE TABLE `{execution_plan}` ({columns})',
    ]
    with sql:
        for statement in statements:
            sql.execute(statement)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
