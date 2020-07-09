"""Example: plan a query."""
import asyncio
import itertools
import json
import logging
import os
import time
import uuid

import uvloop

from strider.fetcher import Fetcher
from strider.query_planner import generate_plan
from strider.results import Database

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

LOGGER = logging.getLogger(__name__)


async def execute_query(qgraph, **kwargs):
    """Execute a user query.

    1) Generate execution plan.
    2) Set up results database.
    3) Add named nodes to job queue.
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

    # setup results DB
    slots = (
        [f'n_{qnode_id}' for qnode_id in qgraph['nodes']]
        + [f'e_{qedge_id}' for qedge_id in qgraph['edges']]
    )
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
            'qid': node['id'],
            'kid': node['curie'],
            **{
                key: value for key, value in node.items()
                if key not in ('id', 'curie')
            },
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
    await fetcher.run(qgraph, wait=kwargs.get('wait', False))

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
