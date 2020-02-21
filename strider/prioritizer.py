"""async prioritizer (worker)."""
import asyncio
import json
import logging
import os
import re
import sqlite3

import aiormq

from strider.neo4j import HttpInterface
from strider.scoring import score_graph
from strider.worker import Worker, RedisMixin

LOGGER = logging.getLogger(__name__)
NEO4J_HOST = os.getenv('NEO4J_HOST', 'localhost')


class Prioritizer(Worker, RedisMixin):
    """Asynchronous worker to consume results and publish jobs."""

    IN_QUEUE = 'results'

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.neo4j = None
        self.sql = None

    async def setup(self):
        """Set up SQLite, Redis, and Neo4j connections."""
        # SQLite
        self.sql = sqlite3.connect('results.db')

        # Redis
        await self.setup_redis()

        # Neo4j
        seconds = 1
        while True:
            try:
                self.neo4j = HttpInterface(
                    url=f'http://{NEO4J_HOST}:7474',
                )
                break
            except (ConnectionError, OSError) as err:
                if seconds >= 129:
                    raise err
                LOGGER.debug('Failed to connect to Neo4j. Trying again in %d seconds', seconds)
                await asyncio.sleep(seconds)
                seconds *= 2
        await self.neo4j.run_async('MATCH (n) DETACH DELETE n')  # clear it

    async def is_done(self, plan, qid=None, kid=None):
        """Return True iff a job (qid/kid) has already been completed."""
        return bool(await self.redis.sismember(f'{plan}_done', f'({qid}:{kid})'))

    async def on_message(self, message):
        """Handle message from results queue.

        The message should be jsonified dictionary:
        type: object
        properties:
          query_id:
            type: string
          source:
            type: object
            properties:
              kid:
                type: string
              qid:
                type: string
          edge:
            type: object
            properties:
              kid:
                type: string
              qid:
                type: string
          target:
            type: object
            properties:
              kid:
                type: string
              qid:
                type: string
        """
        if self.neo4j is None:
            await self.setup()

        # parse message
        data = json.loads(message.body)
        result_id = ', '.join(
            [f'({node["qid"]}:{node["kid"]})' for node in data['nodes']]
            + [f'({edge["qid"]}:{edge["kid"]})' for edge in data['edges']]
        )
        LOGGER.debug("[result %s]: Processing...", result_id)

        # store result
        # merge with existing nodes/edge, but update edge weight
        node_vars = {
            node['kid']: f'n{i:03d}'
            for i, node in enumerate(data['nodes'])
        }
        edge_vars = {
            edge['kid']: f'e{i:03d}'
            for i, edge in enumerate(data['edges'])
        }
        statement = ''
        for node in data['nodes']:
            statement += f'\nMERGE ({node_vars[node["kid"]]}:`{data["query_id"]}` {{kid:"{node["kid"]}", qid:"{node["qid"]}"}})'
        for edge in data['edges']:
            statement += f'\nMERGE ({node_vars[edge["source_id"]]})-[{edge_vars[edge["kid"]]}:`{data["query_id"]}` {{kid:"{edge["kid"]}", qid:"{edge["qid"]}"}}]->({node_vars[edge["target_id"]]})'
            statement += f'\nON CREATE SET {edge_vars[edge["kid"]]}.new = TRUE'
        for edge in data['edges']:
            statement += f'\nSET {edge_vars[edge["kid"]]}.weight = {len(edge["publications"])}'
        statement += f'\nWITH [{", ".join(edge_vars.values())}] AS es UNWIND es as e'
        statement += '\nWITH e WHERE e.new'
        statement += '\nREMOVE e.new'
        statement += '\nRETURN e'
        result = await self.neo4j.run_async(statement)
        new_edges = [row['e'] for row in result]

        # compute priority:
        # get subgraphs
        subgraphs = []
        subgraph_awaitables = []
        for edge in new_edges:
            statement = f'MATCH (:`{data["query_id"]}`)-[e:`{data["query_id"]}` {{kid:"{edge["kid"]}"}}]->(:`{data["query_id"]}`)'
            statement += '\nCALL strider.getPaths(e) YIELD nodes, edges RETURN nodes, edges'
            subgraph_awaitables.append(self.neo4j.run_async(statement))
        nested_results = await asyncio.gather(*subgraph_awaitables)

        subgraphs.extend(result for results in nested_results for result in results)

        # in case the answer is just disconnected nodes
        subgraphs.append({
            'nodes': {
                node['qid']: node
                for node in data['nodes']
            },
            'edges': {},
        })

        slots = {
            x.decode('utf-8')
            for x in await self.redis.hkeys(f'{data["query_id"]}_slots')
        }

        # compute scores
        scores = [score_graph(subgraph) for subgraph in subgraphs]
        # for each subgraph, add its weight to each component node's priority
        answers = []
        for subgraph, score in zip(subgraphs, scores):
            for node in subgraph['nodes'].values():
                # if await self.redis.hexists(f'{data["query_id"]}_done', node['label']):
                #     continue
                await self.redis.hincrbyfloat(f'{data["query_id"]}_priorities', f'({node["qid"]}:{node["kid"]})', score)
            subgraph_things = {qid for qid in subgraph['nodes'].keys()}
            subgraph_things |= {qid for qid in subgraph['edges'].keys()}
            if subgraph_things == slots:
                answers.append(subgraph)

        # publish answers to the results DB
        if answers:
            LOGGER.debug("[result %s]: Storing answers %s", result_id, str(answers))
            rows = []
            for answer in answers:
                things = {**answer['nodes'], **answer['edges']}
                values = [things[qid]['kid'] for qid in slots]
                rows.append(values)
            placeholders = ', '.join(['?' for _ in range(len(rows[0]))])
            columns = ', '.join([f'`{qid}`' for qid in slots])
            with self.sql:
                self.sql.executemany(
                    f'''INSERT OR IGNORE INTO `{data['query_id']}` ({columns}) VALUES ({placeholders})''',
                    rows
                )

        # publish all nodes to jobs queue
        nodes = [
            (node['qid'], node['kid']) for node in data['nodes']
            if not await self.is_done(data["query_id"], qid=node['qid'], kid=node['kid'])
        ]
        publish_awaitables = []
        for qid, kid in nodes:
            job_id = f'({qid}:{kid})'

            # TODO: track done-ness by step, not node - it matters where we come from
            # never process the same job (node) twice
            if await self.redis.exists(f'{data["query_id"]}_done') and await self.redis.sismember(f'{data["query_id"]}_done', job_id):
                continue
            await self.redis.sadd(f'{data["query_id"]}_done', job_id)

            LOGGER.debug("[result %s]: Queueing job(s) %s", result_id, job_id)
            priority = min(255, int(float(await self.redis.hget(f'{data["query_id"]}_priorities', job_id))))

            # get step(s):
            steps_string = await self.redis.hget(f'{data["query_id"]}_plan', qid, encoding='utf-8')
            try:
                steps = json.loads(steps_string)
            except TypeError:
                steps = dict()

            # TODO: filter out steps that are too specific for source node
            # e.g. we got a disease_or_phenotypic_feature and step requires disease
            for step_id, endpoints in steps.items():
                match = re.fullmatch(r'<?-(\w+)->?(\w+)', step_id)
                if match is None:
                    raise ValueError(f'Cannot parse step id {step_id}')
                edge_qid = match[1]
                # target_qid = match[2]
                if edge_qid in [edge['qid'] for edge in data['edges']]:
                    continue

                job = {
                    'query_id': data['query_id'],
                    'qid': qid,
                    'kid': kid,
                    'step_id': step_id,
                    'endpoints': endpoints,
                }
                publish_awaitables.append(self.channel.basic_publish(
                    routing_key='jobs',
                    body=json.dumps(job).encode('utf-8'),
                    properties=aiormq.spec.Basic.Properties(
                        priority=priority,
                    ),
                ))
        await asyncio.gather(*publish_awaitables)
        return

        # black-list any old jobs for these nodes
        # *not necessary if priority only increases
