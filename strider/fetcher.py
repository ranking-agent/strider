"""async fetcher (worker)."""
import asyncio
import json
import logging
import os
import re
import sqlite3
import time
import urllib

import aiormq
from bmt import Toolkit as BMToolkit
import httpx

from strider.neo4j import HttpInterface
from strider.scoring import score_graph
from strider.worker import Worker, RedisMixin

LOGGER = logging.getLogger(__name__)
NEO4J_HOST = os.getenv('NEO4J_HOST', 'localhost')


class ValidationError(Exception):
    """Invalid node or edge."""


class Fetcher(Worker, RedisMixin):
    """Asynchronous worker to consume jobs and publish results."""

    IN_QUEUE = 'jobs'

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.bmt = BMToolkit()
        self.neo4j = None
        self.sql = None

    async def setup(self):
        """Set up SQLite, Redis, and Neo4j connections."""
        # SQLite
        self.sql = sqlite3.connect('results.db')

        # Redis
        await self.setup_redis()

        # Neo4j
        self.neo4j = HttpInterface(
            url=f'http://{NEO4J_HOST}:7474',
        )
        seconds = 1
        while True:
            try:
                await self.neo4j.run_async('MATCH (n) DETACH DELETE n')  # clear it
                break
            except httpx.HTTPError as err:
                if seconds >= 129:
                    raise err
                LOGGER.debug('Failed to connect to Neo4j. Trying again in %d seconds', seconds)
                await asyncio.sleep(seconds)
                seconds *= 2

    async def is_done(self, plan, qid=None, kid=None):
        """Return True iff a job (qid/kid) has already been completed."""
        return bool(await self.redis.sismember(f'{plan}_done', f'({qid}:{kid})'))

    async def validate(self, element, spec):
        """Validate a node against a query-node specification."""
        for key, value in spec.items():
            if value is None:
                continue
            if key == 'curie':
                if element['id'] != value:
                    raise ValidationError(f'{element["id"]} != {value}')
            elif key == 'type':
                if isinstance(element['type'], str):
                    lineage = self.bmt.ancestors(value) + self.bmt.descendents(value) + [value]

                    # async with httpx.AsyncClient() as client:
                    #     response = await client.get(f'https://bl-lookup-sri.renci.org/bl/{value}/lineage?version=latest')
                    # assert response.status_code < 300
                    # lineage = response.json()
                    if element['type'] not in lineage:
                        raise ValidationError(f'{element["type"]} not in {lineage}')
                elif isinstance(element['type'], list):
                    if value not in element['type']:
                        raise ValidationError(f'{value} not in {element["type"]}')
                else:
                    raise ValueError('Type must be a str or list')
            elif key not in ['id', 'source_id', 'target_id']:
                if element[key] != value:
                    raise ValidationError(f'{element[key]} != {value}')
        return True

    async def on_message(self, message):
        """Handle message from jobs queue.

        The message body should be a jsonified dictionary with fields:
        qid, kid, query_id
        """
        if self.redis is None:
            await self.setup()
        data = json.loads(message.body)
        query_id = data.pop('query_id')
        statement = f'''
            CREATE CONSTRAINT
            ON (n:`{query_id}`)
            ASSERT n.kid_qid IS UNIQUE'''
        result = await self.neo4j.run_async(statement)

        try:
            if not data.get('step_id', None):
                edge_bindings = dict()
                node_bindings = {
                    data['qid']: {
                        'id': data['kid'],
                        'type': data['type'],
                    }
                }
                job_id = f'({data["kid"]}:{data["qid"]})'
                jobs = await self.process_result(query_id, job_id, edge_bindings, node_bindings)
            else:
                jobs = await self.process_message(query_id, data)

            # queue jobs in order of descending priority
            jobs.sort(key=lambda x: x['priority'], reverse=True)
            publish_awaitables = []
            for job in jobs:
                priority = job.pop('priority')
                publish_awaitables.append(self.channel.basic_publish(
                    routing_key='jobs',
                    body=json.dumps(job).encode('utf-8'),
                    properties=aiormq.spec.Basic.Properties(
                        priority=priority,
                    ),
                ))
            await asyncio.gather(*publish_awaitables)
        except Exception as err:
            LOGGER.exception(err)
            raise err

    async def process_message(self, query_id, data):
        """Process parsed message."""
        job_id = f'({data["kid"]}:{data["qid"]}{data["step_id"]})'
        LOGGER.debug("[job %s]: Processing...", job_id)

        step_awaitables = (
            self.take_step(query_id, job_id, data, endpoint)
            for endpoint in data['endpoints']
        )
        nested_jobs = await asyncio.gather(*step_awaitables)
        return [job for jobs in nested_jobs for job in jobs]

    async def take_step(self, query_id, job_id, data, endpoint):
        """Call specific endpoint."""
        match = re.fullmatch(r'<?-(\w+)->?(\w+)', data['step_id'])
        if match is None:
            raise ValueError(f'Cannot parse step id {data["step_id"]}')
        edge_qid = match[1]
        target_qid = match[2]

        # source, edge, and target specs
        edge_spec = await self.get_spec(query_id, edge_qid)
        target_spec = await self.get_spec(query_id, target_qid)
        source_spec = await self.get_spec(query_id, data['qid'])

        async with httpx.AsyncClient() as client:
            request = {
                "query_graph": {
                    # "nodes": [
                    #     source_spec,
                    #     target_spec,
                    # ],
                    # "edges": [
                    #     edge_spec
                    # ],
                    "nodes": [
                        {
                            "curie": data['kid'],
                            "id": source_spec['id'],
                            "type": source_spec['type'],
                        },
                        {
                            "id": target_spec['id'],
                            "type": target_spec['type'],
                        },
                    ],
                    "edges": [
                        {
                            "id": edge_spec['id'],
                            "source_id": edge_spec['source_id'],
                            "target_id": edge_spec['target_id'],
                            "type": edge_spec['type'],
                        },
                    ],
                }
            }
            response = await client.post(endpoint, json=request)
        assert response.status_code < 300

        return await self.process_response(query_id, job_id, response.json())

    async def get_spec(self, query_id, thing_qid):
        """Get specs for slot."""
        json_string = await self.redis.hget(
            f'{query_id}_slots',
            thing_qid,
            encoding='utf-8'
        )
        if json_string is None:
            raise ValueError(f'{query_id}_slots has no {thing_qid}')
        return json.loads(json_string)

    async def process_response(self, query_id, job_id, response):
        """Process response from KP."""
        if response is None:
            return
        nodes_by_id = {node['id']: node for node in response['knowledge_graph']['nodes']}
        edges_by_id = {edge['id']: edge for edge in response['knowledge_graph']['edges']}
        # process all edges, in parallel
        edge_awaitables = []
        for result in response['results']:
            edge_bindings = {
                binding['qg_id']: edges_by_id[binding['kg_id']]
                for binding in result['edge_bindings']
            }
            node_bindings = {
                binding['qg_id']: nodes_by_id[binding['kg_id']]
                for binding in result['node_bindings']
            }
            edge_awaitables.append(self.process_result(query_id, job_id, edge_bindings, node_bindings))
            # jobs.extend(await self.process_result(query_id, job_id, edge_bindings, node_bindings))
        nested_jobs = await asyncio.gather(*edge_awaitables)
        return [job for jobs in nested_jobs for job in jobs]

    async def process_result(self, query_id, job_id, edge_bindings, node_bindings):
        """Process individual result from KP."""
        # filter out results that are incompatible with qgraph
        try:
            await self.validate_result(query_id, job_id, edge_bindings, node_bindings)
        except ValidationError:
            return

        # reformat result
        data = {
            'query_id': query_id,
            'nodes': [
                {
                    'qid': key,
                    'kid': node['id'],
                    **{key: value for key, value in node.items() if key != 'id'},
                }
                for key, node in node_bindings.items()
            ],
            'edges': [
                {
                    'qid': key,
                    'kid': edge['id'],
                    # 'source_id': edge['source_id'],
                    # 'target_id': edge['target_id'],
                    **{key: value for key, value in edge.items() if key != 'id'}
                }
                for key, edge in edge_bindings.items()
            ]
        }

        node_ids = [f'({node["qid"]}:{node["kid"]})' for node in data['nodes']]
        edge_ids = [f'({edge["qid"]}:{edge["kid"]})' for edge in data['edges']]
        result_id = ', '.join(node_ids + edge_ids)
        LOGGER.debug("[result %s]: Processing...", result_id)

        # assign weights to edges
        data['edges'] = await self.assign_weights(data['edges'])

        # store result in Neo4j
        new_edges = await self.store_result(query_id, data)

        # get subgraphs and scores
        subgraphs = await self.get_subgraphs(query_id, data, new_edges)
        scores = [score_graph(subgraph) for subgraph in subgraphs]

        # update priorities
        await self.update_priorities(query_id, subgraphs, scores)

        # publish answers to the results DB
        slots = {
            x.decode('utf-8')
            for x in await self.redis.hkeys(f'{query_id}_slots')
        }
        answers = [
            (subgraph, score)
            for subgraph, score in zip(subgraphs, scores)
            if set(subgraph['nodes'].keys()) | set(subgraph['edges'].keys()) == slots
        ]
        if answers:
            LOGGER.debug(
                "[result %s]: Storing %d %s",
                result_id,
                len(answers),
                'answers' if len(answers) > 1 else 'answer'
            )
            self.store_answers(query_id, answers, slots)

        # publish all nodes to jobs queue
        return await self.queue_jobs(query_id, data, result_id)

        # black-list any old jobs for these nodes
        # *not necessary if priority only increases

    async def validate_result(self, query_id, job_id, edge_bindings, node_bindings):
        """Validate result nodes and edges against qgraph."""
        for qid, edge in edge_bindings.items():
            edge_spec = await self.get_spec(query_id, qid)
            try:
                await self.validate(edge, edge_spec)
            except ValidationError as err:
                LOGGER.debug('[job %s]: Filtered out edge %s: %s', job_id, str(edge), err)
                raise err
        for qid, node in node_bindings.items():
            target_spec = await self.get_spec(query_id, qid)
            await self.validate(node, target_spec)

    async def assign_weights(self, edges):
        """Assign weights to edges using OmniCorp."""
        for edge in edges:
            query = f'http://robokop.renci.org:3210/shared?curie={urllib.parse.quote(edge["source_id"])}&curie={urllib.parse.quote(edge["target_id"])}'
            async with httpx.AsyncClient() as client:
                response = await client.get(query)
            if response.status_code >= 300:
                raise RuntimeError(f'The following OmniCorp query returned a bad response:\n{query}')
            num_pubs = response.json()
            edge['weight'] = num_pubs + 1
        return edges

    async def update_priorities(self, query_id, subgraphs, scores):
        """Update job priorities."""
        # for each subgraph, add its weight to each component node's priority
        for subgraph, score in zip(subgraphs, scores):
            for node in subgraph['nodes'].values():
                # if await self.redis.hexists(f'{query_id}_done', node['label']):
                #     continue
                await self.redis.hincrbyfloat(
                    f'{query_id}_priorities',
                    f'({node["qid"]}:{node["kid"]})',
                    score
                )

    async def get_subgraphs(self, query_id, data, new_edges):
        """Get subgraphs."""
        subgraph_awaitables = []
        for edge in new_edges:
            statement = f'MATCH (:`{query_id}`)-[e:`{query_id}` {{kid:"{edge["kid"]}", qid:"{edge["qid"]}"}}]->(:`{query_id}`)'
            statement += '\nCALL strider.getPaths(e) YIELD nodes, edges RETURN nodes, edges'
            subgraph_awaitables.append(self.neo4j.run_async(statement))
        nested_results = await asyncio.gather(*subgraph_awaitables)
        subgraphs = [result for results in nested_results for result in results]

        # in case the answer is just disconnected nodes
        subgraphs.append({
            'nodes': {
                node['qid']: node
                for node in data['nodes']
            },
            'edges': {},
        })
        return subgraphs

    async def queue_jobs(self, query_id, data, result_id):
        """Queue jobs from result."""
        node_steps = await self.get_jobs(query_id, data)

        jobs = []
        for priority, qid, kid, steps in node_steps:
            job_id = f'({qid}:{kid})'
            LOGGER.debug("[result %s]: Queueing job(s) %s", result_id, job_id)
            for step_id, endpoints in steps.items():
                match = re.fullmatch(r'<?-(\w+)->?(\w+)', step_id)
                if match is None:
                    raise ValueError(f'Cannot parse step id {step_id}')
                edge_qid = match[1]
                # do not retrace your steps
                if edge_qid in [edge['qid'] for edge in data['edges']]:
                    continue

                job = {
                    'query_id': query_id,
                    'qid': qid,
                    'kid': kid,
                    'step_id': step_id,
                    'endpoints': endpoints,
                    'priority': priority,
                }
                jobs.append(job)
        return jobs

    async def get_jobs(self, query_id, data):
        """Get jobs for data nodes."""
        nodes = [
            (node['qid'], node['kid']) for node in data['nodes']
            if not await self.is_done(query_id, qid=node['qid'], kid=node['kid'])
        ]
        node_steps = []
        for qid, kid in nodes:
            job_id = f'({qid}:{kid})'

            # never process the same job twice
            if await self.redis.exists(f'{query_id}_done') and await self.redis.sismember(f'{query_id}_done', job_id):
                continue
            await self.redis.sadd(f'{query_id}_done', job_id)

            priority = min(255, int(float(await self.redis.hget(f'{query_id}_priorities', job_id))))

            # get step(s):
            steps_string = await self.redis.hget(f'{query_id}_plan', qid, encoding='utf-8')
            try:
                steps = json.loads(steps_string)
            except TypeError:
                steps = dict()
            # TODO: filter out steps that are too specific for source node
            # e.g. we got a disease_or_phenotypic_feature and step requires disease
            node_steps.append((priority, qid, kid, steps))
        return node_steps

    async def store_result(self, query_id, data):
        """Store result in Neo4j.

        Return new edges.
        """
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
            statement += f'\nMERGE ({node_vars[node["kid"]]}:`{query_id}` {{kid:"{node["kid"]}", qid:"{node["qid"]}", kid_qid:"{node["kid"]}_{node["qid"]}"}})'
        for edge in data['edges']:
            statement += f'\nMERGE ({node_vars[edge["source_id"]]})-[{edge_vars[edge["kid"]]}:`{query_id}` {{kid:"{edge["kid"]}", qid:"{edge["qid"]}"}}]->({node_vars[edge["target_id"]]})'
            statement += f'\nON CREATE SET {edge_vars[edge["kid"]]}.new = TRUE'
        for edge in data['edges']:
            statement += f'\nSET {edge_vars[edge["kid"]]}.weight = {edge["weight"]}'
        statement += f'\nWITH [{", ".join(edge_vars.values())}] AS es UNWIND es as e'
        statement += '\nWITH e WHERE e.new'
        statement += '\nREMOVE e.new'
        statement += '\nRETURN e'
        result = await self.neo4j.run_async(statement)
        return [row['e'] for row in result]

    def store_answers(self, query_id, answers, slots):
        """Store answers in sqlite."""
        rows = []
        for answer, score in answers:
            things = {**answer['nodes'], **answer['edges']}
            values = [things[qid]['kid'] for qid in slots] + [score, time.time()]
            rows.append(values)
        placeholders = ', '.join(['?' for _ in range(len(rows[0]))])
        columns = ', '.join([f'`{qid}`' for qid in slots] + ['_score', '_timestamp'])
        with self.sql:
            self.sql.executemany(
                f'''INSERT OR IGNORE INTO `{query_id}` ({columns}) VALUES ({placeholders})''',
                rows
            )
