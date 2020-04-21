"""async fetcher (worker)."""
import asyncio
import json
import logging
import re
import time

import aiormq
from bmt import Toolkit as BMToolkit
import httpx

from strider.scoring import score_graph
from strider.worker import Worker, Neo4jMixin, RedisMixin, SqliteMixin
from strider.query import create_query
from strider.result import Result, ValidationError

LOGGER = logging.getLogger(__name__)


def log_exception(method):
    """Wrap method."""
    async def wrapper(*args, **kwargs):
        """Log exception encountered in method, then pass."""
        try:
            return await method(*args, **kwargs)
        except Exception as err:  # pylint: disable=broad-except
            LOGGER.exception(err)
    return wrapper


class Fetcher(Worker, Neo4jMixin, RedisMixin, SqliteMixin):
    """Asynchronous worker to consume jobs and publish results."""

    input_queue = 'jobs'

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.bmt = BMToolkit()
        self.neo4j = None

    async def setup(self):
        """Set up SQLite, Redis, and Neo4j connections."""
        # SQLite
        await self.setup_sqlite()

        # Redis
        await self.setup_redis()

        # Neo4j
        await self.setup_neo4j()
        self.neo4j.run_async('MATCH (n) DETACH DELETE n')

    async def on_message(self, message):
        """Handle message from jobs queue.

        The message body should be a jsonified dictionary with fields:
        qid, kid, query_id
        """
        if self.redis is None:
            await self.setup()
        data = json.loads(message.body)
        query = await create_query(data.pop('query_id'), self.redis)
        statement = f'''
            CREATE CONSTRAINT
            ON (n:`{query.uid}`)
            ASSERT n.kid_qid IS UNIQUE'''
        await self.neo4j.run_async(statement)

        if not data.get('step_id', None):
            result = {
                'edge_bindings': [],
                'node_bindings': [{
                    'qg_id': data['qid'],
                    'kg_id': data['kid'],
                }],
            }
            kgraph = {
                'nodes': {data['kid']: {
                    'id': data['kid'],
                    'type': data['type'],
                }},
                'edges': [],
            }
            result = Result(result, {'knowledge_graph': kgraph}, self.bmt)
            job_id = f'({data["kid"]}:{data["qid"]})'
            await self.process_result(
                query, job_id, result,
            )
        else:
            await self.process_message(query, data, **query.options)

    async def process_message(self, query, data, **kwargs):
        """Process parsed message."""
        job_id = f'({data["kid"]}:{data["qid"]}{data["step_id"]})'
        LOGGER.debug("[query %s]: [job %s]: Starting...", query.uid, job_id)

        step_awaitables = (
            self.take_step(query, job_id, data, endpoint, **kwargs)
            for endpoint in data['endpoints']
        )
        await asyncio.gather(
            *step_awaitables,
        )

    @log_exception
    async def take_step(self, query, job_id, data, endpoint, **kwargs):
        """Call specific endpoint."""
        match = re.fullmatch(r'<?-(\w+)->?(\w+)', data['step_id'])
        if match is None:
            raise ValueError(f'Cannot parse step id {data["step_id"]}')
        edge_qid = match[1]
        target_qid = match[2]

        # source, edge, and target specs
        edge_spec = query.qgraph['edges'][edge_qid]
        target_spec = query.qgraph['nodes'][target_qid]
        source_spec = query.qgraph['nodes'][data['qid']]

        async with httpx.AsyncClient(timeout=None) as client:
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
            try:
                response = await client.post(endpoint, json=request)
            except httpx.ReadTimeout:
                LOGGER.error(
                    "ReadTimeout: endpoint: %s, JSON: %s",
                    endpoint, json.dumps(request)
                )
                return []
        assert response.status_code < 300

        await self.process_response(
            query, job_id,
            response.json(),
            **kwargs,
        )

    async def process_response(self, query, job_id, response, **kwargs):
        """Process response from KP."""
        if response is None:
            return
        response['knowledge_graph'] = {
            'nodes': {
                node['id']: node
                for node in response['knowledge_graph']['nodes']
            },
            'edges': {
                edge['id']: edge
                for edge in response['knowledge_graph']['edges']
            },
        }
        # process all edges, in parallel
        edge_awaitables = []
        for result in response['results']:
            result = Result(result, response, self.bmt)
            edge_awaitables.append(self.process_result(
                query, job_id, result, **kwargs
            ))
        await asyncio.gather(*edge_awaitables)

    async def process_result(
            self,
            query, job_id,
            result,
            **kwargs
    ):
        """Process individual result from KP."""
        # filter out results that are incompatible with qgraph
        try:
            result.validate(query)
        except ValidationError as err:
            LOGGER.debug(
                '[query %s]: [job %s]: Filtered out element: %s',
                query.uid, job_id, err
            )
            return

        # store result in Neo4j
        new_edges = await self.store_result(
            query,
            result,
        )

        # get subgraphs and scores
        subgraphs = await self.get_subgraphs(
            query,
            new_edges,
        )
        # in case the answer is just disconnected nodes
        subgraphs.append({
            'nodes': {
                qid: {**node, 'qid': qid, 'kid': node['id']}
                for qid, node in result.nodes.items()
            },
            'edges': {},
        })
        await asyncio.gather(*[
            self.process_subgraph(query, job_id, subgraph, **kwargs)
            for subgraph in subgraphs
        ])

        # publish all nodes to jobs queue
        await self.queue_jobs(query, result, job_id)

        # black-list any old jobs for these nodes
        # *not necessary if priority only increases

    async def process_subgraph(self, query, job_id, subgraph, **kwargs):
        """Process subgraph."""
        score = await score_graph(subgraph, query.qgraph, **kwargs)
        await self.update_priorities(query, subgraph, score)

        if (
                set(subgraph['nodes'].keys()) == set(query.qgraph['nodes'])
                and set(subgraph['edges'].keys()) == set(query.qgraph['edges'])
        ):
            await self.store_answer(
                query, job_id,
                subgraph, score,
            )

    async def update_priorities(self, query, subgraph, score):
        """Update job priorities."""
        # add the subgraph weight to each node's priority
        for node in subgraph['nodes'].values():
            await query.update_priority(
                f'({node["qid"]}:{node["kid"]})',
                score
            )

    async def get_subgraphs(self, query, new_edges):
        """Get subgraphs."""
        subgraph_awaitables = []
        for edge in new_edges:
            statement = 'MATCH (:`{0}`)-[e {{kid:"{1}", qid:"{2}"}}]->()\n' \
                        'CALL strider.getPaths(e) YIELD nodes, edges\n' \
                        'RETURN nodes, edges'.format(
                            query.uid, edge['kid'], edge['qid']
                        )
            subgraph_awaitables.append(self.neo4j.run_async(statement))
        nested_results = await asyncio.gather(*subgraph_awaitables)
        subgraphs = [
            result
            for results in nested_results
            for result in results
        ]

        return subgraphs

    async def queue_jobs(self, query, result, job_id):
        """Queue jobs from result."""
        node_steps = await self.get_jobs(query, result.nodes)

        publish_awaitables = []
        for priority, qid, kid, steps in node_steps:
            LOGGER.debug(
                "[query %s]: [job %s]: Queueing job(s) (%s:%s)",
                query.uid, job_id, qid, kid
            )
            for step_id, endpoints in steps.items():
                # do not retrace your steps
                match = re.fullmatch(r'<?-(\w+)->?(\w+)', step_id)
                if match is None:
                    raise ValueError(f'Cannot parse step id {step_id}')
                # edge_qid = match[1]
                if match[1] in result.edges:
                    continue

                job = {
                    'query_id': query.uid,
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

    async def get_jobs(self, query, node_bindings):
        """Get jobs for data nodes."""
        nodes = [
            (qid, node['id']) for qid, node in node_bindings.items()
            if not await query.is_done(f'({qid}:{node["id"]})')
        ]
        node_steps = []
        for qid, kid in nodes:
            job_id = f'({qid}:{kid})'

            # never process the same job twice
            if await query.is_done(job_id):
                continue
            await query.finish(job_id)

            priority = await query.get_priority(job_id)

            # get step(s):
            steps_string = await query.get_steps(qid)
            try:
                steps = json.loads(steps_string)
            except TypeError:
                steps = dict()
            node_steps.append((priority, qid, kid, steps))
        return node_steps

    async def store_result(self, query, result):
        """Store result in Neo4j.

        Return new edges.
        """
        # merge with existing nodes/edge, but update edge weight
        node_vars = {
            node['id']: f'n{i:03d}'
            for i, node in enumerate(result.nodes.values())
        }
        edge_vars = {
            edge['id']: f'e{i:03d}'
            for i, edge in enumerate(result.edges.values())
        }
        statement = ''
        for qid, node in result.nodes.items():
            kid = node['id']
            statement += 'MERGE ({0}:`{1}` {{kid_qid:"{2}_{3}"}})\n'.format(
                node_vars[kid],
                query.uid,
                kid,
                qid,
            )
            statement += f'SET {node_vars[kid]}.qid = "{qid}"\n'
            statement += f'SET {node_vars[kid]}.kid = "{kid}"\n'
            for key, value in node.items():
                statement += 'SET {0}.{1} = {2}\n'.format(
                    node_vars[kid],
                    key,
                    json.dumps(value),
                )
        for qid, edge in result.edges.items():
            kid = edge['id']
            statement += 'MERGE ({0})-[{1}:`{2}`]->({3})\n'.format(
                node_vars[edge['source_id']],
                edge_vars[kid],
                query.uid,
                node_vars[edge['target_id']],
            )
            statement += f'ON CREATE SET {edge_vars[kid]}.new = TRUE\n'
            for key, value in edge.items():
                statement += f'SET {edge_vars[kid]}.qid = "{qid}"\n'
                statement += f'SET {edge_vars[kid]}.kid = "{kid}"\n'
                statement += 'SET {0}.{1} = {2}\n'.format(
                    edge_vars[kid],
                    key,
                    json.dumps(value),
                )
        statement += 'WITH [{0}] AS es UNWIND es as e\n'.format(
            ', '.join(edge_vars.values())
        )
        statement += 'WITH e WHERE e.new\n'
        statement += 'REMOVE e.new\n'
        statement += 'RETURN e'
        result = await self.neo4j.run_async(statement)
        return [row['e'] for row in result]

    async def store_answer(self, query, job_id, answer, score):
        """Store answers in sqlite."""
        rows = []
        start_time = await query.get_start_time()
        slots = list(query.qgraph['nodes']) + list(query.qgraph['edges'])
        things = {**answer['nodes'], **answer['edges']}
        values = (
            [json.dumps(things[qid]) for qid in slots]
            + [score, time.time() - start_time]
        )
        rows.append(values)
        LOGGER.debug(
            "[query %s]: [job %s]: Storing answer %s",
            query.uid,
            job_id,
            str(values),
        )
        placeholders = ', '.join(['?' for _ in range(len(rows[0]))])
        columns = ', '.join(
            [f'`{qid}`' for qid in slots]
            + ['_score', '_timestamp']
        )
        await self.sqlite.executemany(
            'INSERT OR IGNORE INTO `{0}` ({1}) VALUES ({2})'.format(
                query.uid,
                columns,
                placeholders,
            ),
            rows
        )
        await self.sqlite.commit()
