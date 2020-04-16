"""async fetcher (worker)."""
import asyncio
import json
import logging
import re
import time

import aiormq
from bmt import Toolkit as BMToolkit
import httpx

from strider.scoring import score_graph, get_support
from strider.worker import Worker, Neo4jMixin, RedisMixin, SqliteMixin
from strider.query import create_query

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


class ValidationError(Exception):
    """Invalid node or edge."""


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

    async def is_done(self, plan, qid=None, kid=None):
        """Return True iff a job (qid/kid) has already been completed."""
        return bool(await self.redis.sismember(
            f'{plan}_done',
            f'({qid}:{kid})'
        ))

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
                    lineage = (
                        self.bmt.ancestors(value)
                        + self.bmt.descendents(value)
                        + [value]
                    )
                    if element['type'] not in lineage:
                        raise ValidationError(
                            f'{element["type"]} not in {lineage}'
                        )
                elif isinstance(element['type'], list):
                    if value not in element['type']:
                        raise ValidationError(
                            f'{value} not in {element["type"]}'
                        )
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
        query = await create_query(data.pop('query_id'), self.redis)
        statement = f'''
            CREATE CONSTRAINT
            ON (n:`{query.uid}`)
            ASSERT n.kid_qid IS UNIQUE'''
        await self.neo4j.run_async(statement)

        if not data.get('step_id', None):
            edge_bindings = dict()
            node_bindings = {
                data['qid']: {
                    'id': data['kid'],
                    'type': data['type'],
                }
            }
            job_id = f'({data["kid"]}:{data["qid"]})'
            await self.process_result(
                query, job_id, edge_bindings, node_bindings
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
        nodes_by_id = {
            node['id']: node
            for node in response['knowledge_graph']['nodes']
        }
        edges_by_id = {
            edge['id']: edge
            for edge in response['knowledge_graph']['edges']
        }
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
            edge_awaitables.append(self.process_result(
                query, job_id, edge_bindings, node_bindings, **kwargs
            ))
        await asyncio.gather(*edge_awaitables)

    async def process_result(
            self,
            query, job_id,
            edge_bindings, node_bindings,
            **kwargs
    ):
        """Process individual result from KP."""
        # filter out results that are incompatible with qgraph
        try:
            await self.validate_result(
                query, job_id,
                edge_bindings, node_bindings,
            )
        except ValidationError:
            return

        # store result in Neo4j
        new_edges = await self.store_result(
            query,
            node_bindings,
            edge_bindings,
        )

        # get subgraphs and scores
        qgraph = query.qgraph
        qnode_ids = list(qgraph['nodes'])
        qedge_ids = list(qgraph['edges'])

        subgraphs = await self.get_subgraphs(
            query,
            new_edges,
        )
        # in case the answer is just disconnected nodes
        subgraphs.append({
            'nodes': {
                qid: {**node, 'qid': qid, 'kid': node['id']}
                for qid, node in node_bindings.items()
            },
            'edges': {},
        })
        scores = await asyncio.gather(*[
            score_graph(subgraph, qgraph, **kwargs)
            for subgraph in subgraphs
        ])

        # update priorities
        await self.update_priorities(query, subgraphs, scores)

        # publish answers to the results DB
        answers = [
            (subgraph, score)
            for subgraph, score in zip(subgraphs, scores)
            if (
                set(subgraph['nodes'].keys()) == set(qnode_ids)
                and set(subgraph['edges'].keys()) == set(qedge_ids)
            )
        ]
        await self.store_answers(
            query, job_id,
            answers,
        )

        # publish all nodes to jobs queue
        await self.queue_jobs(query, node_bindings, edge_bindings, job_id)

        # black-list any old jobs for these nodes
        # *not necessary if priority only increases

    async def validate_result(
            self,
            query, job_id,
            edge_bindings, node_bindings,
    ):
        """Validate result nodes and edges against qgraph."""
        for qid, edge in edge_bindings.items():
            edge_spec = query.qgraph['edges'][qid]
            try:
                await self.validate(edge, edge_spec)
            except ValidationError as err:
                LOGGER.debug(
                    '[query %s]: [job %s]: Filtered out edge %s: %s',
                    query.uid, job_id, str(edge), err
                )
                raise err
        for qid, node in node_bindings.items():
            target_spec = query.qgraph['nodes'][qid]
            await self.validate(node, target_spec)

    async def update_priorities(self, query, subgraphs, scores):
        """Update job priorities."""
        # for each subgraph, add its weight to each component node's priority
        for subgraph, score in zip(subgraphs, scores):
            for node in subgraph['nodes'].values():
                await self.redis.hincrbyfloat(
                    f'{query.uid}_priorities',
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

    async def queue_jobs(self, query, node_bindings, edge_bindings, job_id):
        """Queue jobs from result."""
        node_steps = await self.get_jobs(query, node_bindings)

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
                if match[1] in edge_bindings:
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
            if not await self.is_done(
                query,
                qid=qid,
                kid=node['id'],
            )
        ]
        node_steps = []
        for qid, kid in nodes:
            job_id = f'({qid}:{kid})'

            # never process the same job twice
            if (
                    await self.redis.exists(f'{query.uid}_done')
                    and await self.redis.sismember(f'{query.uid}_done', job_id)
            ):
                continue
            await self.redis.sadd(f'{query.uid}_done', job_id)

            priority = min(255, int(float(await self.redis.hget(
                f'{query.uid}_priorities',
                job_id
            ))))

            # get step(s):
            steps_string = await self.redis.hget(
                f'{query.uid}_plan',
                qid,
                encoding='utf-8',
            )
            try:
                steps = json.loads(steps_string)
            except TypeError:
                steps = dict()
            node_steps.append((priority, qid, kid, steps))
        return node_steps

    async def store_result(self, query, node_bindings, edge_bindings):
        """Store result in Neo4j.

        Return new edges.
        """
        # merge with existing nodes/edge, but update edge weight
        node_vars = {
            node['id']: f'n{i:03d}'
            for i, node in enumerate(node_bindings.values())
        }
        edge_vars = {
            edge['id']: f'e{i:03d}'
            for i, edge in enumerate(edge_bindings.values())
        }
        statement = ''
        for qid, node in node_bindings.items():
            kid = node['id']
            statement += '\nMERGE ({0}:`{1}` {{kid_qid:"{2}_{3}"}})'.format(
                node_vars[kid],
                query.uid,
                kid,
                qid,
            )
            statement += f'\nSET {node_vars[kid]}.qid = "{qid}"'
            statement += f'\nSET {node_vars[kid]}.kid = "{kid}"'
            for key, value in node.items():
                statement += '\nSET {0}.{1} = {2}'.format(
                    node_vars[kid],
                    key,
                    json.dumps(value),
                )
        for qid, edge in edge_bindings.items():
            kid = edge['id']
            statement += '\nMERGE ({0})-[{1}:`{2}`]->({3})'.format(
                node_vars[edge['source_id']],
                edge_vars[kid],
                query.uid,
                node_vars[edge['target_id']],
            )
            statement += f'\nON CREATE SET {edge_vars[kid]}.new = TRUE'
            for key, value in edge.items():
                statement += f'\nSET {edge_vars[kid]}.qid = "{qid}"'
                statement += f'\nSET {edge_vars[kid]}.kid = "{kid}"'
                statement += '\nSET {0}.{1} = {2}'.format(
                    edge_vars[kid],
                    key,
                    json.dumps(value),
                )
        statement += '\nWITH [{0}] AS es UNWIND es as e'.format(
            ', '.join(edge_vars.values())
        )
        statement += '\nWITH e WHERE e.new'
        statement += '\nREMOVE e.new'
        statement += '\nRETURN e'
        result = await self.neo4j.run_async(statement)
        return [row['e'] for row in result]

    async def store_answers(self, query, job_id, answers):
        """Store answers in sqlite."""
        if not answers:
            return
        rows = []
        start_time = float(await self.redis.get(
            f'{query.uid}_starttime'
        ))
        slots = list(query.qgraph['nodes']) + list(query.qgraph['edges'])
        for answer, score in answers:
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
