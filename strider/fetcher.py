"""async fetcher (worker)."""
import asyncio
import itertools
import json
import logging
import os
import re
import time

from bmt import Toolkit as BMToolkit

from strider.scoring import score_graph
from strider.worker import Worker, Neo4jMixin, SqliteMixin
from strider.query import create_query
from strider.result import Result, ValidationError
from strider.kp_registry import Registry
from strider.util import is_neo4j_prop

KPREGISTRY_URL = os.getenv('KPREGISTRY_URL', 'http://localhost:4983')
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


def batches(arr, num):
    """Iterate over arr by batches of size n."""
    for idx in range(0, len(arr), num):
        yield arr[idx:idx + num]


class Fetcher(Worker, Neo4jMixin, SqliteMixin):
    """Asynchronous worker to consume jobs and publish results."""

    input_queue = 'jobs'

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.bmt = BMToolkit()
        self.neo4j = None
        self.query = None
        self.uid = kwargs.get('query_id')

        kp_registry = kwargs.get('kp_registry', None)
        if kp_registry is None:
            kp_registry = Registry(KPREGISTRY_URL)
        self.kp_registry = kp_registry
        self.counter = kwargs.get('counter', itertools.count())
        self.max_fanout = kwargs.get('max_fanout', 100)

    async def setup(self, *args):
        """Set up SQLite and Neo4j connections."""
        assert len(args) == 1
        qgraph = args[0]

        # Neo4j
        await self.setup_neo4j()
        await self.neo4j.run_async('MATCH (n) DETACH DELETE n')

        # initialize query stuff
        self.query = await create_query(qgraph, kp_registry=self.kp_registry)

        # set up a uniqueness constraint on Neo4j
        # without this, simultaneous MERGEs will create duplicate nodes
        statement = f'''
            CREATE CONSTRAINT
            ON (n:`{self.uid}`)
            ASSERT n.kid_qid IS UNIQUE'''
        await self.neo4j.run_async(statement)

    async def run(self, *args, **kwargs):
        """Run async consumer."""
        # SQLite
        await self.setup_sqlite()
        await super().run(*args, **kwargs)

    async def on_message(self, message):
        """Handle message from jobs queue.

        The message body should be a jsonified dictionary with fields:
        qid, kid
        """
        data = message

        if not data.get('step_id', None):
            result = {
                'edge_bindings': dict(),
                'node_bindings': {data['qid']: [{
                    'kg_id': data['kid'],
                }]},
            }
            kgraph = {
                'nodes': {data['kid']: {
                    'id': data['kid'],
                    'type': data['type'],
                }},
                'edges': dict(),
            }
            result = Result(result, self.query.qgraph, kgraph, self.bmt)
            job_id = f'({data["qid"]})'
            await self.process_kp_result(
                job_id, result,
            )
        else:
            await self.process_message(data, **self.query.options)

    async def process_message(self, data, **kwargs):
        """Process parsed message."""
        job_id = f'({data["kid"]}:{data["qid"]}{data["step_id"]})'

        step_awaitables = (
            self.take_step(job_id, data, endpoint, **kwargs)
            for endpoint in data['endpoints']
        )
        await asyncio.gather(
            *step_awaitables,
        )

    def get_kp_request(self, data):
        """Get request to send to KP."""
        match = re.fullmatch(r'<?-(\w+)->?(\w+)', data['step_id'])
        if match is None:
            raise ValueError(f'Cannot parse step id {data["step_id"]}')
        edge_qid = match[1]
        target_qid = match[2]

        # source, edge, and target specs
        edge_spec = self.query.qgraph['edges'][edge_qid]
        target_spec = self.query.qgraph['nodes'][target_qid]
        source_spec = self.query.qgraph['nodes'][data['qid']]

        return {
            "query_graph": {
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

    @log_exception
    async def take_step(self, job_id, data, endpoint, **kwargs):
        """Call specific endpoint."""
        request = self.get_kp_request(data)
        LOGGER.debug(
            '[query %s]: [job %s]: Calling KP...',
            self.uid,
            job_id,
        )
        response = await endpoint(request)

        await self.process_kp_response(
            job_id,
            response,
            **kwargs,
        )

    async def process_kp_response(self, job_id, response, **kwargs):
        """Process response from KP."""
        if response is None:
            return
        num_results = len(response['results'])
        if self.max_fanout >= 0:
            response['results'] = response['results'][:self.max_fanout]
        LOGGER.debug(
            '[query %s]: [job %s]: Processing %d of %d results...',
            self.uid,
            job_id,
            len(response['results']),
            num_results,
        )
        # process edges in batches
        batch_size = 100
        for results in batches(response['results'], batch_size):
            edge_awaitables = []
            for result in results:
                try:
                    result = Result(
                        result,
                        self.query.qgraph,
                        response['knowledge_graph'],
                        self.bmt,
                    )
                except ValidationError as err:
                    LOGGER.debug(
                        '[query %s]: [job %s]: Filtered out element: %s',
                        self.uid, job_id, err
                    )
                    continue
                edge_awaitables.append(self.process_kp_result(
                    job_id, result, **kwargs
                ))
            returned = await asyncio.gather(
                *edge_awaitables,
                return_exceptions=True,
            )
            for err in returned:
                if err is not None:
                    LOGGER.warning(err)

    async def process_kp_result(
            self,
            job_id,
            result,
            **kwargs
    ):
        """Process individual result from KP."""
        # store result in Neo4j
        subgraphs = await self.store_kp_result(
            result,
        )

        # process subgraphs
        await asyncio.gather(*[
            self.process_subgraph(job_id, subgraph, **kwargs)
            for subgraph in subgraphs
        ])

        # publish all nodes to jobs queue
        await asyncio.gather(*[
            self.queue_node_jobs(
                qid,
                node['id'],
                job_id,
                exclude_qedges=result.edges,
            )
            for qid, node in result.nodes.items()
            if node['id'] != ':'.join(job_id[1:-1].split(':')[:-1])
        ])

        # black-list any old jobs for these nodes
        # *not necessary if priority only increases

    async def store_kp_result(self, result):
        """Store result in Neo4j.

        Return new edges.
        """
        # merge with existing nodes/edge, but update edge weight
        node_vars = {
            (node['id'] + '_' + qid): f'n{i:03d}'
            for i, (qid, node) in enumerate(result.nodes.items())
        }
        edge_vars = {
            (edge['id'] + '_' + qid): f'e{i:03d}'
            for i, (qid, edge) in enumerate(result.edges.items())
        }
        statement = ''
        for qid, node in result.nodes.items():
            kid = node['id']
            node_var = node_vars[kid + '_' + qid]
            statement += 'MERGE ({0}:`{1}` {{kid_qid:"{2}_{3}"}})\n'.format(
                node_var,
                self.uid,
                kid,
                qid,
            )
            statement += f'SET {node_var}.qid = "{qid}"\n'
            statement += f'SET {node_var}.kid = "{kid}"\n'
            for key, value in node.items():
                if not is_neo4j_prop(value):
                    continue
                statement += 'SET {0}.{1} = {2}\n'.format(
                    node_var,
                    key,
                    json.dumps(value),
                )
        for qid, kedge in result.edges.items():
            qedge = self.query.qgraph['edges'][qid]
            kid = kedge['id']
            edge_var = edge_vars[kid + '_' + qid]
            try:
                statement += 'MERGE ({0})-[{1}:`{2}`]->({3})\n'.format(
                    node_vars[kedge['source_id'] + '_' + qedge['source_id']],
                    edge_var,
                    self.uid,
                    node_vars[kedge['target_id'] + '_' + qedge['target_id']],
                )
            except KeyError as err1:
                if 'type' in qedge:
                    raise err1
                try:
                    statement += 'MERGE ({0})-[{1}:`{2}`]->({3})\n'.format(
                        node_vars[kedge['source_id'] + '_' + qedge['target_id']],
                        edge_var,
                        self.uid,
                        node_vars[kedge['target_id'] + '_' + qedge['source_id']],
                    )
                except KeyError as err2:
                    raise err2
            statement += f'ON CREATE SET {edge_var}.new = TRUE\n'
            for key, value in kedge.items():
                if not is_neo4j_prop(value):
                    continue
                statement += f'SET {edge_var}.qid = "{qid}"\n'
                statement += f'SET {edge_var}.kid = "{kid}"\n'
                statement += 'SET {0}.{1} = {2}\n'.format(
                    edge_var,
                    key,
                    json.dumps(value),
                )
        statement += 'WITH [{0}] AS es UNWIND es as e\n'.format(
            ', '.join(edge_vars.values())
        )
        statement += 'WITH e WHERE e.new\n'
        statement += 'CALL strider.getPaths(e) YIELD nodes, edges\n'
        statement += 'REMOVE e.new\n'
        statement += 'RETURN nodes, edges'
        result = await self.neo4j.run_async(statement)
        return result

    async def process_subgraph(self, job_id, subgraph, **kwargs):
        """Process subgraph."""
        score = await score_graph(subgraph, self.query.qgraph, **kwargs)
        await self.update_priorities(subgraph, score)

        if (
                set(subgraph['nodes'].keys()) == set(self.query.qgraph['nodes'])
                and set(subgraph['edges'].keys()) == set(self.query.qgraph['edges'])
        ):
            await self.store_answer(
                job_id,
                subgraph, score,
            )

    async def update_priorities(self, subgraph, score):
        """Update job priorities."""
        # add the subgraph weight to each node's priority
        for node in subgraph['nodes'].values():
            await self.query.update_priority(
                f'({node["qid"]}:{node["kid"]})',
                score
            )

    async def queue_node_jobs(
            self,
            qid, kid, job_id,
            exclude_qedges=None,
    ):
        """Queue jobs from node."""
        if self.query.done[f'({qid}:{kid})']:
            return
        steps = await self.query.get_steps(qid, kid)
        priority = await self.query.get_priority(f'({qid}:{kid})')
        for step_id, endpoints in steps.items():
            if exclude_qedges:
                # do not retrace your steps
                match = re.fullmatch(r'<?-(\w+)->?(\w+)', step_id)
                if match is None:
                    raise ValueError(f'Cannot parse step id {step_id}')
                # edge_qid = match[1]
                if match[1] in exclude_qedges:
                    continue

            job = {
                'qid': qid,
                'kid': kid,
                'step_id': step_id,
                'endpoints': endpoints,
            }

            LOGGER.debug(
                "[query %s]: [job %s]: Queueing job (%s:%s%s) (priority %s)",
                self.uid, job_id, kid, qid, step_id, str(priority)
            )
            self.queue.put_nowait((
                -priority,
                next(self.counter),
                job,
            ))

    async def store_answer(self, job_id, answer, score):
        """Store answers in sqlite."""
        rows = []
        start_time = await self.query.get_start_time()
        slots = (
            [f'n_{qnode_id}' for qnode_id in self.query.qgraph['nodes']]
            + [f'e_{qedge_id}' for qedge_id in self.query.qgraph['edges']]
        )
        values = (
            [
                json.dumps(answer['nodes'][qid])
                for qid in self.query.qgraph['nodes']
            ]
            + [
                json.dumps(answer['edges'][qid])
                for qid in self.query.qgraph['edges']
            ]
            + [score, time.time() - start_time]
        )
        rows.append(values)
        LOGGER.debug(
            "[query %s]: [job %s]: Storing answer %s",
            self.uid,
            job_id,
            str(values),
        )
        columns = (
            [f'`{qid}`' for qid in slots]
            + ['_score', '_timestamp']
        )
        self.write_sql(columns, rows)

    def write_sql(self, columns, rows):
        """Write to SQL database."""
        placeholders = ', '.join(['?' for _ in range(len(columns))])
        self.sqlite.executemany(
            'INSERT OR IGNORE INTO `{0}` ({1}) VALUES ({2})'.format(
                self.uid,
                ', '.join(columns),
                placeholders,
            ),
            rows
        )
        self.sqlite.commit()

    async def teardown(self, *args):
        """Tear down."""
        await Worker.teardown(self, *args)
        await SqliteMixin.teardown(self, *args)
        LOGGER.debug(
            '[query %s]: Done.',
            self.uid,
        )
