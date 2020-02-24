"""async fetcher (worker)."""
import asyncio
import json
import logging
import re

import httpx
from bmt import Toolkit as BMToolkit

from strider.worker import Worker, RedisMixin

LOGGER = logging.getLogger(__name__)


class InvalidNode(Exception):
    """Invalid node."""


class Fetcher(Worker, RedisMixin):
    """Asynchronous worker to consume jobs and publish results."""

    IN_QUEUE = 'jobs'

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.bmt = BMToolkit()

    async def setup(self):
        """Set up Redis connection."""
        await self.setup_redis()

    async def validate(self, element, spec):
        """Validate a node against a query-node specification."""
        for key, value in spec.items():
            if value is None:
                continue
            if key == 'curie':
                if element['id'] != value:
                    raise InvalidNode(f'{element["id"]} != {value}')
            elif key == 'type':
                if isinstance(element['type'], str):
                    lineage = self.bmt.ancestors(value) + self.bmt.descendents(value) + [value]

                    # async with httpx.AsyncClient() as client:
                    #     response = await client.get(f'https://bl-lookup-sri.renci.org/bl/{value}/lineage?version=latest')
                    # assert response.status_code < 300
                    # lineage = response.json()
                    if element['type'] not in lineage:
                        raise InvalidNode(f'{element["type"]} not in {lineage}')
                elif isinstance(element['type'], list):
                    if value not in element['type']:
                        raise InvalidNode(f'{value} not in {element["type"]}')
                else:
                    raise ValueError('Type must be a str or list')
            elif key not in ['id', 'source_id', 'target_id']:
                if element[key] != value:
                    raise InvalidNode(f'{element[key]} != {value}')
        return True

    async def on_message(self, message):
        """Handle message from jobs queue.

        The message body should be a jsonified dictionary with fields:
        qid, kid, query_id
        """
        if self.redis is None:
            await self.setup()
        data = json.loads(message.body)
        job_id = f'({data["kid"]}:{data["qid"]}{data["step_id"]})'
        LOGGER.debug("[job %s]: Processing...", job_id)

        step_awaitables = (
            self.take_step(job_id, data, endpoint)
            for endpoint in data['endpoints']
        )
        await asyncio.gather(*step_awaitables)

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

    async def take_step(self, job_id, data, endpoint):
        """Call specific endpoint."""
        match = re.fullmatch(r'<?-(\w+)->?(\w+)', data['step_id'])
        if match is None:
            raise ValueError(f'Cannot parse step id {data["step_id"]}')
        edge_qid = match[1]
        target_qid = match[2]

        # source, edge, and target specs
        edge_spec = await self.get_spec(data['query_id'], edge_qid)
        target_spec = await self.get_spec(data['query_id'], target_qid)
        source_spec = await self.get_spec(data['query_id'], data['qid'])

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
        await self.process_response(job_id, data, response.json())

    async def process_response(self, job_id, data, response):
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
            edge_awaitables.append(self.process_edge(job_id, data, edge_bindings, node_bindings))
        await asyncio.gather(*edge_awaitables)

    async def process_edge(self, job_id, data, edge_bindings, node_bindings):
        """Process edge from KP."""
        # filter out results that are incompatible with qgraph
        for qid, edge in edge_bindings.items():
            edge_spec = await self.get_spec(data['query_id'], qid)
            try:
                await self.validate(edge, edge_spec)
            except InvalidNode as err:
                LOGGER.debug('[job %s]: Filtered out edge %s: %s', job_id, str(edge), err)
                return
        for qid, node in node_bindings.items():
            target_spec = await self.get_spec(data['query_id'], qid)
            try:
                await self.validate(node, target_spec)
            except InvalidNode as err:
                # LOGGER.debug('[job %s]: Filtered out node %s: %s', job_id, str(node), err)
                return

        # generate result struct
        # TODO: fix bindings structure and handling
        result = {
            'query_id': data['query_id'],
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
        LOGGER.debug("[job %s]: Queueing result...", job_id)
        # publish to results queue
        await self.channel.basic_publish(
            routing_key='results',
            body=json.dumps(result).encode('utf-8'),
        )
