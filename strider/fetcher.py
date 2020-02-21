"""async fetcher (worker)."""
import asyncio
import json
import logging
import re
import urllib

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
        job_id = f'({data["qid"]}:{data["kid"]})'
        LOGGER.debug("[job %s]: Processing...", job_id)

        # get step(s):
        steps_string = await self.redis.hget(f'{data["query_id"]}_plan', data['qid'], encoding='utf-8')
        try:
            steps = json.loads(steps_string)
        except TypeError:
            steps = dict()

        # TODO: filter out steps that are too specific for source node
        # e.g. we got a disease_or_phenotypic_feature and step requires disease

        # iterator over steps
        step_awaitables = [
            self.take_steps(job_id, data, step_id, endpoints)
            for step_id, endpoints in steps.items()
        ]
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

    async def take_steps(self, job_id, data, step_id, endpoints):
        """Query a KP and handle results."""
        match = re.fullmatch(r'<?-(\w+)->?(\w+)', step_id)
        if match is None:
            raise ValueError(f'Cannot parse step id {step_id}')
        edge_qid = match[1]
        target_qid = match[2]

        # edge and target specs
        edge_spec = await self.get_spec(data['query_id'], edge_qid)
        target_spec = await self.get_spec(data['query_id'], target_qid)

        # send and process all API requests, in parallel
        # async with httpx.AsyncClient() as client:
        step_awaitables = (
            self.take_step(job_id, data, edge_spec, target_spec, endpoint)
            for endpoint in endpoints
        )
        await asyncio.gather(*step_awaitables)

    async def take_step(self, job_id, data, edge_spec, target_spec, endpoint):
        """Call specific endpoint."""
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
                    # TODO: qg_id -> id
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
            # response = await client.get(f'{endpoint}?source={urllib.parse.quote(data["kid"])}')
            # response = await client.get(f'{endpoint}/{urllib.parse.quote(data["kid"])}')
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
                key: edges_by_id[value]
                for key, value in result['edge_bindings'].items()
            }
            node_bindings = {
                key: nodes_by_id[value]
                for key, value in result['node_bindings'].items()
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
                    'kid': value.pop('id'),
                    **value,
                }
                for key, value in node_bindings.items()
            ],
            'edges': [
                {
                    'qid': key,
                    'kid': value.pop('id'),
                    'source_id': value.pop('source_id'),
                    'target_id': value.pop('target_id'),
                    **value,
                }
                for key, value in edge_bindings.items()
            ]
        }
        LOGGER.debug("[job %s]: Queueing result...", job_id)
        # publish to results queue
        await self.channel.basic_publish(
            routing_key='results',
            body=json.dumps(result).encode('utf-8'),
        )
