"""async fetcher (worker)."""
import asyncio
import json
import logging
import urllib

import httpx

from strider.worker import Worker, RedisMixin

LOGGER = logging.getLogger(__name__)


class InvalidNode(Exception):
    """Invalid node."""


class Fetcher(Worker, RedisMixin):
    """Asynchronous worker to consume jobs and publish results."""

    IN_QUEUE = 'jobs'

    async def setup(self):
        """Set up Redis connection."""
        await self.setup_redis()

    async def validate(self, node, spec):
        """Validate a node against a query-node specification."""
        for key, value in spec.items():
            if value is None:
                continue
            if key == 'id' or key == 'source_id' or key == 'target_id':
                continue
            if key == 'curie':
                if node['id'] != value:
                    raise InvalidNode(f'{node["id"]} != {value}')
                continue
            if key == 'type':
                async with httpx.AsyncClient() as client:
                    response = await client.get(f'https://bl-lookup-sri.renci.org/bl/{value}/lineage?version=latest')
                assert response.status_code < 300
                lineage = response.json()
                if node[key] not in lineage:
                    raise InvalidNode(f'{node[key]} not in {lineage}')
                continue
            if node[key] != value:
                raise InvalidNode(f'{node[key]} != {value}')
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
        LOGGER.debug("[job %s]: processing...", job_id)

        # never process the same job (node) twice
        if await self.redis.exists(f'{data["query_id"]}_done') and await self.redis.sismember(f'{data["query_id"]}_done', job_id):
            return
        await self.redis.sadd(f'{data["query_id"]}_done', job_id)

        # get step(s):
        steps_string = await self.redis.hget(f'{data["query_id"]}_plan', data['qid'], encoding='utf-8')
        try:
            steps = json.loads(steps_string)
        except TypeError:
            steps = dict()

        # TODO: filter out steps that are too specific for source node
        # e.g. we got a disease_or_phenotypic_feature and step requires disease

        # iterator over steps
        for step_id, endpoints in steps.items():
            await self.take_step(job_id, data, step_id, endpoints)

    async def take_step(self, job_id, data, step_id, endpoints):
        """Query a KP and handle results."""
        edge_qid, target_qid = step_id.split('/')
        # get slot and generate filter
        edge_spec = json.loads(await self.redis.hget(
            f'{data["query_id"]}_slots',
            edge_qid,
            encoding='utf-8'
        ))
        target_spec = json.loads(await self.redis.hget(
            f'{data["query_id"]}_slots',
            target_qid,
            encoding='utf-8'
        ))
        async with httpx.AsyncClient() as client:
            awaitables = (
                client.get(f'{endpoint}?source={urllib.parse.quote(data["kid"])}')
                for endpoint in endpoints
            )
            responses = await asyncio.gather(*awaitables)
            assert all(response.status_code < 300 for response in responses)
        for response in responses:
            message = response.json()

            await self.process_message(job_id, data, edge_spec, target_spec, message)

    async def process_message(self, job_id, data, edge_spec, target_spec, message):
        """Process message from KP."""
        if message is None:
            return
        nodes_by_id = {node['id']: node for node in message['knowledge_graph']['nodes']}
        for edge in message['knowledge_graph']['edges']:
            # TODO: fix if kedge is not aligned with qedge
            target = nodes_by_id[edge['target_id']]

            await self.process_edge(job_id, data, edge, edge_spec, target, target_spec)

    async def process_edge(self, job_id, data, edge, edge_spec, target, target_spec):
        """Process edge from KP."""
        # filter out results that are incompatible with qgraph
        try:
            await self.validate(edge, edge_spec)
        except InvalidNode as err:
            LOGGER.debug('Filtered out edge %s: %s', str(edge), err)
            return
        try:
            await self.validate(target, target_spec)
        except InvalidNode as err:
            LOGGER.debug('Filtered out target %s: %s', str(target), err)
            return

        # generate result struct
        result = {
            'query_id': data['query_id'],
            'nodes': [
                {
                    'kid': data['kid'],
                    'qid': data['qid'],
                },
                {
                    'kid': target['id'],
                    'qid': target_spec['id'],
                },
            ],
            'edges': [
                {
                    'kid': edge['id'],
                    'qid': edge_spec['id'],
                    'source_id': data['kid'],
                    'target_id': target['id'],
                },
            ]
        }
        LOGGER.debug("[job %s]: Queueing result...", job_id)
        # publish to results queue
        await self.channel.basic_publish(
            routing_key='results',
            body=json.dumps(result).encode('utf-8'),
        )
