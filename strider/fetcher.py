"""async fetcher (worker)."""
import json
import logging
import urllib

import httpx

from strider.worker import Worker, RedisMixin

LOGGER = logging.getLogger(__name__)


class Fetcher(Worker, RedisMixin):
    """Asynchronous worker to consume jobs and publish results."""

    IN_QUEUE = 'jobs'

    async def setup(self):
        """Set up Redis connection."""
        await self.setup_redis()

    async def on_message(self, message):
        """Handle message from jobs queue.

        The message body should be a jsonified dictionary with fields:
        qid, kid, query_id
        """
        if self.redis is None:
            await self.setup()
        data = json.loads(message.body)
        job_id = f'({data["kid"]}:{data["qid"]})'
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

        # iterator over steps
        for step_id, endpoints in steps.items():
            edge_qid, target_qid = step_id.split('/')
            for endpoint in endpoints:
                # call KP
                async with httpx.AsyncClient() as client:
                    r = await client.get(f'{endpoint}?source={urllib.parse.quote(data["kid"])}')
                results = r.json()

                for result in results:
                    # TODO: filter result
                    # e.g. edge type, target type, target CURIE, ...

                    # generate result struct
                    edge_kid = result[0]
                    target_kid = result[1]
                    result = {
                        'query_id': data['query_id'],
                        'nodes': [
                            {
                                'kid': data['kid'],
                                'qid': data['qid'],
                            },
                            {
                                'kid': target_kid,
                                'qid': target_qid,
                            },
                        ],
                        'edges': [
                            {
                                'kid': edge_kid,
                                'qid': edge_qid,
                                'source_id': data['kid'],
                                'target_id': target_kid,
                            },
                        ]
                    }
                    LOGGER.debug("[job %s]: Queueing result...", job_id)
                    # publish to results queue
                    await self.channel.basic_publish(
                        routing_key='results',
                        body=json.dumps(result).encode('utf-8'),
                    )
