"""async fetcher (worker)."""
import asyncio
import json
import logging
import urllib

import aioredis
import httpx

from strider.worker import Worker

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)


class Fetcher(Worker):
    """Asynchronous worker to consume jobs and publish results."""

    IN_QUEUE = 'jobs'

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.redis = None

    async def setup(self):
        """Set up Redis connection."""
        self.redis = await aioredis.create_redis_pool(
            'redis://localhost'
        )

    async def on_message(self, message):
        """Handle message from jobs queue.

        The message body should be a jsonified dictionary with fields:
        qid, kid, execution_plan
        """
        if self.redis is None:
            await self.setup()
        data = json.loads(message.body)
        job_id = f'({data["kid"]}:{data["qid"]})'
        LOGGER.debug("[job %s]: processing...", job_id)

        # never process the same job (node) twice
        if await self.redis.exists(f'{data["execution_plan"]}_done') and await self.redis.sismember(f'{data["execution_plan"]}_done', job_id):
            return
        await self.redis.sadd(f'{data["execution_plan"]}_done', job_id)

        # get step(s):
        plan = json.loads(await self.redis.get(data['execution_plan'], encoding='utf-8'))
        steps = plan[data['qid']]

        # iterator over steps
        for step_id, endpoints in steps.items():
            edge_qid, target_qid = step_id.split('/')
            for endpoint in endpoints:
                # call KP
                async with httpx.AsyncClient() as client:
                    r = await client.get(f'{endpoint}?source={urllib.parse.quote(data["kid"])}')
                results = r.json()

                for result in results:
                    # generate result struct
                    edge_kid = result[0]
                    target_kid = result[1]
                    result = {
                        'execution_plan': data['execution_plan'],
                        'source': {
                            'kid': data['kid'],
                            'qid': data['qid']
                        },
                        'target': {
                            'kid': target_kid,
                            'qid': target_qid
                        },
                        'edge': {
                            'kid': edge_kid,
                            'qid': edge_qid
                        }
                    }
                    # publish to results queue
                    await self.channel.basic_publish(
                        routing_key='results',
                        body=json.dumps(result).encode('utf-8'),
                    )
        # LOGGER.debug("After fetching!")


if __name__ == "__main__":
    # add FileHandler to root logger
    logging.basicConfig(
        filename='logs/strider.log',
        format=LOG_FORMAT,
        level=logging.DEBUG,
    )

    fetcher = Fetcher()

    # start event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(fetcher.run())
    loop.run_forever()
