"""Test prioritizer."""
import asyncio
import logging
import logging.config

import httpx
import uvloop
import yaml

from strider.fetcher import Fetcher
from strider.neo4j import HttpInterface
from strider.prioritizer import Prioritizer

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')


async def setup_tools():
    """Set up RabbitMQ and Neo4j."""
    async with httpx.AsyncClient() as client:
        # add strider exchange to RabbitMQ
        r = await client.put(
            r'http://localhost:15672/api/exchanges/%2f/strider',
            json={"type": "direct", "durable": True},
            auth=('guest', 'guest'),
        )
        assert r.status_code < 300
        # add jobs queue to RabbitMQ
        r = await client.put(
            r'http://localhost:15672/api/queues/%2f/jobs',
            json={"durable": True, "arguments": {"x-max-priority": 255}},
            auth=('guest', 'guest'),
        )
        assert r.status_code < 300
        # add results queue to RabbitMQ
        r = await client.put(
            r'http://localhost:15672/api/queues/%2f/results',
            json={"durable": True},
            auth=('guest', 'guest'),
        )
        assert r.status_code < 300

    neo4j = HttpInterface(
        url=f'http://localhost:7474',
    )
    neo4j.run('MATCH (n) DETACH DELETE n')


async def main():
    """Test."""
    fetcher = Fetcher()
    prioritizer = Prioritizer()
    await fetcher.run()
    await prioritizer.run()

    print('Ready.')


if __name__ == "__main__":
    with open('logging_setup.yml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    logging.config.dictConfig(config)
    # logging.basicConfig(
    #     filename='logs/strider.log',
    #     format=LOG_FORMAT,
    #     level=logging.DEBUG,
    # )

    # start event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.run_forever()
