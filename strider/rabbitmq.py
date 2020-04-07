"""Set up RabbitMQ."""
import asyncio
import logging
import os

import aiormq
import httpx
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
LOGGER = logging.getLogger(__name__)
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_USER = os.getenv('RABBITMQ_USER', 'localhost')
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD', 'localhost')


async def connect_to_rabbitmq():
    """Connect to RabbitMQ."""
    seconds = 1
    while True:
        try:
            connection = await aiormq.connect(
                'amqp://{0}:{1}@{2}:5672/%2F'.format(
                    RABBITMQ_USER,
                    RABBITMQ_PASSWORD,
                    RABBITMQ_HOST,
                )
            )
            break
        except ConnectionError as err:
            if seconds >= 65:
                raise err
            LOGGER.debug(
                'Failed to connect to RabbitMQ. Trying again in %d seconds',
                seconds,
            )
            await asyncio.sleep(seconds)
            seconds *= 2
    return connection


async def setup():
    """Set up RabbitMQ."""
    async with httpx.AsyncClient() as client:
        # add strider exchange to RabbitMQ
        response = await client.put(
            rf'http://{RABBITMQ_HOST}:15672/api/exchanges/%2f/strider',
            json={"type": "direct", "durable": True},
            auth=(RABBITMQ_USER, RABBITMQ_PASSWORD),
        )
        assert response.status_code < 300
        # add jobs queue to RabbitMQ
        response = await client.put(
            rf'http://{RABBITMQ_HOST}:15672/api/queues/%2f/jobs',
            json={"durable": True, "arguments": {"x-max-priority": 255}},
            auth=(RABBITMQ_USER, RABBITMQ_PASSWORD),
        )
        assert response.status_code < 300


if __name__ == "__main__":
    # start event loop
    asyncio.run(setup())
