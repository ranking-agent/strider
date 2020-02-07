"""Set up RabbitMQ."""
import asyncio
import os

import httpx
import uvloop

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_USER = os.getenv('RABBITMQ_USER', 'localhost')
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD', 'localhost')


async def setup():
    """Set up RabbitMQ."""
    async with httpx.AsyncClient() as client:
        # add strider exchange to RabbitMQ
        r = await client.put(
            rf'http://{RABBITMQ_HOST}:15672/api/exchanges/%2f/strider',
            json={"type": "direct", "durable": True},
            auth=(RABBITMQ_USER, RABBITMQ_PASSWORD),
        )
        assert r.status_code < 300
        # add jobs queue to RabbitMQ
        r = await client.put(
            rf'http://{RABBITMQ_HOST}:15672/api/queues/%2f/jobs',
            json={"durable": True, "arguments": {"x-max-priority": 255}},
            auth=(RABBITMQ_USER, RABBITMQ_PASSWORD),
        )
        assert r.status_code < 300
        # add results queue to RabbitMQ
        r = await client.put(
            rf'http://{RABBITMQ_HOST}:15672/api/queues/%2f/results',
            json={"durable": True},
            auth=(RABBITMQ_USER, RABBITMQ_PASSWORD),
        )
        assert r.status_code < 300


if __name__ == "__main__":
    # start event loop
    asyncio.run(setup())
