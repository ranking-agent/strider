"""Base Worker class."""
from abc import ABC, abstractmethod
import asyncio
import logging
import os

import aioredis
import aiosqlite
import httpx

from strider.neo4j import HttpInterface
from strider.rabbitmq import connect_to_rabbitmq, setup as setup_rabbitmq

LOGGER = logging.getLogger(__name__)
NEO4J_HOST = os.getenv('NEO4J_HOST', 'localhost')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_USER = os.getenv('RABBITMQ_USER', 'guest')
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD', 'guest')


class Neo4jMixin(ABC):  # pylint: disable=too-few-public-methods
    """Mixin to hold a Neo4j database connection."""

    def __init__(self):
        """Initialize."""
        self.neo4j = None

    async def setup_neo4j(self):
        """Set up Neo4j connection."""
        self.neo4j = HttpInterface(
            url=f'http://{NEO4J_HOST}:7474',
        )
        seconds = 1
        while True:
            try:
                # clear it
                await self.neo4j.run_async('CALL dbms.procedures()')
                break
            except httpx.HTTPError as err:
                if seconds >= 129:
                    raise err
                LOGGER.debug(
                    'Failed to connect to Neo4j. Trying again in %d seconds',
                    seconds
                )
                await asyncio.sleep(seconds)
                seconds *= 2


class SqliteMixin(ABC):  # pylint: disable=too-few-public-methods
    """Mixin to hold a SQLite database connection."""

    def __init__(self):
        """Initialize."""
        self.sqlite = None

    async def setup_sqlite(self):
        """Set up SQLite connection."""
        self.sqlite = await aiosqlite.connect('results.db')


class RedisMixin(ABC):  # pylint: disable=too-few-public-methods
    """Mixin to hold a Redis database connection."""

    def __init__(self):
        """Initialize."""
        self.redis = None

    async def setup_redis(self):
        """Set up Redis connection."""
        seconds = 1
        while True:
            try:
                self.redis = await aioredis.create_redis_pool(
                    f'redis://{REDIS_HOST}',
                    encoding='utf-8'
                )
                break
            except (ConnectionError, OSError) as err:
                if seconds > 65:
                    raise err
                LOGGER.debug(
                    'Failed to connect to Redis. Trying again in %d seconds',
                    seconds,
                )
                await asyncio.sleep(seconds)
                seconds *= 2


class Worker(ABC):
    """Asynchronous worker to consume messages from input_queue."""

    def __init__(self, queue, max_jobs=-1):
        """Initialize."""
        self.queue = queue
        self.max_jobs = max_jobs

    @abstractmethod
    async def on_message(self, message):
        """Handle message from results queue."""

    async def consume(self):
        """Consume messages."""
        while True:
            # wait for an item from the producer
            _, item = await self.queue.get()

            # process message
            await self.on_message(item)

            # Notify the queue that the item has been processed
            self.queue.task_done()

    @abstractmethod
    async def setup(self, *args):
        """Set up services."""

    async def run(self):
        """Run async RabbitMQ consumer."""
        await self.setup()
        # schedule the consumers
        # create three worker tasks to process the queue concurrently
        tasks = [
            asyncio.create_task(self.consume())
            for _ in range(self.max_jobs)
        ]
        # wait until the consumer has processed all items
        await self.queue.join()
        # the consumers are still waiting for items, cancel them
        for consumer in tasks:
            consumer.cancel()
