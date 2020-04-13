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

    @property
    @abstractmethod
    def input_queue(self):
        """Name of the queue from which this worker will consume."""

    def __init__(self, max_jobs=-1):
        """Initialize."""
        self.connection = None
        self.channel = None
        self._is_connected = False
        self.max_jobs = max_jobs
        self.active_jobs = 0

    async def connect(self):
        """Connect to RabbitMQ."""
        if self._is_connected:
            return

        # Perform connection
        self.connection = await connect_to_rabbitmq()

        # Creating a channel
        self.channel = await self.connection.channel()
        await self.channel.basic_qos(prefetch_count=1)

        await setup_rabbitmq()

        self._is_connected = True

    async def _on_message(self, message):
        """Handle message from results queue."""
        self.active_jobs += 1
        if self.max_jobs > 0 and self.active_jobs < self.max_jobs:
            await self.ack(message)
            try:
                await self.on_message(message)
            except Exception as err:
                LOGGER.exception(err)
                raise err
        else:
            try:
                await self.on_message(message)
            except Exception as err:
                LOGGER.exception(err)
                raise err
            finally:
                await self.ack(message)
        self.active_jobs -= 1

    async def ack(self, message, timeout=0):
        """Wait for timeout, then ack."""
        if timeout:
            await asyncio.sleep(timeout)
        await message.channel.basic_ack(
            message.delivery.delivery_tag
        )

    @abstractmethod
    async def on_message(self, message):
        """Handle message from results queue."""

    async def run(self):
        """Run async RabbitMQ consumer."""
        await self.connect()
        await self.channel.basic_consume(
            self.input_queue, self._on_message
        )
