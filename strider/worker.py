"""Base Worker class."""
from abc import ABC, abstractmethod
import asyncio
import logging
import os

import aioredis
import aiormq

from strider.rabbitmq import setup as setup_rabbitmq

LOGGER = logging.getLogger(__name__)
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_USER = os.getenv('RABBITMQ_USER', 'guest')
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD', 'guest')


class RedisMixin(ABC):
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
                LOGGER.debug('Failed to connect to Redis. Trying again in %d seconds', seconds)
                await asyncio.sleep(seconds)
                seconds *= 2


class Worker(ABC):
    """Asynchronous worker to consume messages from IN_QUEUE."""

    @property
    @abstractmethod
    def IN_QUEUE(self):
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
        seconds = 1
        while True:
            try:
                self.connection = await aiormq.connect(f'amqp://{RABBITMQ_USER}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:5672/%2F')
                break
            except ConnectionError as err:
                if seconds >= 65:
                    raise err
                LOGGER.debug('Failed to connect to RabbitMQ. Trying again in %d seconds', seconds)
                await asyncio.sleep(seconds)
                seconds *= 2

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
            await self.on_message(message)
        else:
            try:
                await self.on_message(message)
            finally:
                await self.ack(message)
        self.active_jobs -= 1

        # wait_between_comsumes = 1  # seconds
        # await asyncio.gather(
        #     self.ack(message, timeout=wait_between_consumes),
        #     self.on_message(message),
        # )

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
        consume_ok = await self.channel.basic_consume(
            self.IN_QUEUE, self._on_message
        )
