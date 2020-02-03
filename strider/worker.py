"""Base Worker class."""
from abc import ABC, abstractmethod
import logging

import aiormq

LOGGER = logging.getLogger(__name__)


class Worker():
    """Asynchronous worker to consume messages from IN_QUEUE."""

    @property
    @abstractmethod
    def IN_QUEUE(self):
        """Name of the queue from which this worker will consume."""

    def __init__(self):
        """Initialize."""
        self.connection = None
        self.channel = None
        self._is_connected = False

    async def connect(self):
        """Connect to RabbitMQ."""
        if self._is_connected:
            return

        # Perform connection
        self.connection = await aiormq.connect("amqp://guest:guest@localhost:5672/%2F")

        # Creating a channel
        self.channel = await self.connection.channel()

        self._is_connected = True

    @abstractmethod
    async def on_message(self, message):
        """Handle message from results queue."""

    async def run(self):
        """Run async RabbitMQ consumer."""
        await self.connect()
        consume_ok = await self.channel.basic_consume(
            self.IN_QUEUE, self.on_message, no_ack=True
        )
