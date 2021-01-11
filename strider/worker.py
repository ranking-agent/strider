"""Base Worker class."""
from abc import ABC, abstractmethod
import asyncio
from contextlib import suppress
import itertools
import logging
import sqlite3
from typing import List

LOGGER = logging.getLogger(__name__)


class SqliteMixin(ABC):  # pylint: disable=too-few-public-methods
    """Mixin to hold a SQLite database connection."""

    def __init__(self):
        """Initialize."""
        self.sqlite = None

    async def setup_sqlite(self):
        """Set up SQLite connection."""
        self.sqlite = sqlite3.connect('results.db')

    async def teardown(self, *args):
        """Tear down SQLite connection."""
        self.sqlite.close()


class Worker(ABC):
    """Asynchronous worker to consume messages from input_queue."""

    def __init__(
            self,
            num_workers: int = 1,
            **kwargs,
    ):
        """Initialize."""
        self.num_workers = num_workers

        self.queue = asyncio.PriorityQueue()
        self.counter = itertools.count()

    @abstractmethod
    async def on_message(self, message):
        """Handle message from results queue."""

    async def _on_message(self, message):
        """Handle message from results queue."""
        try:
            await self.on_message(message)
        except Exception as err:
            LOGGER.exception(
                "Aborted processing of %s due to error: %s",
                message,
                str(err),
            )

    async def consume(self):
        """Consume messages."""
        while True:
            # wait for an item from the producer
            _, item = await self.queue.get()

            # process message
            await self._on_message(item)

            # Notify the queue that the item has been processed
            self.queue.task_done()

    @abstractmethod
    async def setup(self, *args):
        """Set up services."""

    async def finish(self, tasks: List[asyncio.Task]):
        """Wait for Strider to finish, then tear everything down."""
        # wait until the consumer has processed all items
        await self.queue.join()
        await self.teardown(tasks)

    async def teardown(self, tasks: List[asyncio.Task]):
        """Tear down consumers after queue is emptied."""
        # the consumers are still waiting for items, cancel them
        for consumer in tasks:
            consumer.cancel()

        for consumer in tasks:
            with suppress(asyncio.CancelledError):
                await consumer

    async def run(self, *args, wait: bool = False, **kwargs):
        """Run async consumer."""
        # schedule the consumers
        # create max_jobs worker tasks to process the queue concurrently
        tasks = [
            asyncio.create_task(self.consume())
            for _ in range(self.num_workers)
        ]
        finish = asyncio.create_task(self.finish(tasks))
        if wait:
            await finish

    async def put(self, message, priority=0):
        """Put message on queue."""
        await self.queue.put(((priority, next(self.counter)), message))
