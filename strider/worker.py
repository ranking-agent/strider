"""Base Worker class."""
from abc import ABC, abstractmethod
import asyncio
from contextlib import suppress
import logging
import os
import sqlite3

import httpx

from strider.neo4j import HttpInterface

LOGGER = logging.getLogger(__name__)
NEO4J_HOST = os.getenv('NEO4J_HOST', 'localhost')


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
        self.sqlite = sqlite3.connect('results.db')

    async def teardown(self, *args):
        """Tear down SQLite connection."""
        self.sqlite.close()


class Worker(ABC):
    """Asynchronous worker to consume messages from input_queue."""

    def __init__(self, queue, max_jobs=-1, **kwargs):
        """Initialize."""
        self.queue = queue
        self.max_jobs = max_jobs

    @abstractmethod
    async def on_message(self, message):
        """Handle message from results queue."""

    async def _on_message(self, message):
        """Handle message from results queue."""
        try:
            await self.on_message(message)
        except Exception as err:
            LOGGER.exception(err)
            raise

    async def consume(self):
        """Consume messages."""
        while True:
            # wait for an item from the producer
            _, _, item = await self.queue.get()

            # process message
            await self._on_message(item)

            # Notify the queue that the item has been processed
            self.queue.task_done()

    @abstractmethod
    async def setup(self, *args):
        """Set up services."""

    async def finish(self, tasks):
        """Wait for Strider to finish, then tear everything down."""
        # wait until the consumer has processed all items
        await self.queue.join()
        await self.teardown(tasks)

    async def teardown(self, tasks):
        """Tear down consumers after queue is emptied."""
        # the consumers are still waiting for items, cancel them
        for consumer in tasks:
            consumer.cancel()

        for consumer in tasks:
            with suppress(asyncio.CancelledError):
                await consumer

    async def run(self, *args):
        """Run async consumer."""
        await self.setup(*args)
        # schedule the consumers
        # create three worker tasks to process the queue concurrently
        tasks = [
            asyncio.create_task(self.consume())
            for _ in range(self.max_jobs)
        ]
        finish = asyncio.create_task(self.finish(tasks))
        await finish
