"""Neo4j Interfaces."""
from abc import ABC, abstractmethod
import logging
from urllib.parse import urlparse

import httpx

LOGGER = logging.getLogger(__name__)


class Neo4jInterface(ABC):
    """Abstract interface to Neo4j database."""

    def __init__(self, url=None, credentials=None, **kwargs):
        """Initialize."""
        url = urlparse(url)
        self.hostname = url.hostname
        self.port = url.port
        if credentials is not None:
            self.auth = (credentials['username'], credentials['password'])
        else:
            self.auth = None

    @abstractmethod
    def run(self, statement, *args):
        """Run statement."""


class HttpInterface(Neo4jInterface):
    """HTTP interface to Neo4j database."""

    def __init__(self, **kwargs):
        """Initialize."""
        super().__init__(**kwargs)
        self.url = f'http://{self.hostname}:{self.port}/db/data/transaction/commit'

    def run(self, statement, *args):
        """Run statement."""
        response = httpx.post(
            self.url,
            auth=self.auth,
            json={"statements": [{"statement": statement}]},
        )
        result = response.json()['results'][0]
        result = [
            dict(zip(result['columns'], datum['row']))
            for datum in result['data']
        ]
        return result

    async def run_async(self, statement, *args):
        """Run statement."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.url,
                auth=self.auth,
                json={"statements": [{"statement": statement}]},
            )
        result = response.json()['results'][0]
        result = [
            dict(zip(result['columns'], datum['row']))
            for datum in result['data']
        ]
        return result
