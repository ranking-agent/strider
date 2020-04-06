"""Neo4j Interfaces."""
from abc import ABC
import logging
from urllib.parse import urlparse

import httpx

LOGGER = logging.getLogger(__name__)


class HttpInterface(ABC):
    """HTTP interface to Neo4j database."""

    def __init__(self, url=None, credentials=None):
        """Initialize."""
        url = urlparse(url)
        self.url = 'http://{0}:{1}/db/data/transaction/commit'.format(
            url.hostname,
            url.port,
        )
        if credentials is not None:
            self.auth = (credentials['username'], credentials['password'])
        else:
            self.auth = None

    def run(self, statement):
        """Run statement."""
        response = httpx.post(
            self.url,
            auth=self.auth,
            json={"statements": [{"statement": statement}]},
        )
        assert response.status_code < 300
        result = response.json()['results'][0]
        result = [
            dict(zip(result['columns'], datum['row']))
            for datum in result['data']
        ]
        return result

    async def run_async(self, statement):
        """Run statement."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.url,
                auth=self.auth,
                json={"statements": [{"statement": statement}]},
            )
        assert response.status_code < 300
        response = response.json()
        if response['errors']:
            raise RuntimeError(response['errors'])
        results = response['results']
        if not results:
            return []
        result = results[0]
        result = [
            dict(zip(result['columns'], datum['row']))
            for datum in result['data']
        ]
        return result
