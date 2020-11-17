"""KP registry."""
import logging
from typing import Union

import httpx

LOGGER = logging.getLogger(__name__)


class Registry():
    """KP registry."""

    def __init__(self, url):
        """Initialize."""
        self.url = url

    async def __aenter__(self):
        """Enter context."""
        return self

    async def __aexit__(self, *args):
        """Exit context."""

    async def setup(self):
        """Set up database table."""

    async def get_all(self):
        """Get all KPs."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f'{self.url}/kps',
            )
            assert response.status_code < 300
        return response.json()

    async def get_one(self, url):
        """Get a specific KP."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f'{self.url}/kps/{url}',
            )
            assert response.status_code < 300
        provider = response.json()
        return provider[5]['details']

    async def add(self, **kps):
        """Add KP(s)."""
        # kps = {
        #     kp.name: await kp.get_operations()
        #     for kp in kps
        # }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{self.url}/kps',
                json=kps
            )
            assert response.status_code < 300

    async def delete_one(self, url):
        """Delete a specific KP."""
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f'{self.url}/kps/{url}',
            )
            assert response.status_code < 300

    async def search(
            self,
            source_types: Union[str, list[str]] = None,
            edge_types: Union[str, list[str]] = None,
            target_types: Union[str, list[str]] = None,
            allowlist=None, denylist=None,
    ):
        """Search for KPs matching a pattern."""
        if isinstance(source_types, str):
            source_types = [source_types]
        if isinstance(edge_types, str):
            edge_types = [edge_types]
        if isinstance(target_types, str):
            target_types = [target_types]
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{self.url}/search',
                json={
                    'source_type': source_types,
                    'target_type': target_types,
                    'edge_type': edge_types,
                }
            )
            response.raise_for_status()
        return {
            kpid: details
            for kpid, details in response.json().items()
            if (
                (allowlist is None or kpid in allowlist)
                and (denylist is None or kpid not in denylist)
            )
        }

    async def delete_all(self):
        """Delete all KPs."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{self.url}/clear',
            )
            assert response.status_code < 300
        return response.json()
