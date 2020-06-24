"""KP registry."""
from collections import defaultdict
import json
import logging

import aiosqlite
from fastapi import HTTPException
import httpx
import sqlite3

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
        return response.json()

    async def add(self, kps):
        """Add KP(s)."""
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

    async def search(self, source_types, edge_types, target_types):
        """Search for KPs matching a pattern."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{self.url}/search',
                json={
                    'source_type': source_types,
                    'target_type': target_types,
                    'edge_type': edge_types,
                }
            )
            assert response.status_code < 300
        return response.json()

    async def delete_all(self):
        """Delete all KPs."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{self.url}/clear',
            )
            assert response.status_code < 300
        return response.json()

    async def call(self, url, request):
        """Call a KP."""
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                response = await client.post(url, json=request)
            except httpx.ReadTimeout:
                LOGGER.error(
                    "ReadTimeout: endpoint: %s, JSON: %s",
                    url, json.dumps(request)
                )
                return []
        assert response.status_code < 300
        return response
