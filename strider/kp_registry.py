"""KP registry."""
import json
import logging

import httpx

LOGGER = logging.getLogger(__name__)


async def call_kp(url, request):
    """Call KP.

    Input request can be a string or JSON-able object.
    Output is the raw text of the response body.
    """
    if not isinstance(request, str):
        request = json.dumps(request).encode('utf-8')
    async with httpx.AsyncClient(
            timeout=None,
            headers={'Content-Type': 'application/json'},
    ) as client:
        try:
            response = await client.post(url, data=request)
        except httpx.ReadTimeout:
            LOGGER.error(
                "ReadTimeout: endpoint: %s, JSON: %s",
                url, json.dumps(request)
            )
            return []
    if response.status_code >= 300:
        LOGGER.error(
            'KP error (%s):\n'
            'request:\n%s\n'
            'response:\n%s',
            url,
            request,
            response.text,
        )
        raise RuntimeError('KP call failed.')
    return response.text


def kp_func(
        url,
        preferred_prefix=None,
        in_transformers=None,
        out_transformers=None,
):
    """Generate KP function."""
    async def func(request):
        """Call KP with pre- and post-processing."""
        request = {'message': request}
        request = json.dumps(request)
        response = await call_kp(url, request)
        response = json.loads(response)
        return response
    return func


class Registry():
    """KP registry."""

    def __init__(self, url):
        """Initialize."""
        self.url = url
        self.locals = dict()

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
        return kp_func(
            url,
            preferred_prefix=provider[5]['details']['preferred_prefix'],
            in_transformers=provider[5]['details']['in_transformers'],
            out_transformers=provider[5]['details']['out_transformers'],
        )

    async def add(self, *kps):
        """Add KP(s)."""
        kps = {
            kp.name: await kp.get_operations()
            for kp in kps
        }
        self.locals.update({
            kp.name: kp
            for kp in kps
        })
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
        return [
            kp_func(url, **details)
            for url, details in response.json().items()
        ]

    async def delete_all(self):
        """Delete all KPs."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{self.url}/clear',
            )
            assert response.status_code < 300
        return response.json()
