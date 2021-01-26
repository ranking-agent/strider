import logging
import httpx

LOGGER = logging.getLogger(__name__)


class Normalizer():
    def __init__(self, url):
        self.url = url

    async def get_types(self, curies):
        """Get types for a given curie"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f'{self.url}/get_normalized_nodes',
                params=dict(curie=curies)
            )
        types = []
        if response.status_code >= 300:
            LOGGER.warning(
                f"Received {response.status_code} response from normalizer: "
                + response.text
            )
            return types
        results = response.json()
        for c in curies:
            types.extend(results[c]['type'])
        return types
