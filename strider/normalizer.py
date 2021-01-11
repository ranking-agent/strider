import httpx


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
        for c in curies:
            types.extend(response.json()[c]['type'])
        return types
