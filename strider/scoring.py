"""Graph scoring."""
import logging
import urllib

import httpx
import numpy as np

LOGGER = logging.getLogger(__name__)


async def get_support(node1, node2):
    """Get number of publications shared by nodes."""
    query = f'http://robokop.renci.org:3210/shared?curie={urllib.parse.quote(node1)}&curie={urllib.parse.quote(node2)}'
    async with httpx.AsyncClient() as client:
        response = await client.get(query)
    if response.status_code >= 300:
        raise RuntimeError(f'The following OmniCorp query returned a bad response:\n{query}')
    return response.json()


async def score_graph(graph):
    """Score graph."""
    if not graph['edges']:
        return 0
    return 1 / np.sum([1 / edge['weight'] for edge in graph['edges'].values()]).tolist()
