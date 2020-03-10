"""Graph scoring."""
from collections import defaultdict
from itertools import combinations
import logging
import os

import httpx
import numpy as np

LOGGER = logging.getLogger(__name__)
OMNICORP_URL = os.getenv('OMNICORP_URL', 'http://localhost:3210')


async def get_support(node1, node2, synonyms):
    """Get number of publications shared by nodes."""
    # prefer HGNC
    if node1.startswith('NCBIGene'):
        node1 = next(
            (curie for curie in synonyms[node1] if curie.startswith('HGNC')),
            node1
        )
    if node2.startswith('NCBIGene'):
        node2 = next(
            (curie for curie in synonyms[node2] if curie.startswith('HGNC')),
            node2
        )

    query = f'{OMNICORP_URL}/shared'
    async with httpx.AsyncClient() as client:
        response = await client.get(
            query,
            params={'curie': [
                node1,
                node2,
            ]},
        )
    if response.status_code >= 300:
        raise RuntimeError(f'The following OmniCorp query returned a bad response:\n{query}')
    return response.json()


async def score_graph(graph):
    """Score graph.
    
    https://en.wikipedia.org/wiki/Resistance_distance#General_sum_rule
    """
    if not graph['edges']:
        return 0

    kid_to_qids = defaultdict(list)
    node_synonyms = dict()
    for node in graph['nodes'].values():
        kid_to_qids[node['kid']].append(node['qid'])
        node_synonyms[node['kid']] = node.get('equivalent_identifiers', [])
    node_ids = sorted([node['kid'] for node in graph['nodes'].values()])
    num_nodes = len(node_ids)
    laplacian = np.zeros((num_nodes, num_nodes))
    index = {node_id: node_ids.index(node_id) for node_id in node_ids}
    for curie1, curie2 in combinations(node_ids, 2):
        admittance = 1 + await get_support(curie1, curie2, node_synonyms)
        i, j = index[curie1], index[curie2]
        laplacian[i, j] += -admittance
        laplacian[j, i] += -admittance
        laplacian[i, i] += admittance
        laplacian[j, j] += admittance
    eigvals = np.linalg.eigvals(laplacian).tolist()
    eigvals = np.array(sorted(eigvals)[1:])
    kirchhoff_index = (num_nodes * np.sum(1 / eigvals)).tolist()
    return 1 / kirchhoff_index
