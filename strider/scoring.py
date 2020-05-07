"""Graph scoring."""
import asyncio
from itertools import combinations
import json
import logging
import os

import httpx
import numpy as np

LOGGER = logging.getLogger(__name__)
OMNICORP_URL = os.getenv('OMNICORP_URL', 'http://localhost:3210')
OMNICORP_RETRIES = 5
NUM_PUBS = 27840000


async def get_support(node1, node2, synonyms):
    """Get number of publications shared by nodes."""
    # prefer HGNC
    if node1.startswith('NCBIGene'):
        node1 = next(
            (curie for curie in synonyms[node1] if curie.startswith('HGNC')),
            node1
        )
    elif node1.startswith('MESH'):
        node1 = next(
            (curie for curie in synonyms[node1] if curie.startswith('CHEBI')),
            node1
        )
    if node2.startswith('NCBIGene'):
        node2 = next(
            (curie for curie in synonyms[node2] if curie.startswith('HGNC')),
            node2
        )
    elif node2.startswith('MESH'):
        node2 = next(
            (curie for curie in synonyms[node2] if curie.startswith('CHEBI')),
            node2
        )

    edge_pubs, source_pubs, target_pubs = await asyncio.gather(
        count_pubs(node1, node2),
        count_pubs(node1),
        count_pubs(node2),
    )
    cov = (edge_pubs / NUM_PUBS) \
        - (source_pubs / NUM_PUBS) * (target_pubs / NUM_PUBS)
    cov = max((cov, 0.0))
    effective_pubs = cov * NUM_PUBS

    return effective_pubs

async def count_pubs(*curies):
    """Count pubs shared by curies."""
    url = f'{OMNICORP_URL}/shared?'
    params = {'curie': curies}
    async with httpx.AsyncClient() as client:
        for _ in range(OMNICORP_RETRIES):
            try:
                response = await client.get(
                    url,
                    params=params,
                )
                break
            except httpx.NetworkError as err:
                LOGGER.warning(
                    'Omnicorp network error: curies: %s, %s. Trying again...',
                    json.dumps(curies),
                    str(err)
                )
    if response.status_code >= 300:
        raise RuntimeError(
            'The following OmniCorp query returned a bad response:\n'
            f'{url}\n{params}\n{response.text}'
        )
    return response.json()


async def add_edge(curies, index, node_synonyms, laplacian, degree):
    """Add weighted edge to laplacian."""
    curie1, curie2 = curies
    i, j = index[curie1], index[curie2]
    admittance = 1 + await get_support(curie1, curie2, node_synonyms)
    admittance /= (degree[curie1] * degree[curie2])

    laplacian[i, j] += -admittance
    laplacian[j, i] += -admittance
    laplacian[i, i] += admittance
    laplacian[j, j] += admittance



async def score_graph(graph, qgraph, support=True):
    """Score graph.

    https://en.wikipedia.org/wiki/Resistance_distance#General_sum_rule
    """
    if not graph['edges']:
        return 0

    node_synonyms = {
        node['kid']: node.get('equivalent_identifiers', [])
        for node in graph['nodes'].values()
    }
    # Namdi: all of the node ids in the graph
    node_ids = sorted([node['kid'] for node in graph['edges'].values()])

    # Namdi edit
    degree = {node['kid']: node['degree'] for node in graph['nodes'].values()}

    LOGGER.debug(degree)
    
    # Namdi: the number of nodes
    num_nodes = len(node_ids)
    LOGGER.debug('num_nodes: %d', num_nodes)
    laplacian = np.zeros((num_nodes, num_nodes))
    
    # Namdi: for a given node_id, return the index of where it was found in the node id list
    index = {node_id: node_ids.index(node_id) for node_id in node_ids}

    awaitables = []

    # Namdi: calculate novlety taking the results = graph
    # Then update the edge wieghts (and edit query() )
    # results = weight_novelty.query(message, *, exclude_sets=False) 
    # change to redges_by_id, results = wieght_novlety.query_new(graph, qgraph)
    # note that, I will be sending only 1 results graph so within query_new
    # kdx = 0 only. I can ignore this dimension. 
    # Also, redges_by_id[ (kdx, redge['id']) ] becomes redges_by_id[ (0, redge['id']) ]
    # eb = redges_by_id[redge_id]['eb']
    # eb['weight'] -> this is the novelty weights
    # results -> list of result -> rgraph
    # redges_by_id: lookup table for (result index, redge id) -> redges
    # NEED to find a way to find the result edge corresponding to curie1, curie2 from graph. 
    # I.e, How do I map curie1, curie2 to and redge / edgebinding
    if support:
        for curie1, curie2 in combinations(node_ids, 2):
            awaitables.append(add_edge(
                (curie1, curie2),
                index,
                node_synonyms,
                laplacian,
                degree,
            ))
            
    else:
        for qedge in qgraph['edges'].values():
            try:
                source = graph['nodes'][qedge['source_id']]
                target = graph['nodes'][qedge['target_id']]
            except KeyError:
                # this is a partial answer and does not include this edge
                continue
            awaitables.append(add_edge(
                (source['kid'], target['kid']),
                index,
                node_synonyms,
                laplacian,
                degree,
            ))

    await asyncio.gather(*awaitables)
    # get array of eigenvalues, except for lowest
    eigvals = np.sort(np.linalg.eigvals(laplacian))[1:]
    # return 1 / kirchhoff_index
    return 1 / (num_nodes * np.sum(1 / eigvals)).tolist()
