"""Graph scoring."""
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


def score_graph(graph):
    """Score graph."""
    if not graph['edges']:
        return 0
    return 1 / np.sum([1 / edge['weight'] for edge in graph['edges'].values()]).tolist()
