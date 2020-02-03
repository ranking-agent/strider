"""Graph scoring."""
import numpy as np


def score_graph(graph):
    """Score graph."""
    if not graph['edges']:
        return None
    return np.mean([edge['weight'] for edge in graph['edges'].values()]).tolist()
