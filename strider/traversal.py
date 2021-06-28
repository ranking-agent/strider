"""Traversal."""

class NoAnswersError(Exception):
    """No answers can be found."""


def traverse_edge(qgraph: dict, qedge_id):
    """Traverse an edge and return the resulting qgraph.

    Remove the edge from the qgraph and pin both of its endpoints.
    """
    endpoints = (
        qgraph["edges"][qedge_id]["subject"],
        qgraph["edges"][qedge_id]["object"],
    )
    return {
        "nodes": {
            qnode_id: (qnode | {"ids": True} if qnode_id in endpoints else qnode)
            for qnode_id, qnode in qgraph["nodes"].items()
        },
        "edges": {
            _qedge_id: qedge
            for _qedge_id, qedge in qgraph["edges"].items()
            if _qedge_id != qedge_id
        }
    }


def get_traversals(qgraph: dict):
    """Get all possible traversals.

    A traversal is a sequence of edges.
    The direction that each edge is traversed is irrelevant.
    """
    pinned_nodes = [
        qnode_id
        for qnode_id, qnode in qgraph["nodes"].items()
        if qnode.get("ids")
    ]
    if len(pinned_nodes) == len(qgraph["nodes"]) and not qgraph["edges"]:
        return [[]]
    traversable_edges = [
        edge_id
        for edge_id, edge in qgraph["edges"].items()
        if edge["subject"] in pinned_nodes or edge["object"] in pinned_nodes
    ]
    if not traversable_edges:
        unreachable_nodes = [qid for qid in qgraph["nodes"] if qid not in pinned_nodes]
        unreachable_edges = list(qgraph["edges"])
        raise NoAnswersError("Query planning cannot reach nodes {} and edges {}".format(
            unreachable_nodes,
            unreachable_edges,
        ))
    traversals = []
    for attached_edge in traversable_edges:
        traversals.extend([
            [attached_edge] + traversal
            for traversal in get_traversals(traverse_edge(qgraph, attached_edge))
        ])
    return traversals
