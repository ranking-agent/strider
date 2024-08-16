"""Graph - a dict with extra methods."""


def connected_edges(message, node_id):
    """Find edges connected to node."""
    outgoing = []
    incoming = []
    for edge_id, edge in message.query_graph.edges.items():
        if node_id == edge.subject:
            outgoing.append(edge_id)
        if node_id == edge.object:
            incoming.append(edge_id)
    return outgoing, incoming


def remove_orphaned(message):
    """Remove nodes with degree 0."""
    message.query_graph.nodes = {
        node_id: node
        for node_id, node in message.query_graph.nodes.items()
        if any(connected_edges(message, node_id))
    }
