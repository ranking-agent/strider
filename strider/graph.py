"""Graph - a dict with extra methods."""

import json


class Graph(dict):
    """Graph."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)

    def __hash__(self):
        """Compute hash."""
        return hash(json.dumps(self, sort_keys=True))

    def connected_edges(self, node_id):
        """Find edges connected to node."""
        outgoing = []
        incoming = []
        for edge_id, edge in self["edges"].items():
            if node_id == edge["subject"]:
                outgoing.append(edge_id)
            if node_id == edge["object"]:
                incoming.append(edge_id)
        return outgoing, incoming

    def remove_orphaned(self):
        """Remove nodes with degree 0."""
        self["nodes"] = {
            node_id: node
            for node_id, node in self["nodes"].items()
            if any(self.connected_edges(node_id))
        }
