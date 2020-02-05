"""Partial answer compilation."""
import copy
import hashlib
import json
import time

from strider.graph import create_node


class InvalidSubgraphError(Exception):
    """Attempted to create an invalid subgraph."""


class Partial():
    """Partial answer subgraph."""

    def __init__(self, **kwargs):
        """Initialize."""
        self.nodes = kwargs.get('nodes', dict())
        self.edges = kwargs.get('edges', dict())

    @property
    def edge_qids(self):
        """Get set of edge qids."""
        return list(self.edges)

    def __add__(self, edge):
        """Add an edge (and associated nodes) to answer."""
        # check that edge can be traversed
        if edge.qid in self.edge_qids:
            raise InvalidSubgraphError()

        new = self.copy()
        for node in edge.nodes:
            if node.qid in new.nodes and new.nodes[node.qid].kid != node.kid:
                raise InvalidSubgraphError()

            new.nodes[node.qid] = node
        new.edges[edge.qid] = edge

        return new

    def combine(self, *others):
        """Combine partial subgraphs together, if possible."""
        out = self
        for other in others:
            for edge in other.edges.values():
                out += edge
        return out

    def copy(self):
        """Return a deep copy of this object."""
        return Partial(
            nodes=copy.deepcopy(self.nodes),
            edges=copy.deepcopy(self.edges)
        )

    def __str__(self):
        """Generate string representation."""
        return f'<Partial nodes={self.nodes}, edges={self.edges}>'

    def __repr__(self):
        """Generate string representation."""
        return str(self)

    def to_dict(self):
        """Generate dictionary representation."""
        return {
            'nodes': {node.qid: node.to_dict() for node in self.nodes.values()},
            'edges': {edge.qid: edge.to_dict() for edge in self.edges.values()},
        }

    def to_concise(self):
        """Generate concise, hashable representation."""
        return (
            tuple(sorted([node.kid for node in self.nodes.values()])),
            tuple(sorted([edge.kid for edge in self.edges.values()])),
        )

    @property
    def hash(self):
        """Generate hash string."""
        # return hash(hashlib.md5(json.dumps(self.to_dict()).encode('utf-8')).hexdigest())
        return hash(self.to_concise())

    def __hash__(self):
        """Generate hash."""
        return self.hash

    def __eq__(self, other):
        """Determine equality."""
        return self.hash == other.hash


def get_paths(query_id=None, kid=None, qid=None, prefix=None, level=0):
    """Get partial answer paths."""
    if isinstance(query_id, str):
        node = create_node(query_id=query_id, kid=kid, qid=qid)
    else:
        node = query_id
    if prefix is None:
        prefix = Partial(nodes={qid: node})
    partials = {prefix}
    for edge in node.edges:
        start_time = time.time()
        if edge.qid in prefix.edges:
            continue
        target_node = edge.other(node)
        new_partials = set()
        for partial in partials:
            try:
                new_prefix = partial + edge
            except InvalidSubgraphError:
                continue
            new_partials |= get_paths(target_node, prefix=new_prefix, level=level + 1)
        partials |= new_partials
        # print('  |' * level + f'  |elapsed: {time.time() - start_time} seconds')
    return partials


if __name__ == "__main__":
    start_time = time.time()
    paths = get_paths(query_id='alpha', kid='b1', qid='B')
    print(f'elapsed: {time.time() - start_time} seconds')
    # sorted_reps = sorted(path.to_concise() for path in paths)
    # for rep in sorted_reps:
    #     print(rep)
