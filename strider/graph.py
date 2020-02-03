"""Node/edge classes."""
from abc import ABC
from functools import cached_property

from strider.neo4j import HttpInterface

HOSTNAME = 'localhost'

_nodes = dict()
_edges = dict()


def create_node(query_id=None, kid=None, qid=None, *args, **kwargs):
    """Create and return node."""
    global _nodes
    # if id in _nodes:
    #     return _nodes[id]
    node = Node(query_id=query_id, kid=kid, qid=qid, *args, **kwargs)
    _nodes[kid] = node
    return node


def create_edge(query_id=None, kid=None, qid=None, *args, **kwargs):
    """Create and return edge."""
    global _edges
    # if id in _edges:
    #     return _edges[id]
    edge = Edge(query_id=query_id, kid=kid, qid=qid, *args, **kwargs)
    _edges[kid] = edge
    return edge


class Base(ABC):
    """Base class for Node and Edge."""

    def __init__(self, query_id=None, kid=None, qid=None, *args, **kwargs):
        """Initialize."""
        self.kid = kid
        self.qid = qid
        self.query_id = query_id
        self.neo4j_interface = HttpInterface(
            url=f'http://{HOSTNAME}:7474',
            # credentials={
            #     'username': 'neo4j',
            #     'password': 'ncatsgamma',
            # },
        )
        self.properties = kwargs

    def __str__(self):
        """Generate string representation."""
        return f'<{type(self).__name__} kid="{self.kid}", qid="{self.qid}">'

    def __repr__(self):
        """Generate string representation."""
        return f'{self.kid}:{self.qid}'

    def to_dict(self):
        """Generate dict representation."""
        return {
            'kid': self.kid,
            'qid': self.qid,
            **self.properties,
        }


class Node(Base):
    """Node in graph."""

    @cached_property
    def edges(self):
        """Get edges attached to node."""
        statement = f'MATCH (:{self.query_id} {{kid: "{self.kid}", qid:"{self.qid}"}})-[e:{self.query_id}]-() RETURN DISTINCT e{{.*}}'
        results = self.neo4j_interface.run(statement)
        edges = [create_edge(query_id=self.query_id, **row['e']) for row in results]
        return edges


class Edge(Base):
    """Edge in graph."""

    @cached_property
    def nodes(self):
        """Get nodes attached to edge."""
        statement = f'MATCH (n:{self.query_id})-[e {{kid: "{self.kid}", qid: "{self.qid}"}}]-() RETURN n{{.*}}'
        results = self.neo4j_interface.run(statement)
        return [create_node(self.query_id, **row['n']) for row in results]

    def other(self, terminus):
        """Get the node on the other end of this edge from terminus."""
        return next(node for node in self.nodes if node.qid != terminus.qid)
