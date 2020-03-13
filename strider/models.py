"""ReasonerStdAPI models."""
from typing import Union, List

from pydantic import BaseModel, AnyUrl


class BiolinkEntity(BaseModel):
    """Biolink entity."""

    __root__: str


class BiolinkRelation(BaseModel):
    """Biolink relation."""

    __root__: str


class QNode(BaseModel):
    """Query node."""

    id: str
    curie: Union[str, List[str]] = None
    type: Union[str, List[str]] = 'named_thing'


class QEdge(BaseModel):
    """Query edge."""

    id: str
    type: Union[str, List[str]] = 'related_to'
    source_id: str
    target_id: str


class Node(BaseModel):
    """Knowledge graph node."""

    id: str
    name: str = None
    type: Union[str, List[str]] = None
    equivalent_identifiers: List[str] = None


class Edge(BaseModel):
    """Knowledge graph edge."""

    id: str
    type: Union[str, List[str]] = None
    source_id: str
    target_id: str
    predicate_id: str = None
    relation_label: str = None
    edge_source: str = None
    ctime: str = None
    source_database: str = None
    relation: str = None
    publications: List[str] = None


class QueryGraph(BaseModel):
    """Query graph."""

    nodes: List[QNode]
    edges: List[QEdge]


class Credentials(BaseModel):
    """Credentials."""

    username: str
    password: str


class RemoteKnowledgeGraph(BaseModel):
    """Remote knowledge graph."""

    url: AnyUrl
    credentials: Credentials = None
    protocol: str = None


class KnowledgeGraph(BaseModel):
    """Knowledge graph."""

    nodes: List[Node]
    edges: List[Edge]


class EdgeBinding(BaseModel):
    """Edge binding."""

    qg_id: str
    kg_id: Union[str, List[str]]


class NodeBinding(BaseModel):
    """Edge binding."""

    qg_id: str
    kg_id: Union[str, List[str]]


class Result(BaseModel):
    """Result."""

    node_bindings: List[NodeBinding]
    edge_bindings: List[EdgeBinding]
    score: float
    timestamp: float


class Message(BaseModel):
    """Message."""

    query_graph: QueryGraph = None
    knowledge_graph: Union[KnowledgeGraph, RemoteKnowledgeGraph] = None
    results: List[Result] = None


class Query(BaseModel):
    """Query."""

    message: Message
