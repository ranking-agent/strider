"""Throttle Utility Functions."""

from collections import defaultdict
from reasoner_pydantic import Message, QueryGraph, AuxiliaryGraphs
from reasoner_pydantic.kgraph import KnowledgeGraph
from reasoner_pydantic.utils import HashableSequence


class BatchingError(Exception):
    """Error batching TRAPI requests."""


class UnableToMerge(BaseException):
    """Unable to merge given query graphs"""


def get_curies(qgraph: QueryGraph) -> dict[str, list[str]]:
    """
    Pull curies from query graph and
    return them as a mapping of node_id -> curie_list
    """
    return {
        node_id: curies
        for node_id, node in qgraph.nodes.items()
        if (curies := node.ids or None) is not None
    }


def get_max_num_curies(requests: list) -> int:
    """
    Given a collection of requests, find the maximum curie length of all query graph nodes
    """
    total_curies = defaultdict(int)
    for request_payload in requests:
        num_curies = get_curies(request_payload.message.query_graph)
        for qnode_id, curies in num_curies.items():
            total_curies[qnode_id] += len(curies)
    # get the max value in dict of curies
    return max(total_curies.values())


def remove_curies(qgraph: QueryGraph) -> dict[str, list[str]]:
    """
    Remove curies from query graph.
    """
    qgraph = qgraph.copy(deep=True)
    for node in qgraph.nodes.values():
        node.ids = None
    return qgraph


def result_contains_node_bindings(result, bindings: dict[str, list[str]]):
    """Check that the result object has all bindings provided (qg_id->kg_id).

    KPs that are returning a (proper) subclass of an entity allowed by the qnode
    may use the optional `qnode_id` field to indicate the associated superclass.
    """
    for qg_id, kg_ids in bindings.items():
        if not any(nb.id in kg_ids for nb in result.node_bindings[qg_id]):
            return False
    return True


def filter_by_curie_mapping(
    message: Message,
    curie_mapping: dict[str, list[str]],
    kp_id: str = "KP",
):
    """
    Filter a message to ensure that all results
    contain the bindings specified in the curie_mapping
    """

    filtered_msg = Message()

    # Only keep results where there is a node binding
    # that connects to our given kgraph_node_id
    filtered_msg.results = HashableSequence(
        __root__=[
            result.copy()
            for result in message.results
            if result_contains_node_bindings(result, curie_mapping)
        ]
    )

    # Construct result-specific knowledge graph
    filtered_msg.knowledge_graph = KnowledgeGraph(
        nodes={
            binding.id: message.knowledge_graph.nodes[binding.id]
            for result in filtered_msg.results
            for _, bindings in result.node_bindings.items()
            for binding in bindings
        },
        edges={
            binding.id: message.knowledge_graph.edges[binding.id]
            for result in filtered_msg.results
            for analysis in result.analyses
            for _, bindings in analysis.edge_bindings.items()
            for binding in bindings
        },
    )

    # Construct result-specific auxiliary graphs
    filtered_aux_graphs = [
        aux_graph_id
        for result in filtered_msg.results or []
        for analysis in result.analyses or []
        for _, bindings in analysis.edge_bindings.items()
        for binding in bindings
        for attribute in message.knowledge_graph.edges[binding.id].attributes or []
        if attribute.attribute_type_id == "biolink:support_graphs"
        for aux_graph_id in attribute.value
    ]
    filtered_aux_graphs.extend(
        [
            aux_graph_id
            for result in filtered_msg.results or []
            for analysis in result.analyses or []
            for aux_graph_id in analysis.support_graphs or []
        ]
    )
    filtered_msg.auxiliary_graphs = AuxiliaryGraphs.parse_obj(
        {
            aux_graph_id: message.auxiliary_graphs[aux_graph_id]
            for aux_graph_id in filtered_aux_graphs
        }
    )

    return filtered_msg


def get_keys_with_value(dct: dict, value):
    """Return keys where the value matches the given"""
    return [k for k, v in dct.items() if v == value]


def log_request(r):
    """Serialize a httpx.Request object into a dict for logging"""
    return {
        "method": r.method,
        "url": str(r.url),
        "headers": dict(r.headers),
        "data": r.read().decode(),
    }


def log_response(r):
    """Serialize a httpx.Response object into a dict for logging"""
    return {
        "status_code": r.status_code,
        "headers": dict(r.headers),
        "data": r.text,
    }
