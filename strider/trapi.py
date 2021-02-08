"""TRAPI utilities."""
from collections import defaultdict

from reasoner_pydantic import Message, Result, QNode, Node, Edge
from reasoner_pydantic.qgraph import QueryGraph

from .util import deduplicate_by


def merge_messages(messages: list[Message]) -> Message:
    """Merge messages."""
    knodes = dict()
    kedges = dict()
    results = []
    for message in messages:
        knodes |= message["knowledge_graph"]["nodes"]
        kedges |= message["knowledge_graph"]["edges"]
        for result in message["results"]:
            # Use pydantic model so that we can deduplicate
            result_pydantic = Result.parse_obj(result)
            if result_pydantic not in results:
                results.append(result_pydantic)
    return {
        "query_graph": messages[0]["query_graph"],
        "knowledge_graph": {
            "nodes": knodes,
            "edges": kedges
        },
        "results": [r.dict() for r in results]
    }


def merge_results(results: list[Result]) -> Result:
    """Merge results."""
    node_bindings = defaultdict(list)
    edge_bindings = defaultdict(list)
    for result in results:
        for key, value in result["node_bindings"].items():
            node_bindings[key].extend(value)
        for key, value in result["edge_bindings"].items():
            edge_bindings[key].extend(value)
    # unique-ify by id
    node_bindings = {
        qnode_id: deduplicate_by(knodes, lambda x: x["id"])
        for qnode_id, knodes in node_bindings.items()
    }
    edge_bindings = {
        qedge_id: deduplicate_by(kedges, lambda x: x["id"])
        for qedge_id, kedges in edge_bindings.items()
    }
    return {
        "node_bindings": node_bindings,
        "edge_bindings": edge_bindings,
    }


def get_curies(message: Message) -> list[str]:
    """Get all node curies used in message.

    Do not examine kedge source and target ids. There ought to be corresponding
    knodes.
    """
    curies = set()
    if 'query_graph' in message:
        for qnode in message['query_graph']['nodes'].values():
            if qnode_id := qnode.get("id", False):
                curies |= set(qnode_id)
    if 'knowledge_graph' in message:
        curies |= set(message['knowledge_graph']['nodes'])
    return curies


def apply_curie_map(message: Message, curie_map: dict[str, str]) -> Message:
    """Translate all pinned qnodes to preferred prefix."""
    new_message = dict()
    new_message["query_graph"] = fix_qgraph(message["query_graph"], curie_map)
    if "knowledge_graph" in message:
        kgraph = message["knowledge_graph"]
        new_message['knowledge_graph'] = {
            'nodes': {
                curie_map.get(knode_id, knode_id): knode
                for knode_id, knode in kgraph['nodes'].items()
            },
            'edges': {
                kedge_id: fix_kedge(kedge, curie_map)
                for kedge_id, kedge in kgraph['edges'].items()
            },
        }
    if "results" in message:
        results = message["results"]
        new_message['results'] = [
            fix_result(result, curie_map)
            for result in results
        ]
    return new_message


def fix_qgraph(qgraph: QueryGraph, curie_map: dict[str, str]) -> QueryGraph:
    """Replace curies with preferred, if possible."""
    return {
        'nodes': {
            qnode_id: fix_qnode(qnode, curie_map)
            for qnode_id, qnode in qgraph['nodes'].items()
        },
        "edges": qgraph["edges"],
    }


def fix_qnode(qnode: QNode, curie_map: dict[str, str]) -> QNode:
    """Replace curie with preferred, if possible."""
    if not qnode.get("id", None):
        return qnode

    if isinstance(qnode["id"], list):
        fixed_ids = [curie_map.get(curie, curie) for curie in qnode['id']]
    else:
        fixed_ids = curie_map.get(qnode["id"], qnode["id"])
    qnode = {
        **qnode,
        "id": fixed_ids,
    }
    return qnode


def fix_knode(knode: Node, curie_map: dict[str, str]) -> Node:
    """Replace curie with preferred, if possible."""
    knode = {
        **knode,
        "id": curie_map.get(knode["id"], knode["id"])
    }
    return knode


def fix_kedge(kedge: Edge, curie_map: dict[str, str]) -> Edge:
    """Replace curie with preferred, if possible."""
    kedge["subject"] = curie_map.get(kedge["subject"], kedge["subject"])
    kedge["object"] = curie_map.get(kedge["object"], kedge["object"])
    return kedge


def fix_result(result: Result, curie_map: dict[str, str]) -> Result:
    """Replace curie with preferred, if possible."""
    result['node_bindings'] = {
        qnode_id: [
            {
                **node_binding,
                "id": curie_map.get(
                    node_binding["id"],
                    node_binding["id"]
                ),
            }
            for node_binding in node_bindings
        ]
        for qnode_id, node_bindings in result['node_bindings'].items()
    }
    return result
