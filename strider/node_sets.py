"""Node sets."""
from collections import defaultdict


def collapse_sets(message: dict) -> None:
    """Collase results according to is_set qnode notations."""
    unique_qnodes = {
        qnode_id
        for qnode_id, qnode in message["query_graph"]["nodes"].items()
        if not qnode.get("is_set", False)
    }
    if len(unique_qnodes) == len(message["query_graph"]["nodes"]):
        return
    unique_qedges = {
        qedge_id
        for qedge_id, qedge in message["query_graph"]["edges"].items()
        if (qedge["subject"] in unique_qnodes and qedge["object"] in unique_qnodes)
    }
    result_buckets = defaultdict(
        lambda: {
            "node_bindings": defaultdict(set),
            "analyses": [{"edge_bindings": defaultdict(set)}],
        }
    )
    for result in message["results"]:
        bucket_key = tuple(
            [
                binding["id"]
                for qnode_id in unique_qnodes
                for binding in result["node_bindings"][qnode_id]
            ]
            + [
                binding["id"]
                for qedge_id in unique_qedges
                for analysis in result.get("analyses", [])
                for binding in analysis["edge_bindings"][qedge_id]
            ]
        )
        for qnode_id in message["query_graph"]["nodes"]:
            result_buckets[bucket_key]["node_bindings"][qnode_id] |= {
                binding["id"] for binding in result["node_bindings"][qnode_id]
            }
        for qedge_id in message["query_graph"]["edges"]:
            result_buckets[bucket_key]["edge_bindings"][qedge_id] |= {
                binding["id"] for binding in result["edge_bindings"][qedge_id]
            }
    for result in result_buckets.values():
        result["node_bindings"] = {
            qnode_id: [{"id": binding} for binding in bindings]
            for qnode_id, bindings in result["node_bindings"].items()
        }
        for analysis in result.get("analyses", []):
            analysis["edge_bindings"] = {
                qedge_id: [{"id": binding} for binding in bindings]
                for qedge_id, bindings in analysis["edge_bindings"].items()
            }
    message["results"] = list(result_buckets.values())
