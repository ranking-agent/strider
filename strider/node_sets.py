"""Node sets."""

from collections import defaultdict
from datetime import datetime
from reasoner_pydantic import Message


def collapse_sets(query: dict, logger) -> None:
    """Collase results according to set_interpretation qnode notations."""
    # just deserializing the query_graph is very fast
    qgraph = query.message.query_graph.dict()
    unique_qnodes = {
        qnode_id
        for qnode_id, qnode in qgraph["nodes"].items()
        if (qnode.get("set_interpretation", None) or "BATCH") == "BATCH"
    }
    if len(unique_qnodes) == len(query.message.query_graph.nodes):
        # no set nodes
        return
    logger.info("Collapsing sets. This might take a while...")
    message = query.message.dict()
    unique_qedges = {
        qedge_id
        for qedge_id, qedge in message["query_graph"]["edges"].items()
        if (qedge["subject"] in unique_qnodes and qedge["object"] in unique_qnodes)
    }
    result_buckets = defaultdict(
        lambda: {
            "node_bindings": defaultdict(set),
            "analyses": [],
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
            for index, analysis in enumerate(result.get("analyses", [])):
                result_buckets[bucket_key]["analyses"].append(
                    {
                        "resource_id": analysis["resource_id"],
                        "edge_bindings": defaultdict(set),
                    }
                )
                result_buckets[bucket_key]["analyses"][index]["edge_bindings"][
                    qedge_id
                ] |= {binding["id"] for binding in analysis["edge_bindings"][qedge_id]}
    for result in result_buckets.values():
        result["node_bindings"] = {
            qnode_id: [{"id": binding, "attributes": []} for binding in bindings]
            for qnode_id, bindings in result["node_bindings"].items()
        }
        for analysis in result.get("analyses", []):
            analysis["edge_bindings"] = {
                qedge_id: [{"id": binding, "attributes": []} for binding in bindings]
                for qedge_id, bindings in analysis["edge_bindings"].items()
            }
    message["results"] = list(result_buckets.values())

    query.message = Message.parse_obj(message)

    logger.info("Finished collapsing all the sets")
