"""Constraint handling."""
import operator
import re

operator_map = {
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
    "matches": lambda s, pattern: re.fullmatch(pattern, s) is not None,
}


def satisfies_attribute_constraint(kel: dict, constraint: dict) -> bool:
    """Determine whether knowledge graph element satisfies attribute constraint.

    If the constrained attribute is missing, returns False.
    """
    try:
        attribute = next(
            attribute
            for attribute in kel.get("attributes", None) or []
            if attribute["attribute_type_id"] == constraint["id"]
        )
    except StopIteration:
        return False
    return constraint.get("not", False) != operator_map[constraint["operator"]](
        attribute["value"],
        constraint["value"],
    )


def result_satisfies_constraints(result: dict, kgraph: dict, qgraph: dict) -> bool:
    """Determine whether result satisfies qgraph constraints."""
    for qnode_id, node_bindings in result["node_bindings"].items():
        for node_binding in node_bindings:
            knode = kgraph["nodes"][node_binding["id"]]
            for constraint in qgraph["nodes"][qnode_id].get("constraints", []):
                if not satisfies_attribute_constraint(knode, constraint):
                    return False
    for qedge_id, edge_bindings in result["edge_bindings"].items():
        for edge_binding in edge_bindings:
            kedge = kgraph["edges"][edge_binding["id"]]
            for constraint in qgraph["edges"][qedge_id].get(
                "attribute_constraints", []
            ):
                if not satisfies_attribute_constraint(kedge, constraint):
                    return False

    return True


def enforce_constraints(message: dict) -> dict:
    """Enforce qgraph constraints from qgraph on results/kgraph."""
    message["results"] = [
        result
        for result in message["results"]
        if result_satisfies_constraints(
            result,
            message["knowledge_graph"],
            message["query_graph"],
        )
    ]
    message["knowledge_graph"] = {
        "nodes": {
            binding["id"]: message["knowledge_graph"]["nodes"][binding["id"]]
            for result in message["results"]
            for _, bindings in result["node_bindings"].items()
            for binding in bindings
        },
        "edges": {
            binding["id"]: message["knowledge_graph"]["edges"][binding["id"]]
            for result in message["results"]
            for _, bindings in result["edge_bindings"].items()
            for binding in bindings
        },
    }
    return message
