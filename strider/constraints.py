"""Constraint handling."""
import operator
import re
from reasoner_pydantic import KnowledgeGraph, HashableSequence

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
            for attribute in kel.attributes or []
            if attribute.attribute_type_id == constraint.id
        )
    except StopIteration:
        return False
    # constraint.negated == constraint.not, but "not" is reserved in Python. Mapped to "negated" in reasoner-pydantic
    return constraint.negated != operator_map[constraint.operator](
        attribute.value,
        constraint.value,
    )


def result_satisfies_constraints(result: dict, kgraph: dict, qgraph: dict) -> bool:
    """Determine whether result satisfies qgraph constraints."""
    for qnode_id, node_bindings in result.node_bindings.items():
        for node_binding in node_bindings:
            knode = kgraph.nodes[node_binding.id]
            for constraint in qgraph.nodes[qnode_id].constraints:
                if not satisfies_attribute_constraint(knode, constraint):
                    return False
    for analysis in result.analyses:
        for qedge_id, edge_bindings in analysis.edge_bindings.items():
            for edge_binding in edge_bindings:
                kedge = kgraph.edges[edge_binding.id]
                for constraint in qgraph.edges[qedge_id].attribute_constraints:
                    if not satisfies_attribute_constraint(kedge, constraint):
                        return False

    return True


def enforce_constraints(message: dict) -> dict:
    """Enforce qgraph constraints from qgraph on results/kgraph."""
    # check if any constraints exist before filtering all results and kgraph
    node_constraints = [
        node.constraints
        for node in message.query_graph.nodes.values()
        if node.constraints and len(node.constraints)
    ]
    edge_constraints = [
        edge.attribute_contraints
        for edge in message.query_graph.edges.values()
        if edge.attribute_constraints and len(edge.attribute_constraints)
    ]
    if node_constraints or edge_constraints:
        message.results = HashableSequence(
            __root__=[
                result
                for result in message.results
                if result_satisfies_constraints(
                    result,
                    message.knowledge_graph,
                    message.query_graph,
                )
            ]
        )
        message.knowledge_graph = KnowledgeGraph.parse_obj(
            {
                "nodes": {
                    binding.id: message.knowledge_graph.nodes[binding.id]
                    for result in message.results
                    for _, bindings in result.node_bindings.items()
                    for binding in bindings
                },
                "edges": {
                    binding.id: message.knowledge_graph.edges[binding.id]
                    for result in message.results
                    for analysis in result.analyses
                    for _, bindings in analysis.edge_bindings.items()
                    for binding in bindings
                },
            }
        )
    return message
