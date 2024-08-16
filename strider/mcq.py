"""Utility functions for MCQ queries."""

from typing import List
from reasoner_pydantic import (
    QNode,
    Result,
    Edge,
    KnowledgeGraph,
    AuxiliaryGraphs,
)


def is_mcq_node(qnode: QNode) -> bool:
    """Determin if query graph node is a set for MCQ (MultiCurieQuery)."""
    return qnode.set_interpretation == "MANY"


def get_mcq_edge_ids(
    result: Result, kgraph: KnowledgeGraph, auxgraph: AuxiliaryGraphs
) -> List[Edge]:
    mcq_edge_ids = []
    for analysis in result.analyses:
        for edge_bindings in analysis.edge_bindings.values():
            for edge_binding in edge_bindings:
                kgraph_edge = edge_binding.id
                for attribute in kgraph.edges[kgraph_edge].attributes:
                    if attribute.attribute_type_id == "biolink:support_graphs":
                        for auxgraph_id in attribute.value:
                            for edge_id in auxgraph[auxgraph_id].edges:
                                mcq_edge_ids.append(edge_id)
    return mcq_edge_ids
