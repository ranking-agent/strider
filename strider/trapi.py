"""TRAPI utilities."""
from itertools import product
import json
import logging
import traceback
import typing

import bmt
from reasoner_pydantic import (
    Message,
    QNode,
    Edge,
    Results,
    KnowledgeGraph,
    AuxiliaryGraphs,
)
from reasoner_pydantic.qgraph import QEdge, QueryGraph
from reasoner_pydantic.shared import BiolinkPredicate
from reasoner_pydantic.utils import HashableSequence

from strider.utils import (
    WBMT,
)
from strider.throttle_utils import (
    get_curies,
)
from strider.normalizer import Normalizer
from strider.config import settings

blocklist = []
with open("blocklist.json", "r") as f:
    blocklist = json.load(f)


def apply_curie_map(
    message: Message,
    curie_map: dict[str, str],
    kp_id: str,
    logger: logging.Logger,
):
    """Translate all pinned qnodes to preferred prefix."""
    map_qgraph_curies(message.query_graph, curie_map)
    if message.knowledge_graph is not None:
        kgraph = message.knowledge_graph

        # Update knode IDs
        knode_mapping = {
            knode_id: curie_map.get(knode_id, [knode_id])[0]
            for knode_id in kgraph.nodes.keys()
        }
        for old, new in knode_mapping.items():
            kgraph.nodes[new] = kgraph.nodes.pop(old)

        # Update kedge subject and object
        for kedge in kgraph.edges.values():
            fix_kedge(kedge, curie_map)

    if message.results is not None:
        fix_results(message, curie_map, kp_id, logger)


def map_qgraph_curies(
    qgraph: QueryGraph,
    curie_map: dict[str, str],
    primary: bool = False,
) -> QueryGraph:
    """Replace curies with preferred, if possible."""
    if qgraph is None:
        return None
    for qnode in qgraph.nodes.values():
        map_qnode_curies(qnode, curie_map, primary)


def get_canonical_qgraphs(
    qgraph: QueryGraph,
) -> typing.List[QueryGraph]:
    """Replace predicates with canonical directions."""
    if qgraph is None:
        return []

    qedge_sets = [
        {qedge_id: qedge for qedge_id, qedge in qedges}
        for qedges in product(
            *[
                [
                    (qedge_id, dir_qedge)
                    for dir_qedge in get_canonical_qedge(qedge)
                    if dir_qedge.predicates
                ]
                for qedge_id, qedge in qgraph.edges.items()
            ]
        )
    ]
    return [
        QueryGraph(nodes=qgraph.nodes.copy(), edges=qedges) for qedges in qedge_sets
    ]


def get_canonical_qedge(
    input_qedge: QEdge,
) -> typing.Tuple[QEdge, QEdge]:
    """Use canonical predicate direction."""

    predicates = HashableSequence[BiolinkPredicate]()
    flipped_predicates = HashableSequence[BiolinkPredicate]()
    for predicate in input_qedge.predicates or []:
        slot = WBMT.bmt.get_element(predicate)
        # if predicate not in bmt
        if slot is None:
            predicates.append(predicate)
            continue
        is_canonical = slot.annotations.get("canonical_predicate", False)
        if is_canonical or slot.symmetric or slot.inverse is None:
            # predicate is canonical, use it
            predicates.append(predicate)
        else:
            flipped_predicates.append(bmt.util.format(slot.inverse, case="snake"))

    qedge = input_qedge.copy()
    flipped_qedge = input_qedge.copy()
    flipped_qedge.predicates = None

    qedge.predicates = predicates
    flipped_qedge.subject, flipped_qedge.object = (
        flipped_qedge.object,
        flipped_qedge.subject,
    )

    if len(flipped_predicates) > 0:
        flipped_qedge.predicates = flipped_predicates

    return qedge, flipped_qedge


def map_qnode_curies(
    qnode: QNode,
    curie_map: dict[str, str],
    primary: bool = False,
) -> QNode:
    """Replace curie with preferred, if possible."""

    if not qnode.ids:
        return

    output_curies = []
    for existing_curie in qnode.ids:
        if primary:
            output_curies.append(curie_map.get(existing_curie, [existing_curie])[0])
        else:
            output_curies.extend(curie_map.get(existing_curie, []))
    if len(output_curies) == 0:
        output_curies = qnode.ids

    qnode.ids = output_curies


def fix_kedge(kedge: Edge, curie_map: dict[str, str]):
    """Replace curie with preferred, if possible."""
    kedge.subject = curie_map.get(kedge.subject, [kedge.subject])[0]
    kedge.object = curie_map.get(kedge.object, [kedge.object])[0]


def fix_results(
    message: Message,
    curie_map: dict[str, str],
    kp_id: str,
    logger: logging.Logger,
):
    """Replace curie with preferred, if possible."""
    for result in message.results or []:
        for qnode_id, node_bindings in result.node_bindings.items():
            for node_binding in node_bindings:
                # set node binding id to preferred prefix
                node_binding.id = curie_map.get(node_binding.id, [node_binding.id])[0]
                if node_binding.query_id is None:
                    continue
                # set node binding query id to preferred prefix
                query_id = curie_map.get(
                    node_binding.query_id, [node_binding.query_id]
                )[0]
                if node_binding.id == query_id:
                    # remove query_id if equivalent to id
                    logger.debug(
                        f"Removing query_id {node_binding.query_id} because it is equivalent to {node_binding.id}."
                    )
                    node_binding.query_id = None
                elif query_id not in message.query_graph.nodes[qnode_id].ids:
                    # if query id isn't in qgraph
                    if node_binding.id in message.query_graph.nodes[qnode_id].ids:
                        # if kgraph id in qgraph, then remove query id
                        # this was probably a preferred prefix
                        logger.debug(
                            f"Removing query_id {node_binding.query_id} because kgraph id is in qgraph."
                        )
                        node_binding.query_id = None
                    else:
                        # both kgraph id and query id not in qgraph,
                        logger.error(
                            f"Got back {node_binding.query_id} query_id from {kp_id} and it doesn't match any query ids."
                        )
                        node_binding.query_id = None
                else:
                    node_binding.query_id = query_id


def filter_ancestor_types(categories):
    """Filter out types that are ancestors of other types in the list"""

    def is_ancestor(a, b):
        """Check if one biolink type is an ancestor of the other"""
        ancestors = WBMT.get_ancestors(b, reflexive=False)
        if a in ancestors:
            return True
        return False

    has_descendant = [
        any(
            is_ancestor(categories[idx], a)
            for a in categories[:idx] + categories[idx + 1 :]
        )
        for idx in range(len(categories))
    ]
    return [category for category, drop in zip(categories, has_descendant) if not drop]


def filter_message(
    message: Message,
    curie_map: dict,
    logger: logging.Logger = logging.getLogger(),
    information_content_threshold: int = settings.information_content_threshold,
    last_hop: bool = False
) -> None:
    """Filter all nodes based on information content."""
    pinned_nodes = get_curies(message.query_graph)
    kept_knowledge_graph = KnowledgeGraph.parse_obj({"nodes": {}, "edges": {}})
    kept_results = Results.parse_obj([])
    kept_aux_graphs = AuxiliaryGraphs.parse_obj({})
    for result in message.results or []:
        keep = True
        for qnode_id, node_bindings in result.node_bindings.items():
            if qnode_id in pinned_nodes:
                # don't filter any pinned nodes
                continue
            for node_binding in node_bindings:
                curie = curie_map.get(node_binding.id)
                if (
                    (
                        curie is not None
                        and curie.information_content < information_content_threshold
                        and not last_hop
                    )
                    or (
                        # Unknown UMLS curies are bound to be low information content
                        curie is None
                        and node_binding.id.startswith("UMLS")
                        and not last_hop
                    )
                    or (
                        # UMLS curies where preferred query is still UMLS are bound to be low information content
                        curie is not None
                        and node_binding.id.startswith("UMLS")
                        and curie.preferred_curie.startswith("UMLS")
                        and not last_hop
                    )
                    or (
                        # Nodes that appear in the blocklist shouldn't be shown
                        curie is not None
                        and curie.preferred_curie in blocklist
                    )
                    or (
                        curie is None
                        and node_binding.id in blocklist
                    )
                ):
                    keep = False
                    if node_binding.id in message.knowledge_graph.nodes:
                        # remove nodes from kgraph
                        logger.debug(
                            f"Removing {node_binding.id} because it's highly promiscuous"
                        )
                        message.knowledge_graph.nodes.pop(node_binding.id)
        if keep:
            try:
                for analysis in result.analyses or []:
                    # add support graphs from result
                    for support_graph_id in analysis.support_graphs or []:
                        kept_aux_graphs[support_graph_id] = message.auxiliary_graphs[
                            support_graph_id
                        ]
                        for edge_id in message.auxiliary_graphs[support_graph_id].edges:
                            kept_knowledge_graph.edges[
                                edge_id
                            ] = message.knowledge_graph.edges[edge_id]
                    # add edges from result
                    for edge_bindings in analysis.edge_bindings.values():
                        for edge_binding in edge_bindings:
                            kept_knowledge_graph.edges[
                                edge_binding.id
                            ] = message.knowledge_graph.edges[edge_binding.id]
                            # add support graphs from edge
                            for attribute in (
                                kept_knowledge_graph.edges[edge_binding.id].attributes
                                or []
                            ):
                                if (
                                    attribute.attribute_type_id
                                    == "biolink:support_graphs"
                                ):
                                    for aux_graph_id in attribute.value:
                                        kept_aux_graphs[
                                            aux_graph_id
                                        ] = message.auxiliary_graphs[aux_graph_id]
                                        for edge_id in message.auxiliary_graphs[
                                            aux_graph_id
                                        ].edges:
                                            kept_knowledge_graph.edges[
                                                edge_id
                                            ] = message.knowledge_graph.edges[edge_id]
                # keep any results that don't have promiscuous nodes
                # only add result if all the knowledge and aux graph stuff worked out
                kept_results.append(result)
            except Exception as e:
                logger.error(f"Error while filtering message: {str(e)}")
                logger.debug(traceback.format_exc())

    message.results = kept_results
    if message.knowledge_graph:
        message.knowledge_graph.edges = kept_knowledge_graph.edges
    else:
        message.knowledge_graph = kept_knowledge_graph
    message.auxiliary_graphs = kept_aux_graphs


async def fill_categories_predicates(
    qg: QueryGraph,
    logger: logging.Logger = logging.getLogger(),
    normalizer: Normalizer = None,
):
    """
    Given a query graph, fill in missing categories and predicates
    using the normalizer or with the most general terms (NamedThing and related_to).
    """

    if normalizer is None:
        normalizer = Normalizer(logger)

    # Fill in missing predicates with most general term
    for edge in qg["edges"].values():
        if ("predicates" not in edge) or (edge["predicates"] is None):
            edge["predicates"] = ["biolink:related_to"]

    # Fill in missing categories with most general term
    for node in qg["nodes"].values():
        if ("categories" not in node) or (node["categories"] is None):
            node["categories"] = ["biolink:NamedThing"]

    # Use node normalizer to add
    # a category to nodes with a curie
    for node in qg["nodes"].values():
        node_id = node.get("ids", None)
        if not node_id:
            if (
                "biolink:Gene" in node["categories"]
                and "biolink:Protein" not in node["categories"]
            ):
                node["categories"].append("biolink:Protein")
            if (
                "biolink:Protein" in node["categories"]
                and "biolink:Gene" not in node["categories"]
            ):
                node["categories"].append("biolink:Gene")
        else:
            if not isinstance(node_id, list):
                node_id = [node_id]

            # Get full list of categorys
            categories = await normalizer.get_types(node_id)

            # Remove duplicates
            categories = list(set(categories))

            if categories:
                # Filter categorys that are ancestors of other categorys we were given
                node["categories"] = filter_ancestor_types(categories)
            elif "categories" not in node:
                node["categories"] = []
