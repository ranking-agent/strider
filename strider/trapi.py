"""TRAPI utilities."""
from collections import defaultdict
import copy
from itertools import product
import logging
import hashlib
import typing

import bmt
from reasoner_pydantic import Message, Result, QNode, Node, Edge
from reasoner_pydantic.qgraph import QEdge, QueryGraph
from reasoner_pydantic.shared import BiolinkPredicate
from reasoner_pydantic.utils import HashableSequence

from strider.util import (
    deduplicate_by,
    WBMT,
    ensure_list,
    filter_none,
    get_from_all,
    build_predicate_direction,
    extract_predicate_direction,
    deduplicate,
    listify_value,
    merge_listify,
    all_equal,
)
from strider.normalizer import Normalizer
from strider.config import settings


def result_hash(result):
    """
    Given a results object, generate a hashable value that
    can be used for comparison.
    """
    node_bindings_information = frozenset(
        (key, frozenset(bound["id"] for bound in value))
        for key, value in result["node_bindings"].items()
    )
    edge_bindings_information = frozenset(
        (key, frozenset(bound["id"] for bound in value))
        for key, value in result["edge_bindings"].items()
    )
    return (node_bindings_information, edge_bindings_information)


def attribute_hash(attribute):
    """
    Given an attribute object, generate a hashable value
    that can be used for comparison
    """
    uid = ""

    # When we iterate over attribute properties use sorted
    # to get a consistent ordering
    for property_name in sorted(attribute.keys()):
        property_value = attribute[property_name]
        if property_name == "attributes" and property_value is not None:
            # These are sub-attributes
            # Add each to the string
            for a in property_value:
                uid += attribute_hash(a)
        else:
            uid += f"{property_name}={str(property_value)}&"

    return uid


def merge_nodes(knodes: list[Node]) -> Node:
    """Smart merge function for KNodes"""
    output_knode = {}

    # We don't really know how to merge names
    # so we just pick the first we are given
    name_values = get_from_all(knodes, "name")
    if name_values:
        output_knode["name"] = name_values[0]

    category_values = get_from_all(knodes, "categories")
    if category_values:
        output_knode["categories"] = deduplicate(merge_listify(category_values))

    attributes_values = get_from_all(knodes, "attributes")
    if attributes_values:
        # Flatten list
        attributes_values = merge_listify(attributes_values)
        # Filter out empty values
        attributes_values = filter_none(attributes_values)
        # Deduplicate
        attributes_values = deduplicate_by(attributes_values, attribute_hash)
        # Write to output kedge
        output_knode["attributes"] = attributes_values

    return output_knode


def merge_edges(kedges: list[Edge]) -> Edge:
    """Smart merge function for KEdges"""
    output_kedge = {}

    attributes_values = get_from_all(kedges, "attributes")
    if attributes_values:
        # Flatten list
        attributes_values = merge_listify(attributes_values)
        # Filter out empty values
        attributes_values = filter_none(attributes_values)
        # Deduplicate
        attributes_values = deduplicate_by(attributes_values, attribute_hash)
        # Write to output kedge
        output_kedge["attributes"] = attributes_values

    predicate_values = get_from_all(kedges, "predicate")
    if not all_equal(predicate_values):
        raise ValueError("Unable to merge edges with non matching predicates")
    output_kedge["predicate"] = predicate_values[0]

    subject_values = get_from_all(kedges, "subject")
    if not all_equal(subject_values):
        raise ValueError("Unable to merge edges with non matching subjects")
    output_kedge["subject"] = subject_values[0]

    object_values = get_from_all(kedges, "object")
    if not all_equal(object_values):
        raise ValueError("Unable to merge edges with non matching objects")
    output_kedge["object"] = object_values[0]

    return output_kedge


def get_curies(message: Message) -> list[str]:
    """Get all node curies used in message.

    Do not examine kedge source and target ids. There ought to be corresponding
    knodes.
    """
    curies = set()
    if message.query_graph is not None:
        for qnode in message.query_graph.nodes.values():
            if qnode.ids:
                curies |= set(qnode.ids)
    if message.knowledge_graph is not None:
        curies |= set(message.knowledge_graph.nodes.keys())
    return list(curies)


def apply_curie_map(message: Message, curie_map: dict[str, str]):
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
        for r in message.results:
            fix_result(r, curie_map)


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
        is_canonical = slot.annotations.get("biolink:canonical_predicate", False)
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


def fix_result(result: Result, curie_map: dict[str, str]):
    """Replace curie with preferred, if possible."""
    for nb_set in result.node_bindings.values():
        for nb in nb_set:
            nb.id = curie_map.get(nb.id, [nb.id])[0]


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
        normalizer = Normalizer(settings.normalizer_url, logger)

    # Fill in missing predicates with most general term
    for edge in qg["edges"].values():
        if ("predicates" not in edge) or (edge["predicates"] is None):
            edge["predicates"] = ["biolink:related_to"]

    # Fill in missing categories with most general term
    for node in qg["nodes"].values():
        if ("categories" not in node) or (node["categories"] is None):
            node["categories"] = ["biolink:NamedThing"]

    logger.debug("Contacting node normalizer to get categorys for curies")

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


def is_valid_node_binding(message, nb, qgraph_node):
    """
    Check whether the given kgraph node
    satisifies the given qgraph node
    """
    if qgraph_node.get("ids", None) is not None:
        if nb["id"] not in qgraph_node["ids"]:
            return False

    kgraph_node = message["knowledge_graph"]["nodes"][nb["id"]]
    if qgraph_node.get("categories", None) is not None:
        # Build list of allowable categories for kgraph nodes
        qgraph_allowable_categories = []
        for c in ensure_list(qgraph_node["categories"]):
            qgraph_allowable_categories.extend(WBMT.get_descendants(c))

        # Check that at least one of the categories
        # on this kgraph node is allowed
        if not any(c in qgraph_allowable_categories for c in kgraph_node["categories"]):
            return False
    return True


def is_valid_edge_binding(message, eb, qgraph_edge):
    """
    Check whether the given kgraph edge
    satisifies the given qgraph edge
    """
    kgraph_edge = message["knowledge_graph"]["edges"][eb["id"]]
    if qgraph_edge.get("predicates", None) is not None:
        # Build list of allowable predicates for kgraph edges
        allowable_predicates = []
        for p in ensure_list(qgraph_edge["predicates"]):
            allowable_predicates.extend(WBMT.get_descendants(p))
            p_inverse = WBMT.predicate_inverse(p)
            if p_inverse:
                allowable_predicates.extend(WBMT.get_descendants(p_inverse))

        # Check that all predicates on this
        # kgraph edge are allowed
        if kgraph_edge["predicate"] not in allowable_predicates:
            return False

    return True


def filter_by_qgraph(message, qgraph):
    """
    Filter a message to ensure that all results
    and edges match the given query graph
    """

    # Only keep results where all edge
    # and node bindings are valid
    message["results"] = [
        result
        for result in message["results"]
        if all(
            is_valid_node_binding(message, nb, qgraph["nodes"][qg_id])
            for qg_id, nb_list in result["node_bindings"].items()
            for nb in nb_list
        )
        and all(
            is_valid_edge_binding(message, eb, qgraph["edges"][qg_id])
            for qg_id, eb_list in result["edge_bindings"].items()
            for eb in eb_list
        )
    ]

    remove_unbound_from_kg(message)


def remove_unbound_from_kg(message):
    """
    Remove all knowledge graph nodes and edges without a binding
    """

    bound_knodes = set()
    for result in message["results"]:
        for node_binding_list in result["node_bindings"].values():
            for nb in node_binding_list:
                bound_knodes.add(nb["id"])
    bound_kedges = set()
    for result in message["results"]:
        for edge_binding_list in result["edge_bindings"].values():
            for nb in edge_binding_list:
                bound_kedges.add(nb["id"])

    message["knowledge_graph"]["nodes"] = {
        nid: node
        for nid, node in message["knowledge_graph"]["nodes"].items()
        if nid in bound_knodes
    }
    message["knowledge_graph"]["edges"] = {
        eid: edge
        for eid, edge in message["knowledge_graph"]["edges"].items()
        if eid in bound_kedges
    }
