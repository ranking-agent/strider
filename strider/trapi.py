"""TRAPI utilities."""
from collections import defaultdict
import copy
import json
import logging
import hashlib

import bmt
from reasoner_pydantic import Message, Result, QNode, Node, Edge, KnowledgeGraph
from reasoner_pydantic.qgraph import QueryGraph

from strider.util import deduplicate_by, WBMT, ensure_list, filter_none, get_from_all, \
    build_predicate_direction, extract_predicate_direction, \
    deduplicate, listify_value, merge_listify, all_equal
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


def merge_nodes(knodes: list[Node]) -> Node:
    """ Smart merge function for KNodes """
    output_knode = {}

    # We don't really know how to merge names
    # so we just pick the first we are given
    name_values = get_from_all(knodes, "name")
    if name_values:
        output_knode["name"] = name_values[0]

    category_values = get_from_all(knodes, "categories")
    if category_values:
        output_knode["categories"] = \
            deduplicate(merge_listify(category_values))

    attributes_values = get_from_all(knodes, "attributes")
    if attributes_values:
        output_knode["attributes"] = \
            deduplicate_by(
                filter_none(merge_listify(attributes_values)),
                lambda d: json.dumps(d, sort_keys=True))

    return output_knode


def merge_edges(kedges: list[Edge]) -> Edge:
    """ Smart merge function for KEdges """
    output_kedge = {}

    attributes_values = get_from_all(kedges, "attributes")
    if attributes_values:
        output_kedge["attributes"] = \
            deduplicate_by(
                filter_none(merge_listify(attributes_values)),
                lambda d: json.dumps(d, sort_keys=True))

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


def merge_kgraphs(kgraphs: list[KnowledgeGraph]) -> KnowledgeGraph:
    """ Merge knowledge graphs. """

    knodes = [kgraph["nodes"] for kgraph in kgraphs]
    kedges = [kgraph["edges"] for kgraph in kgraphs]

    # Merge Nodes
    output = {"nodes": {}, "edges": {}}

    all_node_keys = set()
    for kgraph in kgraphs:
        all_node_keys.update(kgraph["nodes"].keys())
    for node_key in all_node_keys:
        node_values = get_from_all(knodes, node_key)
        merged_node = merge_nodes(node_values)
        output["nodes"][node_key] = merged_node

    # Merge Edges
    all_edge_keys = set()
    for kgraph in kgraphs:
        all_edge_keys.update(kgraph["edges"].keys())
    for edge_key in all_edge_keys:
        edge_values = get_from_all(kedges, edge_key)
        merged_edge = merge_edges(edge_values)
        output["edges"][edge_key] = merged_edge

    return output


def merge_messages(messages: list[Message]) -> Message:
    """Merge messages."""

    # Build knowledge graph edge IDs so that we can merge duplicates
    for m in messages:
        build_unique_kg_edge_ids(m)

    kgraphs = [m["knowledge_graph"] for m in messages]

    results = []
    for message in messages:
        results.extend(message["results"])
    results_deduplicated = deduplicate_by(results, result_hash)

    return {
        "query_graph": messages[0]["query_graph"],
        "knowledge_graph": merge_kgraphs(kgraphs),
        "results": results_deduplicated
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
    if message.get("query_graph") is not None:
        for qnode in message['query_graph']['nodes'].values():
            if qnode_id := qnode.get("ids", False):
                curies |= set(qnode_id)
    if message.get("knowledge_graph") is not None:
        curies |= set(message['knowledge_graph']['nodes'])
    return curies


def apply_curie_map(message: Message, curie_map: dict[str, str]) -> Message:
    """Translate all pinned qnodes to preferred prefix."""
    new_message = dict()
    new_message["query_graph"] = map_qgraph_curies(message["query_graph"], curie_map)
    if message.get("knowledge_graph") is not None:
        kgraph = message["knowledge_graph"]
        new_message['knowledge_graph'] = {
            'nodes': {
                curie_map.get(knode_id, [knode_id])[0]: knode
                for knode_id, knode in kgraph['nodes'].items()
            },
            'edges': {
                kedge_id: fix_kedge(kedge, curie_map)
                for kedge_id, kedge in kgraph['edges'].items()
            },
        }
    if message.get("results") is not None:
        results = message["results"]
        new_message['results'] = [
            fix_result(result, curie_map)
            for result in results
        ]
    return new_message


def map_qgraph_curies(
        qgraph: QueryGraph,
        curie_map: dict[str, str],
        primary: bool = False,
) -> QueryGraph:
    """Replace curies with preferred, if possible."""
    if qgraph is None:
        return None
    return {
        'nodes': {
            qnode_id: map_qnode_curies(qnode, curie_map, primary)
            for qnode_id, qnode in qgraph['nodes'].items()
        },
        "edges": qgraph["edges"],
    }


def canonicalize_qgraph(
        qgraph: QueryGraph,
) -> QueryGraph:
    """Replace predicates with canonical directions."""
    return {
        'nodes': qgraph['nodes'],
        "edges": {
            qedge_id: canonicalize_qedge(qedge)
            for qedge_id, qedge in qgraph["edges"].items()
        },
    }


def canonicalize_qedge(
    qedge: dict,
):
    """Use canonical predicate direction."""
    qedge = copy.deepcopy(qedge)
    flipped = None
    predicates = []
    for predicate in qedge.get("predicates") or []:
        slot = WBMT.bmt.get_element(predicate)
        if slot is None:
            flipped = False
            continue
        is_canonical = slot.annotations.get("biolink:canonical_predicate", False)
        if flipped is None:
            flipped = not is_canonical
        elif flipped == is_canonical:
            raise NotImplementedError("There are multiple predicates, mixed canonical and not")
        if flipped:
            predicates.append(bmt.util.format(slot.inverse, case="snake"))
    if flipped:
        qedge["subject"], qedge["object"] = qedge["object"], qedge["subject"]
        qedge["predicates"] = predicates
    return qedge


def map_qnode_curies(
        qnode: QNode,
        curie_map: dict[str, str],
        primary: bool = False,
) -> QNode:
    """Replace curie with preferred, if possible."""
    qnode = copy.deepcopy(qnode)
    if not qnode.get("ids", None):
        return qnode

    listify_value(qnode, "ids")

    output_curies = []
    for existing_curie in qnode["ids"]:
        if primary:
            output_curies.append(
                curie_map.get(existing_curie, [existing_curie])[0]
            )
        else:
            output_curies.extend(
                curie_map.get(existing_curie, [])
            )
    if len(output_curies) == 0:
        output_curies = qnode["ids"]

    qnode = {
        **qnode,
        "ids": output_curies,
    }
    return qnode


def fix_knode(knode: Node, curie_map: dict[str, str]) -> Node:
    """Replace curie with preferred, if possible."""
    knode = {
        **knode,
        "id": curie_map.get(knode["id"], [knode["id"]])[0]
    }
    return knode


def fix_kedge(kedge: Edge, curie_map: dict[str, str]) -> Edge:
    """Replace curie with preferred, if possible."""
    kedge["subject"] = curie_map.get(kedge["subject"], [kedge["subject"]])[0]
    kedge["object"] = curie_map.get(kedge["object"], [kedge["object"]])[0]
    return kedge


def fix_result(result: Result, curie_map: dict[str, str]) -> Result:
    """Replace curie with preferred, if possible."""
    result['node_bindings'] = {
        qnode_id: [
            {
                **node_binding,
                "id": curie_map.get(
                    node_binding["id"],
                    [node_binding["id"]]
                )[0],
            }
            for node_binding in node_bindings
        ]
        for qnode_id, node_bindings in result['node_bindings'].items()
    }
    return result


def filter_ancestor_types(categories):
    """ Filter out types that are ancestors of other types in the list """

    def is_ancestor(a, b):
        """ Check if one biolink type is an ancestor of the other """
        ancestors = WBMT.get_ancestors(b, reflexive=False)
        if a in ancestors:
            return True
        return False

    has_descendant = [
        any(
            is_ancestor(categories[idx], a)
            for a in categories[:idx] + categories[idx + 1:]
        )
        for idx in range(len(categories))
    ]
    return [
        category
        for category, drop in zip(categories, has_descendant)
        if not drop
    ]


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
    for edge in qg['edges'].values():
        if ('predicates' not in edge) or (edge['predicates'] is None):
            edge['predicates'] = ['biolink:related_to']

    # Fill in missing categories with most general term
    for node in qg['nodes'].values():
        if ('categories' not in node) or (node['categories'] is None):
            node['categories'] = ['biolink:NamedThing']
        elif "biolink:ChemicalSubstance" in node["categories"]:
            node["categories"].append("biolink:SmallMolecule")

    logger.debug("Contacting node normalizer to get categorys for curies")

    # Use node normalizer to add
    # a category to nodes with a curie
    for node in qg['nodes'].values():
        node_id = node.get('ids', None)
        if not node_id:
            continue
        if not isinstance(node_id, list):
            node_id = [node_id]

        # Get full list of categorys
        categories = await normalizer.get_types(node_id)
        if "biolink:SmallMolecule" in categories:
            categories.append("biolink:ChemicalSubstance")

        # Remove duplicates
        categories = list(set(categories))

        if categories:
            # Filter categorys that are ancestors of other categorys we were given
            node['categories'] = filter_ancestor_types(categories)
        elif "categories" not in node:
            node["categories"] = []


def build_unique_kg_edge_ids(message: Message):
    """
    Replace KG edge IDs with a string that represents
    whether the edge can be merged with other edges
    """

    # Make a copy of the edge keys because we're about to change them
    for edge_id in list(message["knowledge_graph"]["edges"].keys()):
        edge = message["knowledge_graph"]["edges"].pop(edge_id)
        new_edge_id_string = f"{edge['subject']}-{edge['predicate']}-{edge['object']}"

        # Build hash from ID string
        new_edge_id = hashlib.blake2b(
            new_edge_id_string.encode(),
            digest_size=6,
        ).hexdigest()

        # Update knowledge graph
        message["knowledge_graph"]["edges"][new_edge_id] = edge

        # Update results
        for result in message["results"]:
            for edge_binding_list in result["edge_bindings"].values():
                for eb in edge_binding_list:
                    if eb["id"] == edge_id:
                        eb["id"] = new_edge_id


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
            qgraph_allowable_categories.extend(
                WBMT.get_descendants(c)
            )

        # Check that at least one of the categories
        # on this kgraph node is allowed
        if not any(
            c in qgraph_allowable_categories
            for c in kgraph_node["categories"]
        ):
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
            allowable_predicates.extend(
                WBMT.get_descendants(p)
            )
            p_inverse = WBMT.predicate_inverse(p)
            if p_inverse:
                allowable_predicates.extend(
                    WBMT.get_descendants(p_inverse)
                )


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
        result for result in message["results"]
        if all(
            is_valid_node_binding(message, nb, qgraph["nodes"][qg_id])
            for qg_id, nb_list in result["node_bindings"].items()
            for nb in nb_list
        ) and all(
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
        nid:node for nid,node in message["knowledge_graph"]["nodes"].items()
        if nid in bound_knodes
    }
    message["knowledge_graph"]["edges"] = {
        eid:edge for eid,edge in message["knowledge_graph"]["edges"].items()
        if eid in bound_kedges
    }
