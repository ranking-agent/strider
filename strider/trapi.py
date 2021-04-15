"""TRAPI utilities."""
from collections import defaultdict
import logging

from reasoner_pydantic import Message, Result, QNode, Node, Edge, KnowledgeGraph
from reasoner_pydantic.qgraph import QueryGraph

from strider.util import deduplicate_by, WrappedBMT, get_from_all, \
    build_predicate_direction, extract_predicate_direction, \
    deduplicate, merge_listify, all_equal
from strider.normalizer import Normalizer
from strider.config import settings

WBMT = WrappedBMT()


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

    category_values = get_from_all(knodes, "category")
    if category_values:
        output_knode["category"] = \
            deduplicate(merge_listify(category_values))

    attributes_values = get_from_all(knodes, "attributes")
    if attributes_values:
        output_knode["attributes"] = \
            merge_listify(attributes_values)

    return output_knode

def merge_edges(kedges: list[Edge]) -> Edge:
    """ Smart merge function for KEdges """
    output_kedge = {}

    attributes_values = get_from_all(kedges, "attributes")
    if attributes_values:
        output_kedge["attributes"] = \
            merge_listify(attributes_values)

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
    output = {"nodes": {}, "edges" : {}}

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


def filter_ancestor_types(types):
    """ Filter out types that are ancestors of other types in the list """

    def is_ancestor(a, b):
        """ Check if one biolink type is an ancestor of the other """
        ancestors = WBMT.get_ancestors(b)
        if a in ancestors:
            return True
        return False

    specific_types = ['biolink:NamedThing']
    for new_type in types:
        for existing_type_id, existing_type in enumerate(specific_types):
            existing_type = specific_types[existing_type_id]
            if is_ancestor(new_type, existing_type):
                continue
            if is_ancestor(existing_type, new_type):
                specific_types[existing_type_id] = new_type
            else:
                specific_types.append(new_type)
    return specific_types


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
        normalizer = Normalizer(settings.normalizer_url)

    # Fill in missing predicates with most general term
    for edge in qg['edges'].values():
        if ('predicate' not in edge) or (edge['predicate'] is None):
            edge['predicate'] = 'biolink:related_to'

    # Fill in missing categories with most general term
    for node in qg['nodes'].values():
        if ('category' not in node) or (node['category'] is None):
            node['category'] = 'biolink:NamedThing'

    logger.debug("Contacting node normalizer to get categorys for curies")

    # Use node normalizer to add
    # a category to nodes with a curie
    for node in qg['nodes'].values():
        node_id = node.get('id', None)
        if not node_id:
            continue
        if not isinstance(node_id, list):
            node_id = [node_id]

        # Get full list of categorys
        categories = await normalizer.get_types(node_id)

        if categories:
            # Filter categorys that are ancestors of other categorys we were given
            node['category'] = filter_ancestor_types(categories)
        elif "category" not in node:
            node["category"] = []


def add_descendants(
        qg: QueryGraph,
        logger: logging.Logger = logging.getLogger(),
) -> QueryGraph:
    """
    Use the Biolink Model Toolkit to add descendants
    to categories and predicates in a graph.
    """

    logger.debug("Using BMT to get descendants of node and edge types")

    # Use BMT to convert node categorys to categorys + descendants
    for node in qg['nodes'].values():
        if 'category' not in node:
            continue
        new_category_list = []
        for t in node['category']:
            new_category_list.extend(WBMT.get_descendants(t))
        node['category'] = new_category_list

    # Same with edges
    for edge in qg['edges'].values():
        if 'predicate' not in edge:
            continue
        new_predicate_list = []
        for t in edge['predicate']:
            predicate, direction = extract_predicate_direction(t)
            for new_predicate in WBMT.get_descendants(predicate):
                new_predicate_list.append(
                    build_predicate_direction(new_predicate, direction))
        edge['predicate'] = new_predicate_list

    logger.debug({
        "description": "Expanded query graph with descendants",
        "qg": qg,
    })

    return qg


def build_unique_kg_edge_ids(message: Message):
    """
    Replace KG edge IDs with a string that represents
    whether the edge can be merged with other edges
    """
    for edge_id in list(message["knowledge_graph"]["edges"].keys()):
        edge = message["knowledge_graph"]["edges"].pop(edge_id)
        new_edge_id = f"{edge['subject']}-{edge['predicate']}-{edge['object']}"

        # Update knowledge graph
        message["knowledge_graph"]["edges"][new_edge_id] = edge

        # Update results
        for result in message["results"]:
            for edge_binding_list in result["edge_bindings"].values():
                for eb in edge_binding_list:
                    if eb["id"] == edge_id:
                        eb["id"] = new_edge_id
