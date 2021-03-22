"""Query planner."""
from collections import defaultdict, namedtuple
import itertools
import logging
import copy
from typing import Generator

from reasoner_pydantic import QueryGraph

from strider.kp_registry import Registry
from strider.util import WrappedBMT, extract_predicate_direction
from strider.config import settings
from strider.trapi import add_descendants

LOGGER = logging.getLogger(__name__)

Step = namedtuple("Step", ["source", "edge", "target"])
Operation = namedtuple(
    "Operation", ["source_category", "edge_predicate", "target_category"])

REVERSE_EDGE_SUFFIX = '.reverse'
SYMMETRIC_EDGE_SUFFIX = '.symmetric'
INVERSE_EDGE_SUFFIX = '.inverse'


WBMT = WrappedBMT()


def find_next_list_property(search_dict, fields_to_check):
    """ Find first object in a dictionary where object[field] is a list """
    for key, val in search_dict.items():
        for field in fields_to_check:
            if field in val and isinstance(val[field], list):
                return key, field
    return None, None


def permute_graph(
        graph: dict,
        node_fields: list[str] = [],
        edge_fields: list[str] = [],
) -> Generator[dict, None, None]:
    """
    Take in a graph that has some unbound properties
    and return a list of query graphs where every property is bound

    Example: If a graph has ['Disease', 'Gene'] as a type for a node,
    two query graphs will be returned, one with node type Disease and one with type Gene
    """

    stack = []
    stack.append(copy.deepcopy(graph))

    while len(stack) > 0:
        current_graph = stack.pop()

        # Find our next node to permute
        next_node_id, next_node_field = \
            find_next_list_property(
                current_graph['nodes'], node_fields)

        if next_node_id:
            # Permute this node and push permutations to stack
            next_node = current_graph['nodes'][next_node_id]
            for field_value in next_node[next_node_field]:
                permutation_copy = copy.deepcopy(current_graph)
                # Fix field value
                permutation_copy['nodes'][next_node_id][next_node_field] = field_value
                # Add to stack
                stack.append(permutation_copy)
            continue

        # Find our next edge to permute
        next_edge_id, next_edge_field = \
            find_next_list_property(
                current_graph['edges'], edge_fields)

        if next_edge_id:
            # Permute this edge and push permutations to stack
            next_edge = current_graph['edges'][next_edge_id]
            for field_value in next_edge[next_edge_field]:
                permutation_copy = copy.deepcopy(current_graph)
                # Fix predicate
                permutation_copy['edges'][next_edge_id][next_edge_field] = field_value
                # Add to stack
                stack.append(permutation_copy)
            continue

        # This graph is fully permuted, add to results
        yield current_graph


class NoAnswersError(Exception):
    """No answers can be found."""


async def annotate_operation_graph(
        operation_graph: dict[str, dict],
        kp_registry: Registry = None,
):
    """
    Look up kps for each edge in an operation graph
    and add it to the "kps" property of the edge
    """
    if kp_registry is None:
        kp_registry = Registry(settings.kpregistry_url)

    for edge_id in list(operation_graph['edges'].keys()):
        edge = operation_graph["edges"][edge_id]
        if "provided_by" in edge:
            allowlist = edge["provided_by"].get("allowlist", None)
            denylist = edge["provided_by"].get("denylist", None)
        else:
            allowlist = denylist = None

        source = operation_graph['nodes'][edge['source']]
        target = operation_graph['nodes'][edge['target']]

        kp_results = await step_to_kps(
            source, edge, target,
            kp_registry,
            allowlist=allowlist, denylist=denylist,
        )

        if len(kp_results) == 0:
            del operation_graph["edges"][edge_id]
            continue

        edge["kps"] = []
        for kp_name, kp_result in kp_results.items():
            operations = kp_result.pop("operations")
            for operation in operations:
                edge["kps"].append({
                    "name": kp_name,
                    **kp_result,
                    **operation,
                })


def make_og_edge(
        qg_edge_id: str,
        qg_edge: dict,
        edge_reverse: bool,
        predicate_reverse: bool,
        predicates: list = None,
) -> dict:
    """
    Build an operation graph edge from a query graph.

    Accepts parameters to invert the edge direction (source -> target, target -> source)
    and the predicate direction (-treats-> or <-treats-).

    Also accepts a custom list of predicates.
    """
    og_edge = {}

    # Annotate operation graph edge
    # with the query graph edge_id
    og_edge["qg_edge_id"] = qg_edge_id
    # Also annotate with whether we are traversing
    # this edge in reverse
    og_edge["qg_traversal_reverse"] = predicate_reverse

    if edge_reverse:
        og_edge["source"] = qg_edge["object"]
        og_edge["target"] = qg_edge["subject"]
    else:
        og_edge["source"] = qg_edge["subject"]
        og_edge["target"] = qg_edge["object"]

    # Default value
    if not predicates:
        predicates = qg_edge["predicate"]

    if predicate_reverse:
        og_edge["predicate"] = [f"<-{p}-" for p in predicates]
    else:
        og_edge["predicate"] = [f"-{p}->" for p in predicates]
    return og_edge


async def qg_to_og(qgraph):
    """
    Convert query graph to operation graph

    This will add reverse, symmetric,
    and inverse edges when they are available
    """
    ograph = {
        "nodes": copy.deepcopy(qgraph["nodes"]),
        "edges": dict(),
    }
    # Add reverse edges so that we can traverse edges in both directions
    for edge_id, edge in qgraph["edges"].items():
        # Forward edge
        ograph["edges"][edge_id] = make_og_edge(
            edge_id, edge,
            edge_reverse=False,
            predicate_reverse=False
        )
        # Reverse edge
        ograph["edges"][edge_id + REVERSE_EDGE_SUFFIX] = make_og_edge(
            edge_id, edge,
            edge_reverse=True,
            predicate_reverse=True
        )

        symmetric_predicates = [
            p for p in edge["predicate"]
            if WBMT.predicate_is_symmetric(p)
        ]
        if len(symmetric_predicates) > 0:
            # Forward symmetric edge
            ograph["edges"][edge_id + SYMMETRIC_EDGE_SUFFIX] = make_og_edge(
                edge_id, edge,
                edge_reverse=False,
                predicate_reverse=True,
                predicates=symmetric_predicates)
            # Reverse symmetric edge
            ograph["edges"][edge_id + REVERSE_EDGE_SUFFIX + SYMMETRIC_EDGE_SUFFIX] = \
                make_og_edge(
                edge_id, edge,
                edge_reverse=True,
                predicate_reverse=False,
                predicates=symmetric_predicates)

        # Inverse edge
        inverse_predicates = [
            WBMT.predicate_inverse(p) for p in edge["predicate"]
        ]
        inverse_predicates = list(filter(None, inverse_predicates))
        if len(inverse_predicates) > 0:
            # Forward inverse edge
            ograph["edges"][edge_id + INVERSE_EDGE_SUFFIX] = make_og_edge(
                edge_id, edge,
                edge_reverse=True,
                predicate_reverse=False,
                predicates=inverse_predicates)
            # Reverse inverse edge
            ograph["edges"][edge_id + REVERSE_EDGE_SUFFIX + INVERSE_EDGE_SUFFIX] = \
                make_og_edge(
                edge_id, edge,
                edge_reverse=False,
                predicate_reverse=True,
                predicates=inverse_predicates
            )

    return ograph


def fix_categories_predicates(query_graph):
    """
    Given a permuted query graph with one KP,
    fix the node categories and predicates to match the KP

    Returns false if there is a conflicting node type
    """
    for edge in query_graph['edges'].values():
        kp = edge["kps"]
        reverse = kp["reverse"]

        if reverse:
            source_node = query_graph["nodes"][edge["object"]]
            target_node = query_graph["nodes"][edge["subject"]]
        else:
            source_node = query_graph["nodes"][edge["subject"]]
            target_node = query_graph["nodes"][edge["object"]]

        existing_source_category = source_node["category"] \
            if not isinstance(source_node["category"], list) else None
        new_source_category = kp["source_category"]
        if existing_source_category and \
           existing_source_category != new_source_category:
            return False
        source_node["category"] = new_source_category

        existing_target_category = target_node["category"] \
            if not isinstance(target_node["category"], list) else None
        new_target_category = kp["target_category"]
        if existing_target_category and \
           existing_target_category != new_target_category:
            return False
        target_node["category"] = new_target_category

        existing_predicate = edge["predicate"] \
            if not isinstance(edge["predicate"], list) else None
        new_predicate = kp["edge_predicate"]
        if existing_predicate and \
           existing_predicate != new_predicate:
            return False
        edge["predicate"] = new_predicate
    return True


def filter_categories_predicates(operation_graph):
    """
    Filter out categories and predicates that are not defined in any KP
    """
    for node in operation_graph['nodes'].values():
        node['filtered_categories'] = set()
    for edge in operation_graph['edges'].values():
        edge['filtered_predicates'] = set()

    for edge in operation_graph['edges'].values():
        for kp in edge["kps"].values():
            for kp_operation in kp["operations"]:
                source_node = operation_graph["nodes"][edge["source"]]
                target_node = operation_graph["nodes"][edge["target"]]

                source_node['filtered_categories'].add(
                    kp_operation["source_category"])
                target_node['filtered_categories'].add(
                    kp_operation["target_category"])
                edge['filtered_predicates'].add(
                    kp_operation["edge_predicate"]
                )

    # We only filter nodes that are touching at least one edge ("connected")
    connected_nodes = set()
    for edge in operation_graph['edges'].values():
        connected_nodes.add(edge['source'])
        connected_nodes.add(edge['target'])
    for node_id in connected_nodes:
        node = operation_graph['nodes'][node_id]
        if 'category' not in node:
            continue
        node['category'] = list(node.pop('filtered_categories'))

    # Also filter edge predicates as well
    for edge in operation_graph['edges'].values():
        edge['predicate'] = list(edge.pop('filtered_predicates'))


def get_next_nodes(
        graph: dict[str, list[str]],
        path: list[str],
):
    """
    Find next nodes to traverse

    Prefer nodes that are adjacent to the last
    node in the path.
    """
    for node in reversed(path):
        adjacent_nodes = graph[node]
        valid_adjacent_nodes = [
            node for node in adjacent_nodes
            if node not in path
        ]
        if len(valid_adjacent_nodes) > 0:
            return valid_adjacent_nodes
    return []


def traversals_from_node(
    graph: dict[str, list[str]],
    source: str,
) -> Generator[list[str], None, None]:
    """
    Yield all traversals originating from a source node.
    """

    # Initialize stack
    stack = [[source]]

    while len(stack) != 0:
        current_path = stack.pop()

        # Find next nodes to traverse
        next_nodes = get_next_nodes(graph, current_path)

        # Visited every node possible, done now
        if len(next_nodes) == 0:
            yield current_path

        for next_node in next_nodes:
            # Add to stack for further iteration
            new_path = current_path.copy()
            new_path.append(next_node)
            stack.append(new_path)


def ensure_traversal_connected(graph, path):
    """
    Validate a traversal to make sure that it covers all unpinned
    nodes. This may not be the case if there are two disconnected
    components in the query graph.
    """

    graph_nodes = set(graph['nodes'].keys())
    pinned_nodes = {
        key for key, value in graph["nodes"].items()
        if value.get("id", None) is not None
    }
    path_nodes = {n for n in path if n in graph["nodes"].keys()}

    # path_nodes + pinned_nodes must cover all nodes in the graph
    return pinned_nodes | path_nodes == graph_nodes


# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
async def generate_plans(
        qgraph: QueryGraph,
        kp_registry: Registry = None,
        logger: logging.Logger = LOGGER,
) -> list[dict[Step, list]]:
    """
    Given a query graph, build plans that consists of steps
    and the KPs we need to contact to evaluate those steps.

    Also return the expanded query graph so that we can use it during the
    solving process.
    """

    qgraph = copy.deepcopy(qgraph)

    logger.info("Generating plan for query graph")

    logger.info(
        "Contacting KP registry to ask for KPs to solve this query graph")

    operation_graph = await qg_to_og(qgraph)

    add_descendants(operation_graph, logger)

    await annotate_operation_graph(operation_graph, kp_registry)

    logger.debug({
        "message": "Operation graph with KPs",
        "operation_graph": operation_graph,
    })

    annotate_query_graph(qgraph, operation_graph)

    # Filter down node categories using those
    # we have recieved from the KP registry
    # to limit the number of permutations.
    # filter_categories_predicates(operation_graph)

    # TODO fix this line
    logger.debug({
        "message": "Filtered query graph based on operations",
        "filtered_qgraph": qgraph,
    })

    query_graph_permutations = permute_graph(qgraph, edge_fields=["kps"])

    # Build a list of pinned nodes
    pinned_nodes = [
        key for key, value in qgraph["nodes"].items()
        if value.get("id", None) is not None
    ]

    logger.debug({
        "message": "Pinned nodes to search from",
        "pinned": pinned_nodes,
    })

    plans = []

    logger.info("Searching query graph permutations for plans")

    # TODO replace with generator
    for current_qg in query_graph_permutations:
        valid = fix_categories_predicates(current_qg)
        if not valid:
            continue

        # Create a graph where all nodes and edges
        # from the operation graph are nodes ("reified")
        #
        # This graph is stored as an adjacency list
        reified_graph = {}
        for node_id in current_qg['nodes'].keys():
            reified_graph[node_id] = []
        for edge_id in current_qg['edges'].keys():
            reified_graph[edge_id] = []

        # Fill in adjacencies
        for edge_id, edge in current_qg['edges'].items():
            reverse = edge["kps"]["reverse"]
            if not reverse:
                # Adjacency for forward edge
                reified_graph[edge['subject']].append(edge_id)
                reified_graph[edge_id].append(edge['object'])
            else:
                # Adjacency for reverse edge
                reified_graph[edge['object']].append(edge_id)
                reified_graph[edge_id].append(edge['subject'])

        # Starting at each pinned node, find possible traversals through
        # the operation graph
        possible_traversals = []
        for pinned in pinned_nodes:
            possible_traversals.extend(
                traversals_from_node(reified_graph, pinned)
            )

        possible_traversals = filter(
            lambda t: ensure_traversal_connected(current_qg, t),
            possible_traversals)

        possible_traversals = list(possible_traversals)

        for traversal in possible_traversals:
            # Build our list of steps in the plan
            plan = defaultdict(list)
            for index, edge_id in enumerate(traversal):
                # Skip iteration for non-edges
                if edge_id not in current_qg['edges'].keys():
                    continue

                # We need to know which way to step through the edge
                # so we use the previous value in the traversal
                # which is always the source node
                edge = current_qg['edges'][edge_id]
                source_node_id = traversal[index - 1]

                kp = edge["kps"]
                reverse = kp["reverse"]
                if reverse:
                    step = Step(edge['object'], edge_id, edge['subject'])
                else:
                    step = Step(edge['subject'], edge_id, edge['object'])

                # Attach information about categorys to kp info
                plan[step].append(kp)
            plans.append(plan)

    num_duplicated_plans = len(plans)

    # collapse plans
    unique_map = defaultdict(lambda: defaultdict(list))
    for plan in plans:
        key = tuple(plan.keys())
        for step, kps in plan.items():
            for kp in kps:
                if kp not in unique_map[key][step]:
                    unique_map[key][step].append(kp)
    plans = list(unique_map.values())

    logger.info(
        f"Found {num_duplicated_plans} plans, collapsed down to {len(plans)} plans")

    if len(plans) == 0:
        logger.warning({
            "code": "QueryNotTraversable",
            "message":
            """
                We couldn't find any possible plans starting from a pinned node
                that traverse every edge and node in your query graph
            """
        })

    return plans


def filter_operation_graph_kps(
    operation_graph: dict[str, dict],
):
    """
    Remove KPs that don't match the operation graph's
    node categories or edge predicates.
    """
    for edge in operation_graph["edges"].values():
        edge_operation = {
            "edge_predicate": edge["predicate"],
            "source_category": operation_graph["nodes"][edge["source"]]["category"],
            "target_category": operation_graph["nodes"][edge["target"]]["category"],
        }
        edge["kps"] = {name: kp
                       for name, kp in edge["kps"].items()
                       if edge_operation in kp["operations"]
                       }


def annotate_query_graph(
    query_graph: dict[str, dict],
    operation_graph: dict[str, dict],
):
    """
    Use an annotated operation_graph to annotate
    a query graphs with KPs.
    """
    for edge_id, edge in query_graph["edges"].items():
        edge["kps"] = get_query_graph_edge_kps(
            operation_graph, edge_id)


def get_query_graph_edge_kps(
    operation_graph: dict[str, dict],
    qg_edge_id: str,
) -> list[dict]:
    """
    Get KPs from the operation graph that
    correspond to a query graph edge
    """
    kps = []

    for og_edge in operation_graph["edges"].values():
        if og_edge["qg_edge_id"] != qg_edge_id:
            continue
        for kp in og_edge["kps"]:
            kps.append({
                **kp,
                "reverse": og_edge["qg_traversal_reverse"],
            })
    return kps


# pylint: disable=too-many-arguments
async def step_to_kps(
        source, edge, target,
        kp_registry: Registry,
        allowlist=None, denylist=None,
):
    """Find KP endpoint(s) that enable step."""
    response = await kp_registry.search(
        source['category'],
        edge['predicate'],
        target['category'],
        allowlist=allowlist,
        denylist=denylist,
    )
    # rename node type -> category
    # rename edge type -> predicate
    for kp in response.values():
        for op in kp['operations']:
            op['edge_predicate'] = op.pop('edge_type')
            op['source_category'] = op.pop('source_type')
            op['target_category'] = op.pop('target_type')
    return response
