"""Query planner."""
from collections import defaultdict, namedtuple
import logging
import copy
from typing import Generator, Callable

from reasoner_pydantic import QueryGraph

from strider.kp_registry import Registry
from strider.util import WrappedBMT
from strider.config import settings

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


def permute_graph(graph: dict) -> Generator[dict, None, None]:
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

        # Any fields on a node that might need to be permuted
        node_fields_to_permute = ['category', 'id']

        # Find our next node to permute
        next_node_id, next_node_field = \
            find_next_list_property(
                current_graph['nodes'], node_fields_to_permute)

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

        # Any fields on an edge that might need to be permuted
        edge_fields_to_permute = ['predicate']

        # Find our next edge to permute
        next_edge_id, next_edge_field = \
            find_next_list_property(
                current_graph['edges'], edge_fields_to_permute)

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


def get_operation(
        query_graph: dict,
        edge: dict,
        reverse: bool = False,
) -> Operation:
    """ Get the types from an edge in the query graph """

    if reverse:
        return Operation(
            query_graph['nodes'][edge['object']]['category'],
            f"<-{edge['predicate']}-",
            query_graph['nodes'][edge['subject']]['category'],
        )
    else:
        return Operation(
            query_graph['nodes'][edge['subject']]['category'],
            f"-{edge['predicate']}->",
            query_graph['nodes'][edge['object']]['category'],
        )


async def get_operation_kp_map(
        query_graph,
        kp_registry: Registry = None,
) -> dict[Operation, list[dict]]:
    """
    Look up kps for each operation in a query graph

    Put the results in a master dictionary
    """
    if kp_registry is None:
        kp_registry = Registry(settings.kpregistry_url)

    operation_graph = await qg_to_og(query_graph)

    operation_kp_map = {}

    for edge in operation_graph['edges'].values():
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

        for kp_name, kp in kp_results.items():

            kp_without_ops = {x: kp[x] for x in kp if x != 'operations'}
            kp_without_ops['name'] = kp_name

            for op in kp['operations']:
                operation = Operation(**op)
                if operation not in operation_kp_map:
                    operation_kp_map[operation] = []
                operation_kp_map[operation].append(kp_without_ops)
    return operation_kp_map


def make_og_edge(
        qg_edge: dict,
        edge_reverse: bool,
        predicate_reverse: bool,
        predicates: list = None,
):
    og_edge = {}

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


async def qg_to_og(
        qgraph,
        logger: logging.Logger = logging.getLogger(),
        reverse: bool = True,
):
    """Convert query graph to operation graph"""
    ograph = {
        "nodes": copy.deepcopy(qgraph["nodes"]),
        "edges": dict(),
    }
    # Add reverse edges so that we can traverse edges in both directions
    for edge_id, edge in qgraph["edges"].items():
        # Forward edge
        ograph["edges"][edge_id] = make_og_edge(
            edge,
            edge_reverse=False,
            predicate_reverse=False
        )
        # Reverse edge
        ograph["edges"][edge_id + REVERSE_EDGE_SUFFIX] = make_og_edge(
            edge,
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
                edge,
                edge_reverse=False,
                predicate_reverse=True,
                predicates=symmetric_predicates)
            # Reverse symmetric edge
            ograph["edges"][edge_id + REVERSE_EDGE_SUFFIX + SYMMETRIC_EDGE_SUFFIX] = \
                make_og_edge(
                edge,
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
                edge,
                edge_reverse=True,
                predicate_reverse=False,
                predicates=inverse_predicates)
            # Reverse inverse edge
            ograph["edges"][edge_id + REVERSE_EDGE_SUFFIX + INVERSE_EDGE_SUFFIX] = \
                make_og_edge(
                edge,
                edge_reverse=False,
                predicate_reverse=True,
                predicates=inverse_predicates)

    return ograph


async def filter_categories_predicates(graph, operation_kp_map):
    """Filter out categories and predicates that cannot be found using KPs."""
    for node in graph['nodes'].values():
        node['filtered_categories'] = set()
    for edge in graph['edges'].values():
        edge['filtered_predicates'] = set()

    for edge in graph['edges'].values():
        sub = graph['nodes'][edge['subject']]
        obj = graph['nodes'][edge['object']]

        for op in operation_kp_map:
            # Check if forward operation matches
            if (
                    op.source_category in sub["category"]
                    and op.target_category in obj["category"]
                    and op.edge_predicate in [f"-{p}->" for p in edge["predicate"]]
            ):
                sub['filtered_categories'].add(op.source_category)
                obj['filtered_categories'].add(op.target_category)
                edge['filtered_predicates'].add(op.edge_predicate[1:-2])
            # Reverse
            if (
                    op.source_category in obj["category"]
                    and op.target_category in sub["category"]
                    and op.edge_predicate in [f"<-{p}-" for p in edge["predicate"]]
            ):
                obj['filtered_categories'].add(op.source_category)
                sub['filtered_categories'].add(op.target_category)
                edge['filtered_predicates'].add(op.edge_predicate[2:-1])

    # We only filter nodes that are touching at least one edge ("connected")
    connected_nodes = set()
    for edge in graph['edges'].values():
        connected_nodes.add(edge['subject'])
        connected_nodes.add(edge['object'])
    for node_id in connected_nodes:
        node = graph['nodes'][node_id]
        if 'category' not in node:
            continue
        node['category'] = list(node.pop('filtered_categories'))

    # Also filter edge predicates as well
    for edge in graph['edges'].values():
        edge['predicate'] = list(edge.pop('filtered_predicates'))
    return None


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

    logger.info("Generating plan for query graph")

    logger.info(
        "Contacting KP registry to ask for KPs to solve this query graph")

    operation_kp_map = await get_operation_kp_map(qgraph, kp_registry)

    logger.info(
        f"Found {len(operation_kp_map)} possible KP operations we can use")

    logger.debug({
        "message": "KP operations from the query graph",
        "operation_kp_map": operation_kp_map,
    })

    # Filter down node categories using those
    # we have recieved from the KP registry
    # to limit the number of permutations.
    await filter_categories_predicates(qgraph, operation_kp_map)

    logger.debug({
        "message": "Filtered query graph based on operations",
        "filtered_qgraph": qgraph,
    })

    query_graph_permutations = permute_graph(qgraph)

    annotated_query_graph_permutations = \
        annotate_qg_list_with_kps(query_graph_permutations, operation_kp_map)

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

    for current_qg in annotated_query_graph_permutations:

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
            if len(edge['forward_kps']) > 0:
                # Adjacency for forward edge
                reified_graph[edge['subject']].append(edge_id)
                reified_graph[edge_id].append(edge['object'])
            if len(edge['reverse_kps']) > 0:
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
                reverse = source_node_id == edge['object']

                if reverse:
                    step = Step(edge['object'], edge_id, edge['subject'])
                else:
                    step = Step(edge['subject'], edge_id, edge['object'])
                op = get_operation(current_qg, edge, reverse=reverse)

                # Attach information about categorys to kp info
                for kp in edge['forward_kps'] + edge['reverse_kps']:
                    plan[step].append({
                        **op._asdict(),
                        **kp,
                    })
            plans.append(plan)

    num_duplicated_plans = len(plans)

    # collapse plans
    unique_map = defaultdict(lambda: defaultdict(list))
    for plan in plans:
        key = tuple(plan.keys())
        for step, kps in plan.items():
            unique_map[key][step].extend(kps)
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


def annotate_qg_list_with_kps(
    qg_list: Generator[dict, None, None],
    operation_kp_map: dict[Operation, list[dict]],
) -> Generator[dict, None, None]:
    """
    Use an operation KP map to annotate graph edges
    with KPs. Only yields graphs where every edge
    has either a forward or backward KP available.
    """
    for qg in qg_list:
        valid = True
        for edge in qg['edges'].values():
            forward_op = get_operation(qg, edge)
            edge['forward_kps'] = operation_kp_map.get(forward_op, [])
            reverse_op = get_operation(qg, edge, reverse=True)
            edge['reverse_kps'] = operation_kp_map.get(reverse_op, [])

            if len(edge['forward_kps']) == 0 and len(edge['reverse_kps']) == 0:
                valid = False
        if valid:
            yield qg


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
