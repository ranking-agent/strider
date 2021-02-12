"""Query planner."""
from collections import defaultdict, namedtuple
import logging
import copy
from typing import Generator

from reasoner_pydantic import QueryGraph

from strider.kp_registry import Registry
from strider.util import WrappedBMT, last
from strider.config import settings

LOGGER = logging.getLogger(__name__)

Step = namedtuple("Step", ["source", "edge", "target"])
Operation = namedtuple(
    "Operation", ["source_category", "edge_predicate", "target_category"])

REVERSE_EDGE_SUFFIX = '.reverse'


def find_next_list_property(search_dict, fields_to_check):
    """ Find first object in a dictionary where object[field] is a list """
    for key, val in search_dict.items():
        for field in fields_to_check:
            if field in val and isinstance(val[field], list):
                return key, field
    return None, None


def permute_og(operation_graph: dict) -> Generator[dict, None, None]:
    """
    Take in a query graph that has some unbound properties
    and return a list of query graphs where every property is bound

    Example: If a query graph has ['Disease', 'Gene'] as a type for a node,
    two query graphs will be returned, one with node type Disease and one with type Gene
    """

    stack = []
    stack.append(copy.deepcopy(operation_graph))

    while len(stack) > 0:
        current_qg = stack.pop()

        # Any fields on a node that might need to be permuted
        node_fields_to_permute = ['category', 'id']

        # Find our next node to permute
        next_node_id, next_node_field = \
            find_next_list_property(
                current_qg['nodes'], node_fields_to_permute)

        if next_node_id:
            # Permute this node and push permutations to stack
            next_node = current_qg['nodes'][next_node_id]
            for field_value in next_node[next_node_field]:
                permutation_copy = copy.deepcopy(current_qg)
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
                current_qg['edges'], edge_fields_to_permute)

        if next_edge_id:
            # Permute this edge and push permutations to stack
            next_edge = current_qg['edges'][next_edge_id]
            for field_value in next_edge[next_edge_field]:
                permutation_copy = copy.deepcopy(current_qg)
                # Fix predicate
                permutation_copy['edges'][next_edge_id][next_edge_field] = field_value
                # Add to stack
                stack.append(permutation_copy)
            continue

        # This QG is fully permuted, add to results
        yield current_qg


class NoAnswersError(Exception):
    """No answers can be found."""


def get_operation(
        operation_graph: dict,
        edge: dict,
) -> Operation:
    """ Get the types from an edge in the query graph """
    return Operation(
        operation_graph['nodes'][edge['source']]['category'],
        edge['predicate'],
        operation_graph['nodes'][edge['target']]['category'],
    )


async def get_operation_kp_map(
        operation_graph,
        kp_registry: Registry = None,
) -> dict[Operation, list[dict]]:
    """Put the results in a master dictionary so that we can look
    them up for each permutation and see which are solvable
    """
    if kp_registry is None:
        kp_registry = Registry(settings.kpregistry_url)

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


async def qg_to_og(
        qgraph,
        logger: logging.Logger = logging.getLogger(),
        reverse: bool = True,
):
    """Convert query graph to operation graph."""
    ograph = {
        "nodes": copy.deepcopy(qgraph["nodes"]),
        "edges": dict(),
    }
    # Add reverse edges so that we can traverse edges in both directions
    for edge_id, edge in qgraph["edges"].items():
        ograph["edges"][edge_id] = {
            "source": edge["subject"],
            "predicate": [f"-{p}->" for p in edge["predicate"]],
            "target": edge["object"],
        }
        if reverse:
            ograph["edges"][edge_id + REVERSE_EDGE_SUFFIX] = {
                "source": edge["object"],
                "predicate": [f"<-{p}-" for p in edge["predicate"]],
                "target": edge["subject"],
            }

    return ograph


async def filter_categories_predicates(operation_graph, operation_kp_map):
    """Filter out categories and predicates that cannot be found using KPs."""
    for node in operation_graph['nodes'].values():
        node['filtered_categories'] = set()
    for edge in operation_graph['edges'].values():
        edge['filtered_predicates'] = set()

    for edge in operation_graph['edges'].values():

        source = operation_graph['nodes'][edge['source']]
        target = operation_graph['nodes'][edge['target']]

        for op in operation_kp_map:
            if (
                    op.source_category in source["category"]
                    and op.target_category in target["category"]
                    and op.edge_predicate in edge["predicate"]
            ):
                source['filtered_categories'].add(op.source_category)
                target['filtered_categories'].add(op.target_category)
                edge['filtered_predicates'].add(op.edge_predicate)

    # Filter down node categories using those
    # we have recieved from the KP registry
    # to limit the number of permutations.

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
    return None


# pylint: disable=too-many-locals
async def find_valid_permutations(
        operation_graph,
        kp_registry: Registry = None,
        logger: logging.Logger = LOGGER,
) -> list[dict]:
    """
    Given an operation graph, generate a list of operation graphs
    that are solvable, and annotate those with a 'request_kp' property
    that shows which KPs to contact to get results.
    """

    # For each edge, ask KP registry for KPs that could solve it

    logger.debug(
        "Contacting KP registry to ask for KPs to solve this query graph")

    operation_kp_map = await get_operation_kp_map(operation_graph, kp_registry)

    await filter_categories_predicates(operation_graph, operation_kp_map)

    # Remove edges with no predicates
    operation_graph['edges'] = {k: edge for k, edge in operation_graph['edges'].items()
                                if len(edge['predicate']) > 0}

    logger.debug(
        f"Found {len(operation_kp_map)} possible KP operations we can use")

    logger.debug(
        "Iterating over QGs to find ones that are solvable")

    permuted_og_list = permute_og(operation_graph)

    filtered_ogs = validate_and_annotate_og_list(
        permuted_og_list,
        operation_kp_map,
    )

    return list(filtered_ogs)


def all_eulerian_paths(operation_graph, source) -> list[list[str]]:
    """
    Return all Eulerian paths originating from a source node.

    An Eulerian path is one that visits all nodes exactly once
    """
    paths = []

    starting_edges = [
        edge_id for edge_id, edge in operation_graph['edges'].items()
        if edge['source'] == source
    ]

    # Initialize stack with all edges that start at our source node
    stack = []
    for edge in starting_edges:
        new_graph = copy.deepcopy(operation_graph)
        del new_graph['edges'][edge]
        stack.append((new_graph, [edge]))

    # Recursively finish paths
    while len(stack) != 0:
        s = stack.pop()

        current_graph = s[0]
        current_path = s[1]

        current_node = operation_graph['edges'][current_path[-1]]['target']

        adjacent_edges = [
            edge_id for edge_id, edge in current_graph['edges'].items()
            if edge['source'] == current_node
        ]

        if len(adjacent_edges) == 0:
            # Done
            paths.append(current_path)
            continue

        for edge in adjacent_edges:
            new_graph = copy.deepcopy(current_graph)
            new_path = current_path.copy()
            # remove edge from graph and add it to path
            new_path.append(edge)
            del new_graph['edges'][edge]
            stack.append((new_graph, new_path))

    return paths


def validate_traversal(operation_graph, path_edges):
    """
    Validate a traversal.

    Traversals must:
    1. Visit every node
    2. Visit every QUERY graph edge exactly once
    """
    all_nodes = operation_graph['nodes'].keys()
    pinned_nodes = [
        key for key, value in operation_graph["nodes"].items()
        if value.get("id", None) is not None
    ]

    path_nodes = set()
    for edge in path_edges:
        path_nodes.add(operation_graph['edges'][edge]['source'])
        path_nodes.add(operation_graph['edges'][edge]['target'])

    # path_nodes + pinned_nodes must cover all nodes in the graph
    if set(pinned_nodes) | set(path_nodes) != all_nodes:
        return False

    def get_undirected_edges(edges):
        return [edge_id.replace(REVERSE_EDGE_SUFFIX, '') for edge_id in edges]

    graph_edges_undirected = set(
        get_undirected_edges(
            operation_graph["edges"].keys()
        )
    )
    path_edges_undirected = get_undirected_edges(path_edges)

    # Check that we traverse each query graph edge exactly once
    for edge in graph_edges_undirected:
        if path_edges_undirected.count(edge) != 1:
            return False
    return True


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

    logger.debug("Generating plan for query graph")

    operation_graph = await qg_to_og(qgraph, logger)
    filtered_og_list = await find_valid_permutations(
        operation_graph, kp_registry, logger
    )

    # Build a list of pinned nodes
    pinned_nodes = [
        key for key, value in qgraph["nodes"].items()
        if value.get("id", None) is not None
    ]

    if len(filtered_og_list) == 0:
        logger.warning({
            "code": "QueryNotTraversable",
            "message":
                """
                    We couldn't find a KP for every edge in your query graph
                """
        })
        return []

    plans = []

    for current_og in filtered_og_list:
        possible_traversals = []

        # Starting at each pinned node, construct a plan
        for pinned in pinned_nodes:
            eulerian_paths = all_eulerian_paths(current_og, pinned)
            # A valid traversal might not include every edge on the path
            # so we also add subsets of paths as possible traversals
            for path in eulerian_paths:
                for i in range(len(path)):
                    possible_traversals.append(last(path, i))

        # Validate each traversal
        valid_traversals = [
            path for path in possible_traversals
            if validate_traversal(operation_graph, path)
        ]

        for path in valid_traversals:
            # Build our list of steps in the plan
            # information to the original query graph
            plan = defaultdict(list)
            for edge_id in path:
                edge = current_og['edges'][edge_id]
                # Find edges that get us from
                # There could be multiple edges that need to each
                # be added as a separate step
                op = get_operation(current_og, edge)

                # Reverse edges don't actually exist, so the steps in the plan
                # should just refer to the forward edges
                if edge_id.endswith(REVERSE_EDGE_SUFFIX):
                    edge_id = edge_id.replace(REVERSE_EDGE_SUFFIX, '')

                step = Step(edge['source'], edge_id, edge['target'])
                # Attach information about categorys to kp info
                for kp in edge['request_kps']:
                    plan[step].append({
                        **op._asdict(),
                        **kp,
                    })
            plans.append(plan)

    # collapse plans
    unique_map = defaultdict(lambda: defaultdict(list))
    for plan in plans:
        key = tuple(plan.keys())
        for step, kps in plan.items():
            unique_map[key][step].extend(kps)
    plans = list(unique_map.values())

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


def validate_and_annotate_og_list(
    og_list: Generator[dict, None, None],
    operation_kp_map: dict[Operation, list[dict]],
) -> Generator[dict, None, None]:
    """
    Check if operation graph has a valid plan, and if it does,
    annotate with KP name
    """
    for og in og_list:
        valid = True
        for edge in og['edges'].values():
            op = get_operation(og, edge)
            kps_available = operation_kp_map.get(op, None)
            if not kps_available:
                valid = False
            edge['request_kps'] = kps_available
        if valid:
            yield og


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
