"""Query planner."""
from collections import defaultdict, namedtuple
import logging
import copy
from typing import Generator

from reasoner_pydantic import QueryGraph

from strider.kp_registry import Registry
from strider.normalizer import Normalizer
from strider.util import WrappedBMT
from strider.config import settings

WBMT = WrappedBMT()

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


async def expand_qg(
        qg: QueryGraph,
        logger: logging.Logger = LOGGER,
        normalizer: Normalizer = None,
) -> QueryGraph:
    """
    Given a query graph, use the Biolink model to expand categories and predicates
    to include their descendants. Also get categories for pinned nodes.
    """
    if normalizer is None:
        normalizer = Normalizer(settings.normalizer_url)

    logger.debug("Using BMT to get descendants of node and edge types")

    qg = copy.deepcopy(qg)
    # Use BMT to convert node categorys to categorys + descendants
    for node in qg['nodes'].values():
        if 'category' not in node:
            continue
        if not isinstance(node['category'], list):
            node['category'] = [node['category']]
        new_category_list = []
        for t in node['category']:
            new_category_list.extend(WBMT.get_descendants(t))
        node['category'] = new_category_list

    # Same with edges
    for edge in qg['edges'].values():
        if 'predicate' not in edge:
            continue
        if not isinstance(edge['predicate'], list):
            edge['predicate'] = [edge['predicate']]
        new_predicate_list = []
        for t in edge['predicate']:
            new_predicate_list.extend(WBMT.get_descendants(t))
        edge['predicate'] = new_predicate_list

    logger.debug({
        "description": "Expanded query graph with descendants",
        "qg": qg,
    })

    logger.debug("Contacting node normalizer to get categorys for curies")

    # Use node normalizer to add
    # a category to nodes with a curie
    for node in qg['nodes'].values():
        if not node.get('id'):
            continue
        if not isinstance(node['id'], list):
            node['id'] = [node['id']]

        # Get full list of categorys
        categories = await normalizer.get_types(node['id'])

        if categories:
            # Filter categorys that are ancestors of other categorys we were given
            node['category'] = filter_ancestor_types(categories)
        elif "category" not in node:
            node["category"] = []

    return qg


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

    logger.debug({
        "description": "Query graph with categorys added to curies",
        "expanded_qg": operation_graph,
    })

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


def dfs(operation_graph, source) -> (list[str], list[str]):
    """
    Run depth first search on the graph

    Returns a list of nodes and edges
    that we traversed. Will not go back to
    previously visited nodes or edges.
    """
    path_nodes = [source]
    path_edges = []

    stack = [source]

    # The most efficient way to do a depth first search
    # is with an adjacency list, so we build one here

    while len(stack) != 0:
        s = stack.pop()

        # Build adjacency list
        adjacent_nodes = []
        adjacent_edges = []

        for edge_id, edge in operation_graph['edges'].items():
            if edge['source'] == s:
                adjacent_nodes.append(edge['target'])
                adjacent_edges.append(edge_id)

        for node_id, edge_id in zip(adjacent_nodes, adjacent_edges):
            if edge_id in path_edges:
                continue
            if node_id in path_nodes:
                continue
            stack.append(node_id)
            path_nodes.append(node_id)
            path_edges.append(edge_id)

    return path_nodes, path_edges


async def generate_plans(
        qgraph: QueryGraph,
        kp_registry: Registry = None,
        normalizer: Normalizer = None,
        logger: logging.Logger = LOGGER,
) -> (list[dict[Step, list]], dict[str, dict]):
    """
    Given a query graph, build plans that consists of steps
    and the KPs we need to contact to evaluate those steps.

    Also return the expanded query graph so that we can use it during the
    solving process.
    """

    logger.debug("Generating plan for query graph")

    expanded_qg = await expand_qg(qgraph, logger, normalizer)
    operation_graph = await qg_to_og(expanded_qg, logger)
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
        return [], {}

    plans = []

    for current_og in filtered_og_list:
        # Starting at each pinned node, construct a plan
        for pinned in pinned_nodes:
            path_nodes, path_edges = dfs(current_og, pinned)

            all_nodes = current_og['nodes'].keys()
            # path_nodes + pinned_nodes must cover all nodes in the graph
            if set(pinned_nodes) | set(path_nodes) != all_nodes:
                continue

            # If we don't traverse every edge either in the forward
            # or reverse direction we can't use this
            path_edges_with_reverse = set()
            for edge_id in path_edges:
                # Remove suffix
                edge_id = edge_id.replace(REVERSE_EDGE_SUFFIX, '')
                path_edges_with_reverse.add(edge_id)
                path_edges_with_reverse.add(edge_id + REVERSE_EDGE_SUFFIX)
            if not all(
                    graph_edge in path_edges_with_reverse
                    for graph_edge in set(current_og['edges'].keys())
            ):
                continue

            # Build our list of steps in the plan
            # information to the original query graph
            plan = defaultdict(list)
            for edge_id in path_edges:
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

    return plans, expanded_qg


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
