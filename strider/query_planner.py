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


def find_next_list_property(search_dict, fields_to_check):
    """ Find first object in a dictionary where object[field] is a list """
    for key, val in search_dict.items():
        for field in fields_to_check:
            if field in val and isinstance(val[field], list):
                return key, field
    return None, None


def permute_qg(query_graph: QueryGraph) -> Generator[QueryGraph, None, None]:
    """
    Take in a query graph that has some unbound properties
    and return a list of query graphs where every property is bound

    Example: If a query graph has ['Disease', 'Gene'] as a type for a node,
    two query graphs will be returned, one with node type Disease and one with type Gene
    """

    stack = []
    stack.append(copy.deepcopy(query_graph))

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
        query_graph: QueryGraph,
        edge: dict,
) -> Operation:
    """ Get the types from an edge in the query graph """
    return Operation(
        query_graph['nodes'][edge['subject']]['category'],
        edge['predicate'],
        query_graph['nodes'][edge['object']]['category'],
    )


def expand_qg(
        qg: QueryGraph,
        logger: logging.Logger = LOGGER
) -> QueryGraph:
    """
    Given a query graph, use the Biolink model to expand categories and predicates
    to include their descendants.
    """

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

        # For each edge predicate, we also need versions
        # for -predicate-> and <-predicate-
        # ltr = left to right
        new_predicate_list_ltr = \
            [f'-{p}->' for p in new_predicate_list]
        new_predicate_list_rtl = \
            [f'<-{p}-' for p in new_predicate_list]

        edge['predicate'] = new_predicate_list_ltr + new_predicate_list_rtl

    logger.debug({
        "description": "Expanded query graph with descendants",
        "qg": qg,
    })

    return qg


# pylint: disable=too-many-locals
async def find_valid_permutations(
        qgraph: QueryGraph,
        kp_registry: Registry = None,
        normalizer: Normalizer = None,
        logger: logging.Logger = LOGGER,
) -> list[QueryGraph]:
    """
    Given a query graph, generate a list of query graphs
    that are solvable, and annotate those with a 'request_kp' property
    that shows which KPs to contact to get results
    """
    if kp_registry is None:
        kp_registry = Registry(settings.kpregistry_url)
    if normalizer is None:
        normalizer = Normalizer(settings.normalizer_url)

    expanded_qg = expand_qg(qgraph, logger)

    logger.debug("Contacting node normalizer to get categorys for curies")

    # Use node normalizer to add
    # a category to nodes with a curie
    for node in expanded_qg['nodes'].values():
        if not node.get('id'):
            continue
        if not isinstance(node['id'], list):
            node['id'] = [node['id']]

        # Get full list of categorys
        categories = await normalizer.get_types(node['id'])

        # Filter categorys that are ancestors of other categorys we were given
        node['category'] = filter_ancestor_types(categories)

    logger.debug({
        "description": "Query graph with categorys added to curies",
        "expanded_qg": expanded_qg,
    })

    operation_kp_map = {}
    # For each edge, ask KP registry for KPs that could solve it

    logger.debug(
        "Contacting KP registry to ask for KPs to solve this query graph")

    for node in expanded_qg['nodes'].values():
        node['filtered_categories'] = set()
    for edge in expanded_qg['edges'].values():
        edge['filtered_predicates'] = set()

    # Put the results in a master dictionary so that we can look
    # them up for each permutation and see which are solvable
    for edge in expanded_qg['edges'].values():
        if "provided_by" in edge:
            allowlist = edge["provided_by"].get("allowlist", None)
            denylist = edge["provided_by"].get("denylist", None)
        else:
            allowlist = denylist = None

        source = expanded_qg['nodes'][edge['subject']]
        target = expanded_qg['nodes'][edge['object']]

        kp_results = await step_to_kps(
            source, edge, target,
            kp_registry,
            allowlist=allowlist, denylist=denylist,
        )

        for kp in kp_results.values():
            for op in kp['operations']:
                source['filtered_categories'].add(
                    op['source_category'])
                target['filtered_categories'].add(
                    op['target_category'])
                edge['filtered_predicates'].add(op['edge_predicate'])

        for kp_name, kp in kp_results.items():

            kp_without_ops = {x: kp[x] for x in kp if x != 'operations'}
            kp_without_ops['name'] = kp_name

            for op in kp['operations']:
                operation = Operation(**op)
                if operation not in operation_kp_map:
                    operation_kp_map[operation] = []
                operation_kp_map[operation].append(kp_without_ops)

    # Filter down node categories using those
    # we have recieved from the KP registry
    # to limit the number of permutations.

    # We only filter nodes that are touching at least one edge ("connected")
    connected_nodes = []
    for edge in expanded_qg['edges'].values():
        connected_nodes.append(edge['subject'])
        connected_nodes.append(edge['object'])
    for node in connected_nodes:
        if 'category' not in node:
            continue
        node['category'] = list(node.pop('filtered_categories'))

    # Also filter edge predicates as well
    for edge in expanded_qg['edges'].values():
        edge['predicate'] = list(edge.pop('filtered_predicates'))

    logger.debug(
        f"Found {len(operation_kp_map)} possible KP operations we can use")

    logger.debug(
        "Iterating over QGs to find ones that are solvable")

    permuted_qg_list = permute_qg(expanded_qg)

    filtered_qgs = validate_and_annotate_qg_list(
        permuted_qg_list,
        operation_kp_map,
    )

    return list(filtered_qgs)


def dfs(graph, source) -> (list[str], list[str]):
    """
    Run depth first search on the graph

    Returns a list of nodes and edges
    that we traversed.
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

        for edge_id, edge in graph['edges'].items():
            if edge['subject'] == s:
                adjacent_nodes.append(edge['object'])
                adjacent_edges.append(edge_id)

        for node_id, edge_id in zip(adjacent_nodes, adjacent_edges):
            stack.append(node_id)
            path_nodes.append(node_id)
            path_edges.append(edge_id)

    return path_nodes, path_edges


async def generate_plans(
        qgraph: QueryGraph,
        kp_registry: Registry = None,
        normalizer: Normalizer = None,
        logger: logging.Logger = LOGGER,
) -> list[dict[Step, list]]:
    """
    Given a query graph, build plans that consists of steps
    and the KPs we need to contact to evaluate those steps
    """

    logger.debug("Generating plan for query graph")

    filtered_qg_list = await find_valid_permutations(
        qgraph, kp_registry, normalizer, logger
    )

    # Build a list of pinned nodes
    pinned_nodes = [
        key for key, value in qgraph["nodes"].items()
        if value.get("id", None) is not None
    ]

    if len(filtered_qg_list) == 0:
        logger.error({
            "code": "QueryNotTraversable",
            "message":
                """
                    We couldn't find a KP for every edge in your query graph
                """
        })
        return []

    plans = []

    for current_qg in filtered_qg_list:
        # Starting at each pinned node, construct a plan
        for pinned in pinned_nodes:
            path_nodes, path_edges = dfs(current_qg, pinned)
            all_nodes = current_qg['nodes'].keys()
            bound_nodes = [
                nid for nid, node in current_qg['nodes'].items() if 'id' in node]

            # path_nodes + bound_nodes must cover all nodes in the graph
            if set(bound_nodes) | set(path_nodes) != all_nodes:
                continue

            # If we don't traverse every edge we can't use this
            if set(path_edges) != set(current_qg['edges'].keys()):
                continue

            # Build our list of steps in the plan
            # information to the original query graph
            plan = defaultdict(list)
            for edge_id in path_edges:
                edge = current_qg['edges'][edge_id]
                # Find edges that get us from
                # There could be multiple edges that need to each
                # be added as a separate step
                step = Step(edge['subject'], edge_id, edge['object'])
                op = get_operation(current_qg, edge)
                # Attach information about categorys to kp info
                for kp in edge['request_kps']:
                    plan[step].append({
                        **op._asdict(),
                        **kp,
                    })
            plans.append(plan)

    if len(plans) == 0:
        logger.error({
            "code": "QueryNotTraversable",
            "message":
            """
                We couldn't find any possible plans starting from a pinned node
                that traverse every edge and node in your query graph
            """
        })

    return plans


def validate_and_annotate_qg_list(
    qg_list: Generator[QueryGraph, None, None],
    operation_kp_map: dict[Operation, list[dict]],
) -> Generator[QueryGraph, None, None]:
    """
    Check if QG has a valid plan, and if it does,
    annotate with KP name
    """
    for qg in qg_list:
        valid = True
        for edge in qg['edges'].values():
            op = get_operation(qg, edge)
            kps_available = operation_kp_map.get(op, None)
            if not kps_available:
                valid = False
            edge['request_kps'] = kps_available
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
