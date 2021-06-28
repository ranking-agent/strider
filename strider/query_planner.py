"""Query planner."""
from collections import defaultdict, namedtuple
import logging
import copy
from typing import Generator

from reasoner_pydantic import QueryGraph

from strider.kp_registry import Registry
from strider.util import WBMT
from strider.config import settings
from strider.traversal import get_traversals, NoAnswersError

LOGGER = logging.getLogger(__name__)

Step = namedtuple("Step", ["source", "edge", "target"])
Operation = namedtuple(
    "Operation", ["source_category", "edge_predicate", "target_category"])

REVERSE_EDGE_SUFFIX = '.reverse'
SYMMETRIC_EDGE_SUFFIX = '.symmetric'
INVERSE_EDGE_SUFFIX = '.inverse'


def find_next_list_property(search_dict, fields_to_check):
    """ Find first object in a dictionary where object[field] is a list """
    for key, val in search_dict.items():
        for field in fields_to_check:
            if field in val and isinstance(val[field], list):
                return key, field
    return None, None


def permute_graph(
        graph: dict,
        node_field_map: dict[str] = {},
        edge_field_map: dict[str] = {},
) -> Generator[dict, None, None]:
    """
    Take in a graph that has some unbound properties
    and return a list of query graphs where every property is bound.

    Parameters are dictionaries that are the mapping from the
    field on an unbound graph to a bound graph. 
    For example { "kps" : "kp" } will map the kps field to a field called kp.

    Example: If a graph has ['Disease', 'Gene'] as a type for a node,
    two query graphs will be returned, one with node type Disease and one with type Gene
    """

    # pylint: disable=dangerous-default-value
    # It is okay to disable this rule in this function
    # because we don't modify the default value

    stack = []
    stack.append(copy.deepcopy(graph))

    while len(stack) > 0:
        current_graph = stack.pop()

        # Find our next node to permute
        next_node_id, next_node_field = \
            find_next_list_property(
                current_graph['nodes'], node_field_map.keys())

        if next_node_id:
            # Permute this node and push permutations to stack
            next_node = current_graph['nodes'][next_node_id]

            # Pull field values and iterate over them
            field_value_list = next_node[next_node_field]
            for field_value in field_value_list:
                permutation_copy = current_graph.copy()
                permutation_copy["nodes"] = permutation_copy["nodes"].copy()
                permutation_copy["nodes"][next_node_id] = \
                    permutation_copy["nodes"][next_node_id].copy()

                # Remove old field
                permutation_copy["nodes"][next_node_id].pop(next_node_field)
                # Set mapped field
                permutation_copy["nodes"][next_node_id][
                    node_field_map[next_node_field]
                ] = field_value

                # Add to stack
                stack.append(permutation_copy)
            continue

        # Find our next edge to permute
        next_edge_id, next_edge_field = \
            find_next_list_property(
                current_graph['edges'], edge_field_map.keys())

        if next_edge_id:
            # Permute this edge and push permutations to stack
            next_edge = current_graph['edges'][next_edge_id]

            # Pull field value and iterate over list
            field_value_list = next_edge[next_edge_field]
            for field_value in field_value_list:
                permutation_copy = current_graph.copy()
                permutation_copy["edges"] = permutation_copy["edges"].copy()
                permutation_copy["edges"][next_edge_id] = \
                    permutation_copy["edges"][next_edge_id].copy()

                # Remove old field
                permutation_copy["edges"][next_edge_id].pop(next_edge_field)
                # Set mapped field
                permutation_copy['edges'][next_edge_id][
                    edge_field_map[next_edge_field]
                ] = field_value

                # Add to stack
                stack.append(permutation_copy)
            continue

        # This graph is fully permuted, add to results
        yield current_graph


def count_permutations(
        graph: dict,
        node_fields: list[str] = [],
        edge_fields: list[str] = [],
):
    """
    Get the number of permutations that will be generated
    from the permute_graph method.
    """

    # pylint: disable=dangerous-default-value
    # It is okay to disable this rule in this function
    # because we don't modify the default value

    total_permutations = 1
    for node in graph["nodes"].values():
        for field in node_fields:
            if field in node and isinstance(node[field], list):
                total_permutations *= len(node[field])
    for edge in graph["edges"].values():
        for field in edge_fields:
            if field in edge and isinstance(edge[field], list):
                total_permutations *= len(edge[field])
    return total_permutations


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

    # Annotate with whether we have switched
    # the source and target
    og_edge["edge_reverse"] = edge_reverse

    if edge_reverse:
        og_edge["source"] = qg_edge["object"]
        og_edge["target"] = qg_edge["subject"]
    else:
        og_edge["source"] = qg_edge["subject"]
        og_edge["target"] = qg_edge["object"]

    # Default value
    if not predicates:
        predicates = qg_edge["predicates"]

    if predicate_reverse:
        og_edge["predicates"] = [f"<-{p}-" for p in predicates]
    else:
        og_edge["predicates"] = [f"-{p}->" for p in predicates]
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
            p for p in edge["predicates"]
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
            WBMT.predicate_inverse(p) for p in edge["predicates"]
        ]
        inverse_predicates = list(filter(None, inverse_predicates))
        if len(inverse_predicates) > 0:
            # Forward inverse edge
            ograph["edges"][edge_id + INVERSE_EDGE_SUFFIX] = make_og_edge(
                edge_id, edge,
                edge_reverse=False,
                predicate_reverse=True,
                predicates=inverse_predicates)
            # Reverse inverse edge
            ograph["edges"][edge_id + REVERSE_EDGE_SUFFIX + INVERSE_EDGE_SUFFIX] = \
                make_og_edge(
                edge_id, edge,
                edge_reverse=True,
                predicate_reverse=False,
                predicates=inverse_predicates
            )

    return ograph


def fix_categories_predicates(query_graph):
    """
    Given a permuted query graph with one KP,
    fix the node categories and predicates to match the KP

    Raises ValueError if there is a conflicting node type
    """

    # Clear existing categories and predicates
    for node in query_graph["nodes"].values():
        node.pop("category", None)
    for edge in query_graph["edges"].values():
        edge.pop("predicate", None)

    for edge in query_graph['edges'].values():
        kp = edge["kp"]
        reverse = kp["reverse"]

        if reverse:
            source_node = query_graph["nodes"][edge["object"]]
            target_node = query_graph["nodes"][edge["subject"]]
        else:
            source_node = query_graph["nodes"][edge["subject"]]
            target_node = query_graph["nodes"][edge["object"]]

        if "category" in source_node \
                and source_node["category"] != kp["source_category"]:
            raise ValueError()
        source_node["category"] = kp["source_category"]

        if "category" in target_node \
                and target_node["category"] != kp["target_category"]:
            raise ValueError()
        target_node["category"] = kp["target_category"]

        if "predicate" in edge \
                and edge["predicate"] != kp["edge_predicate"]:
            raise ValueError()
        edge["predicate"] = kp["edge_predicate"]


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
    traversals = get_traversals(qgraph)

    if kp_registry is None:
        kp_registry = Registry(settings.kpregistry_url)

    kps = dict()
    for qedge_id in qgraph["edges"]:
        qedge = qgraph["edges"][qedge_id]
        kp_results = await kp_registry.search(
            qgraph["nodes"][qedge["subject"]]['categories'],
            qedge['predicates'],
            qgraph["nodes"][qedge["object"]]['categories'],
        )
        if not kp_results:
            msg = f"No KPs for qedge '{qedge_id}'"
            logger.error(msg)
            raise NoAnswersError(msg)
        for kp in kp_results.values():
            for op in kp["operations"]:
                op["subject"] = (qedge["subject"], op.pop("subject_category"))
                op["object"] = (qedge["object"], op.pop("object_category"))
        kps[qedge_id] = list(kp_results.values())

    if len(traversals) == 0:
        logger.warning({
            "code": "QueryNotTraversable",
            "message":
            """
                We couldn't find any possible plans starting from a pinned node
                that traverse every edge and node in your query graph
            """
        })

    return traversals, kps


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
                "reverse": og_edge["edge_reverse"],
            })
    return kps


# pylint: disable=too-many-arguments
async def step_to_kps(
        subject, edge, object,
        kp_registry: Registry,
        allowlist=None, denylist=None,
):
    """Find KP endpoint(s) that enable step."""
    return await kp_registry.search(
        subject['categories'],
        edge['predicates'],
        object['categories'],
        allowlist=allowlist,
        denylist=denylist,
    )
