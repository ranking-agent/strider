"""Query planner."""
from collections import namedtuple
import logging
import copy
from typing import Generator

from reasoner_pydantic import QueryGraph

from strider.kp_registry import Registry
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
        provided_by = {"allowlist": None, "denylist": None} | qedge.get("provided_by", {})
        kp_results = await kp_registry.search(
            qgraph["nodes"][qedge["subject"]]['categories'],
            qedge['predicates'],
            qgraph["nodes"][qedge["object"]]['categories'],
            allowlist=provided_by["allowlist"],
            denylist=provided_by["denylist"],
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
