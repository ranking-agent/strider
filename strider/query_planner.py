"""Query planner."""

from collections import defaultdict, namedtuple
import copy
from itertools import chain
import logging
import math
from reasoner_pydantic import QueryGraph
from typing import Generator, Union, Dict

from strider.caching import get_kp_registry
from strider.config import settings
from strider.traversal import get_traversals, NoAnswersError
from strider.utils import WBMT
from strider.mcq import is_mcq_node

LOGGER = logging.getLogger(__name__)

Step = namedtuple("Step", ["source", "edge", "target"])
Operation = namedtuple(
    "Operation", ["source_category", "edge_predicate", "target_category"]
)

REVERSE_EDGE_SUFFIX = ".reverse"
SYMMETRIC_EDGE_SUFFIX = ".symmetric"
INVERSE_EDGE_SUFFIX = ".inverse"


def find_next_list_property(search_dict, fields_to_check):
    """Find first object in a dictionary where object[field] is a list"""
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

    for edge in query_graph["edges"].values():
        kp = edge["kp"]
        reverse = kp["reverse"]

        if reverse:
            source_node = query_graph["nodes"][edge["object"]]
            target_node = query_graph["nodes"][edge["subject"]]
        else:
            source_node = query_graph["nodes"][edge["subject"]]
            target_node = query_graph["nodes"][edge["object"]]

        if (
            "category" in source_node
            and source_node["category"] != kp["source_category"]
        ):
            raise ValueError()
        source_node["category"] = kp["source_category"]

        if (
            "category" in target_node
            and target_node["category"] != kp["target_category"]
        ):
            raise ValueError()
        target_node["category"] = kp["target_category"]

        if "predicate" in edge and edge["predicate"] != kp["edge_predicate"]:
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
        valid_adjacent_nodes = [node for node in adjacent_nodes if node not in path]
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

    graph_nodes = set(graph["nodes"].keys())
    pinned_nodes = {
        key
        for key, value in graph["nodes"].items()
        if value.get("id", None) is not None
    }
    path_nodes = {n for n in path if n in graph["nodes"].keys()}

    # path_nodes + pinned_nodes must cover all nodes in the graph
    return pinned_nodes | path_nodes == graph_nodes


def search(
    registry,
    subject_category,
    predicate,
    object_category,
    maturity,
    **kwargs,
):
    """Search for KPs matching a pattern."""
    # maturity is list of enums
    kps = {}
    for kp, val in registry.items():
        if val["maturity"] != maturity:
            # maturity not allowed, skipping
            continue
        kp_name = val["infores"]
        for operation in val["operations"]:
            if (
                any(
                    category == operation["subject_category"]
                    for category in subject_category
                )
                and any(predicate == operation["predicate"] for predicate in predicate)
                and any(
                    category == operation["object_category"]
                    for category in object_category
                )
            ):
                kps[kp_name] = {
                    "url": val["url"],
                    "title": kp,
                    "infores": val["infores"],
                    "maturity": val["maturity"],
                    "operations": [],
                    "details": copy.deepcopy(val["details"]),
                }
                kps[kp_name]["operations"].append(operation.copy())

    return kps


def get_kp_operations_queries(
    subject_categories: Union[str, list[str]] = None,
    predicates: Union[str, list[str]] = None,
    object_categories: Union[str, list[str]] = None,
):
    """Build queries to send to kp registry."""
    if isinstance(subject_categories, str):
        subject_categories = [subject_categories]
    if isinstance(predicates, str):
        predicates = [predicates]
    if isinstance(object_categories, str):
        object_categories = [object_categories]
    subject_categories = [
        desc for cat in subject_categories for desc in WBMT.get_descendants(cat)
    ]
    predicates = [desc for pred in predicates for desc in WBMT.get_descendants(pred)]
    inverse_predicates = [
        desc
        for pred in predicates
        if (inverse := WBMT.predicate_inverse(pred))
        for desc in WBMT.get_descendants(inverse)
    ]
    object_categories = [
        desc for cat in object_categories for desc in WBMT.get_descendants(cat)
    ]

    return subject_categories, object_categories, predicates, inverse_predicates


async def generate_plan(
    qgraph: QueryGraph,
    backup_kps: dict,
    logger: logging.Logger = None,
) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Generate traversal plan."""
    # check that qgraph is traversable
    get_traversals(qgraph.dict())

    if logger is None:
        logger = logging.getLogger(__name__)
    kps = dict()
    plan = dict()
    registry = await get_kp_registry()
    if registry is None:
        logger.warning(
            "Unable to get kp registry from cache. Falling back to in-memory registry..."
        )
        registry = backup_kps
    for qedge_id in qgraph.edges:
        inverse_kps = dict()
        qedge = qgraph.edges[qedge_id]
        provided_by = {"allowlist": None, "denylist": None} | qedge.dict().pop(
            "provided_by", {}
        )
        if is_mcq_node(qgraph.nodes[qedge.subject]) or is_mcq_node(
            qgraph.nodes[qedge.object]
        ):
            # TODO: update from hard-coded MCQ KPs
            direct_kps = {
                "infores:answer-coalesce": {
                    "url": settings.mcq_answer_coalescer,
                    "title": "Answer Coalescer",
                    "infores": "infores:answer-coalesce",
                    "maturity": "development",
                    "operations": [],
                    "details": {"preferred_prefixes": {}},
                },
                "infores:genetics-data-provider": {
                    "url": settings.mcq_genetics,
                    "title": "Genetics KP",
                    "infores": "infores:genetics-data-provider",
                    "maturity": "development",
                    "operations": [],
                    "details": {"preferred_prefixes": {}},
                },
                "infores:cohd": {
                    "url": settings.mcq_cohd,
                    "title": "COHD KP",
                    "infores": "infores:cohd",
                    "maturity": "development",
                    "operations": [],
                    "details": {"preferred_prefixes": {}},
                },
                "infores:semsemian": {
                    "url": settings.mcq_semsemian,
                    "title": "Semsemian Monarch KP",
                    "infores": "infores:semsemian",
                    "maturity": "development",
                    "operations": [],
                    "details": {"preferred_prefixes": {}},
                },
            }
        else:
            # normal lookup edge
            (
                subject_categories,
                object_categories,
                predicates,
                inverse_predicates,
            ) = get_kp_operations_queries(
                qgraph.nodes[qedge.subject].categories,
                qedge.predicates,
                qgraph.nodes[qedge.object].categories,
            )
            direct_kps = search(
                registry,
                subject_categories,
                predicates,
                object_categories,
                settings.openapi_server_maturity,
            )
            if inverse_predicates:
                inverse_kps = search(
                    registry,
                    object_categories,
                    inverse_predicates,
                    subject_categories,
                    settings.openapi_server_maturity,
                )
        kp_results = {
            kpid: details
            for kpid, details in chain(*(direct_kps.items(), inverse_kps.items()))
            if (
                (provided_by["allowlist"] is None or kpid in provided_by["allowlist"])
                and (
                    provided_by["denylist"] is None
                    or kpid not in provided_by["denylist"]
                )
            )
        }
        if not kp_results:
            msg = f"No KPs for qedge '{qedge_id}'"
            logger.info(msg)
            raise NoAnswersError(msg)
        for kp in kp_results.values():
            for op in kp["operations"]:
                op["subject"] = (qedge.subject, op.pop("subject_category"))
                op["object"] = (qedge.object, op.pop("object_category"))
        plan[qedge_id] = list(kp_results.keys())
        kps.update(kp_results)
    return plan, kps


def annotate_query_graph(
    query_graph: dict[str, dict],
    operation_graph: dict[str, dict],
):
    """
    Use an annotated operation_graph to annotate
    a query graphs with KPs.
    """
    for edge_id, edge in query_graph["edges"].items():
        edge["kps"] = get_query_graph_edge_kps(operation_graph, edge_id)


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
            kps.append(
                {
                    **kp,
                    "reverse": og_edge["edge_reverse"],
                }
            )
    return kps


N = 1_000_000  # total number of nodes
R = 25  # number of edges per node


def get_next_qedge(qgraph):
    """Get next qedge to solve."""
    qgraph = copy.deepcopy(qgraph)
    for qnode in qgraph["nodes"].values():
        if (
            qnode.get("set_interpretation") == "MANY"
            and len(qnode.get("member_ids") or []) > 0
        ):
            # MCQ
            qnode["ids"] = len(qnode["member_ids"])
        elif qnode.get("ids") is not None:
            qnode["ids"] = len(qnode["ids"])
        else:
            qnode["ids"] = N
    pinnednesses = {
        qnode_id: get_pinnedness(qgraph, qnode_id) for qnode_id in qgraph["nodes"]
    }
    efforts = {
        qedge_id: math.log(qgraph["nodes"][qedge["subject"]]["ids"])
        + math.log(qgraph["nodes"][qedge["object"]]["ids"])
        for qedge_id, qedge in qgraph["edges"].items()
    }
    edge_priorities = {
        qedge_id: pinnednesses[qedge["subject"]]
        + pinnednesses[qedge["object"]]
        - efforts[qedge_id]
        for qedge_id, qedge in qgraph["edges"].items()
    }
    qedge_id = max(edge_priorities, key=edge_priorities.get)
    return qedge_id, qgraph["edges"][qedge_id]


def get_pinnedness(qgraph, qnode_id):
    """Get pinnedness of each node."""
    adjacency_mat = get_adjacency_matrix(qgraph)
    num_ids = get_num_ids(qgraph)
    return -compute_log_expected_n(
        adjacency_mat,
        num_ids,
        qnode_id,
    )


def compute_log_expected_n(adjacency_mat, num_ids, qnode_id, last=None, level=0):
    """Compute the log of the expected number of unique knodes bound to the specified qnode in the final results."""
    log_expected_n = math.log(num_ids[qnode_id])
    if level < 10:
        for neighbor, num_edges in adjacency_mat[qnode_id].items():
            if neighbor == last:
                continue
            # ignore contributions >0 - edges should only _further_ constrain n
            log_expected_n += num_edges * min(
                max(
                    compute_log_expected_n(
                        adjacency_mat,
                        num_ids,
                        neighbor,
                        qnode_id,
                        level + 1,
                    ),
                    0,
                )
                + math.log(R / N),
                0,
            )
    return log_expected_n


def get_adjacency_matrix(qgraph):
    """Get adjacency matrix."""
    A = defaultdict(lambda: defaultdict(int))
    for qedge in qgraph["edges"].values():
        A[qedge["subject"]][qedge["object"]] += 1
        A[qedge["object"]][qedge["subject"]] += 1
    return A


def get_num_ids(qgraph):
    """Get the number of ids for each node."""
    return {qnode_id: qnode["ids"] for qnode_id, qnode in qgraph["nodes"].items()}
