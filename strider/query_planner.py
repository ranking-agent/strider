"""Query planner."""
import asyncio
from collections import defaultdict, namedtuple
import logging
import os
import re
import copy
import time
from typing import Callable

from bmt import Toolkit as BMToolkit
from reasoner_pydantic import QueryGraph, Message

from strider.kp_registry import Registry
from strider.normalizer import Normalizer
from strider.util import WrappedBMT
from strider.trapi import fix_qgraph
from strider.compatibility import Synonymizer
from strider.config import settings

BMT = BMToolkit()
WBMT = WrappedBMT(BMT)

LOGGER = logging.getLogger(__name__)

Step = namedtuple("Step", ["source", "edge", "target"])


def permute_qg(qg):
    permutations = []

    stack = []
    stack.append(copy.deepcopy(qg))

    while len(stack) > 0:
        current_qg = stack.pop()
        # Find next node that needs to be permuted
        remaining_unbound_nodes = (i for (i, n) in current_qg['nodes'].items()
                                   if 'type' in n and isinstance(n['type'], list))
        next_node_id = next(remaining_unbound_nodes, None)

        # Find next edge that needs to be permuted
        remaining_unbound_edges = (i for (i, n) in current_qg['edges'].items()
                                   if isinstance(n['predicate'], list))
        next_edge_id = next(remaining_unbound_edges, None)

        if next_node_id:
            # Permute this node and push permutations to stack
            next_node = current_qg['nodes'][next_node_id]
            for t in next_node['type']:
                permutation_copy = copy.deepcopy(current_qg)
                # Fix type
                permutation_copy['nodes'][next_node_id]['type'] = t
                print(f"Adding to stack: {permutation_copy}")
                # Add to stack
                stack.append(permutation_copy)
        elif next_edge_id:
            # Permute this edge and push permutations to stack
            next_edge = current_qg['edges'][next_edge_id]
            for p in next_edge['predicate']:
                permutation_copy = copy.deepcopy(current_qg)
                # Fix predicate
                permutation_copy['edges'][next_edge_id]['predicate'] = p
                # Add to stack
                stack.append(permutation_copy)
        else:
            # fully permuted, add to results
            permutations.append(current_qg)
    return permutations


def get_kp_request_template(
        qgraph: QueryGraph,
        step: Step,
        curie_map: dict[str, str] = None,
) -> Callable:
    """Get request to send to KP."""
    qgraph = fix_qgraph(qgraph, curie_map)

    def request(curie: str) -> Message:
        return {
            "query_graph": {
                "nodes": {
                    step.source: {
                        **qgraph['nodes'][step.source],
                        "id": curie,
                    },
                    step.target: qgraph['nodes'][step.target],
                },
                "edges": {step.edge: qgraph['edges'][step.edge]},
            },
        }
    return request


class NoAnswersError(Exception):
    """No answers can be found."""


def complete_plan(
        ops: dict[Step, dict],
        qgraph: QueryGraph,
        found: list[str],
        plan: list[Step] = None,
) -> list[dict[Step, dict]]:
    """Return all completed plans."""
    if plan is None:
        plan = []
    if len(found) >= len(qgraph["nodes"]) + len(qgraph["edges"]):
        return [plan]
    completions = []
    for source_id in reversed(found):
        for step, kps in ops.items():
            if source_id != step.source:
                continue
            if not kps:
                continue
            if step.edge in found:
                continue
            completions.extend(complete_plan(
                ops,
                qgraph,
                found + [step.edge, step.target],
                plan + [step],
            ))
    return completions


def filter_ancestor_types(types):
    """ Filter out types that are ancestors of other types in the list """

    def is_ancestor(a, b):
        """ Check if one biolink type is an ancestor of the other """
        ancestors = bmt_ancestors(b)
        if a in ancestors:
            return True
        return False

    specific_types = ['biolink:NamedThing']
    for new_type in types:
        for existing_type_id in range(len(specific_types)):
            existing_type = specific_types[existing_type_id]
            if is_ancestor(new_type, existing_type):
                continue
            elif is_ancestor(existing_type, new_type):
                specific_types[existing_type_id] = new_type
            else:
                specific_types.append(new_type)
    return specific_types


async def generate_plan(
        qgraph: QueryGraph,
        kp_registry: Registry = None,
        normalizer: Normalizer = None,
):
    """Generate a query execution plan."""
    if kp_registry is None:
        kp_registry = Registry(settings.kpregistry_url)
    if normalizer is None:
        normalizer = Normalizer(settings.normalizer_url)

    breakpoint()

    # Use BMT to convert node types to types + descendents
    for node in qgraph['nodes'].values():
        if 'type' not in node:
            continue
        if not isinstance(node['type'], list):
            node['type'] = [node['type']]
        new_type_list = []
        for t in node['type']:
            new_type_list.extend(bmt_descendents(t))
        node['type'] = new_type_list

    # Same with edges
    for edge in qgraph['edges'].values():
        if 'predicate' not in edge:
            continue
        if not isinstance(edge['predicate'], list):
            edge['predicate'] = [edge['predicate']]
        new_predicate_list = []
        for t in edge['predicate']:
            new_predicate_list.extend(bmt_descendents(t))
        edge['predicate'] = new_predicate_list

    # Use node normalizer to add
    # a type to nodes with a curie
    for node in qgraph['nodes'].values():
        if 'id' not in node:
            continue
        if not isinstance(node['id'], list):
            node['id'] = [node['id']]

        # Get full list of types
        types = await normalizer.get_types(node['id'])

        # Filter types that are ancestors of other types we were given
        node['type'] = filter_ancestor_types(types)

    permuted_qg_list = permute_qg(qgraph)

    kp_lookup_dict = {}
    # For each edge, ask KP registry for KPs that could solve it
    candidate_steps = defaultdict(list)
    for edge in qgraph['edges'].values():
        if "provided_by" in edge:
            allowlist = edge["provided_by"].get("allowlist", None)
            denylist = edge["provided_by"].get("denylist", None)
        else:
            allowlist = denylist = None

        source = qgraph['nodes'][edge['subject']]
        target = qgraph['nodes'][edge['object']]

        kp_results = await step_to_kps(
            source, edge, target,
            kp_registry,
            allowlist=allowlist, denylist=denylist,
        )
        for kp_name, kp in kp_results.items():
            for op in kp['operations']:
                dict_key = Step(op['source_type'],
                                op['edge_type'], op['target_type'])
                if dict_key not in kp_lookup_dict:
                    kp_lookup_dict[dict_key] = []
                kp_lookup_dict[dict_key].append({
                    'name': kp_name,
                    'url': kp['url'],
                })

    print(f"KP Lookup Dictionary: {kp_lookup_dict}")

    # find possible plans
    pinned = [
        key for key, value in qgraph["nodes"].items()
        if value.get("id", None) is not None
    ]
    plans = complete_plan(real_steps, qgraph, pinned)
    if not plans:
        raise NoAnswersError("Cannot traverse query graph using KPs")

    plans = [{
        key: real_steps[key]
        for key in plan
    } for plan in plans]

    # add request templates to steps
    synonymizer = Synonymizer()
    for plan in plans:
        for step, endpoints in plan.items():
            for _, meta in endpoints.items():
                meta["request_template"] = get_kp_request_template(
                    qgraph,
                    step,
                    synonymizer.map(meta["preferred_prefixes"]),
                )

    return plans


async def step_to_kps(
        source, edge, target,
        kp_registry: Registry,
        allowlist=None, denylist=None,
):
    """Find KP endpoint(s) that enable step."""
    edge_types = [f'-{edge_type}->' for edge_type in edge['predicate']]
    print(f"Searching: {source['type']}{edge_types}{target['type']}")
    return await kp_registry.search(
        source['type'],
        edge_types,
        target['type'],
        allowlist=allowlist,
        denylist=denylist,
    )
