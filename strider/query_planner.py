"""Query planner."""
import asyncio
from collections import defaultdict, namedtuple
import logging
import os
from typing import Callable

from bmt import Toolkit as BMToolkit
from reasoner_pydantic import QueryGraph, Message

from strider.kp_registry import Registry
from strider.util import snake_case, spaced
from strider.trapi import fix_qgraph
from strider.compatibility import Synonymizer

BMT = BMToolkit()
KPREGISTRY_URL = os.getenv('KPREGISTRY_URL', 'http://localhost:4983')
LOGGER = logging.getLogger(__name__)

Step = namedtuple("Step", ["source", "edge", "target"])


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


async def generate_plan(
        qgraph: QueryGraph,
        kp_registry: Registry = None,
):
    """Generate a query execution plan."""
    if kp_registry is None:
        kp_registry = Registry(KPREGISTRY_URL)

    # get candidate steps
    # i.e. steps we could imagine taking through the qgraph
    candidate_steps = defaultdict(list)
    for qedge_id, qedge in qgraph['edges'].items():
        if "provided_by" in qedge:
            allowlist = qedge["provided_by"].get("allowlist", None)
            denylist = qedge["provided_by"].get("denylist", None)
        else:
            allowlist = denylist = None
        candidate_steps[qedge["subject"]].append(
            (
                qedge_id,
                qedge["object"],
                allowlist,
                denylist,
            )
        )
        candidate_steps[qedge["object"]].append(
            (
                qedge_id,
                qedge["subject"],
                allowlist,
                denylist,
            )
        )

    # evaluate which candidates are realizable
    real_steps = dict()
    for source_id, steps in candidate_steps.items():
        # real_steps[source_id] = dict()
        for edge_id, target_id, allowlist, denylist in steps:
            source = {**qgraph['nodes'][source_id], "key": source_id}
            edge = qgraph['edges'][edge_id]
            target = qgraph['nodes'][target_id]
            step = Step(source_id, edge_id, target_id)
            real_steps[step] = await step_to_kps(
                source, edge, target,
                kp_registry,
                allowlist=allowlist, denylist=denylist,
            )

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
    source_types, target_types, edge_types = await asyncio.gather(
        expand_bl(source.get("category", None)),
        expand_bl(target.get("category", None)),
        expand_bl(edge.get("predicate", None))
    )

    if source["key"] == edge["subject"]:
        edge_types = [f'-{edge_type}->' for edge_type in edge_types]
    else:
        edge_types = [f'<-{edge_type}-' for edge_type in edge_types]
    return await kp_registry.search(
        source_types,
        edge_types,
        target_types,
        allowlist=allowlist,
        denylist=denylist,
    )


async def expand_bl(concept):
    """Return lineage of biolink concept."""
    if concept is None:
        concept = 'biolink:NamedThing'
    _concept = spaced(concept)
    return snake_case(
        BMT.ancestors(_concept)
        + BMT.descendents(_concept)
    ) + [concept]
