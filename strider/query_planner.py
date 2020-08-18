"""Query planner."""
import asyncio
from collections import defaultdict
import logging
import os

from bmt import Toolkit as BMToolkit

from strider.kp_registry import Registry
from strider.util import snake_case, spaced

BMT = BMToolkit()
KPREGISTRY_URL = os.getenv('KPREGISTRY_URL', 'http://localhost:4983')
LOGGER = logging.getLogger(__name__)


class Planner():
    """Planner."""

    def __init__(self, kp_registry=None):
        """Initialize."""
        if kp_registry is None:
            kp_registry = Registry(KPREGISTRY_URL)
        self.kp_registry = kp_registry

    def validate_traversable(self, plan, qgraph):
        """Validate that the query graph is traversable by this plan.

        Raises RuntimeError if not traversable.
        """
        # initialize starting nodes
        to_visit = {
            node_id
            for node_id, node in qgraph['nodes'].items()
            if 'curie' in node
        }

        # follow plan
        visited = set()
        while to_visit:
            source_id = to_visit.pop()  # don't visit this node again
            visited.add(source_id)
            # remember to visit target nodes
            target_ids = set()
            for step_id, kps in plan.get(source_id, dict()).items():
                if not kps:
                    continue
                edge_id = step_id.split('-')[1]
                target_ids.add(qgraph['edges'][edge_id]['target_id'])
                visited.add(edge_id)
            to_visit |= (target_ids - visited)

        # make sure we visited everything
        things = (
            set(qgraph['nodes'])
            | set(qgraph['edges'])
        )
        if visited != things:
            missing = things - visited
            message = 'The query is not traversable. ' \
                      'The following nodes/edges cannot be reached: ' \
                      '{0}'.format(', '.join(missing))
            raise RuntimeError(message)

    async def plan(self, qgraph):
        """Generate a query execution plan."""
        # get candidate steps
        # i.e. steps we could imagine taking through the qgraph
        candidate_steps = defaultdict(list)
        for edge in qgraph['edges'].values():
            candidate_steps[edge['source_id']].append(
                (f'-{edge["id"]}->', edge["target_id"])
            )
            candidate_steps[edge['target_id']].append(
                (f'<-{edge["id"]}-', edge["source_id"])
            )

        # evaluate which candidates are realizable
        plan = dict()
        for source_id, steps in candidate_steps.items():
            plan[source_id] = dict()
            for edge_id, target_id in steps:
                step_id = edge_id + target_id
                source = qgraph['nodes'][source_id]
                edge = qgraph['edges'][edge_id.split('-')[1]]
                target = qgraph['nodes'][target_id]
                plan[source_id][step_id] = await self.step_to_kps(
                    source, edge, target
                )

        self.validate_traversable(plan, qgraph)

        return plan

    async def step_to_kps(self, source, edge, target):
        """Find KP endpoint(s) that enable step."""
        responses = await asyncio.gather(
            expand_bl(source["type"]),
            expand_bl(target["type"]),
            expand_bl(edge["type"])
        )
        source_types, target_types, edge_types = responses

        if source['id'] == edge['source_id']:
            edge_types = [f'-{edge_type}->' for edge_type in edge_types]
        else:
            edge_types = [f'<-{edge_type}-' for edge_type in edge_types]
        return await self.kp_registry.search(
            source_types,
            edge_types,
            target_types,
        )


async def generate_plan(query_graph, kp_registry=None):
    """Generate a query execution plan."""
    return await Planner(kp_registry).plan(query_graph)


async def expand_bl(concept):
    """Return lineage of biolink concept."""
    if concept is None:
        concept = 'named_thing'
    _concept = spaced(concept)
    return snake_case(
        BMT.ancestors(_concept)
        + BMT.descendents(_concept)
    ) + [concept]
