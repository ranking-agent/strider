"""Query planner."""
import asyncio
from collections import defaultdict
import os

from fastapi import HTTPException
import httpx

from strider.biolink_model import BiolinkModel

KPREGISTRY_URL = os.getenv('KPREGISTRY_URL', 'http://localhost:4983')
BLM = BiolinkModel('https://raw.githubusercontent.com/biolink/biolink-model/master/biolink-model.yaml')


class Planner():
    """Planner."""

    def __init__(self, query_graph):
        """Initialize."""
        self.query_nodes_by_id = {
            node['id']: node
            for node in query_graph['nodes']
        }
        self.query_edges_by_id = {
            edge['id']: edge
            for edge in query_graph['edges']
        }

    @property
    def query_nodes(self):
        """Get query nodes."""
        return self.query_nodes_by_id.values()

    @property
    def query_edges(self):
        """Get query edges."""
        return self.query_edges_by_id.values()

    def validate_traversable(self, plan):
        """Validate that the query graph is traversable by this plan.

        Raises RuntimeError if not traversable.
        """
        # initialize starting nodes
        to_visit = {
            node['id']
            for node in self.query_nodes
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
                edge_id = step_id.split('-')[1]
                if not kps:
                    continue
                target_ids.add(self.query_edges_by_id[edge_id]['target_id'])
                visited.add(edge_id)
            to_visit |= (target_ids - visited)

        # make sure we visited everything
        things = set(node['id'] for node in self.query_nodes) | set(edge['id'] for edge in self.query_edges)
        if visited != things:
            missing = things - visited
            raise HTTPException(
                status_code=400,
                detail=f'The query is not traversable. The following nodes/edges cannot be reached: {", ".join(missing)}'
            )

    async def plan(self):
        """Generate a query execution plan."""
        # get candidate steps
        # i.e. steps we could imagine taking through the qgraph
        candidate_steps = defaultdict(list)
        for edge in self.query_edges:
            candidate_steps[edge['source_id']].append((f'-{edge["id"]}->', edge["target_id"]))
            candidate_steps[edge['target_id']].append((f'<-{edge["id"]}-', edge["source_id"]))

        # evaluate which candidates are realizable
        plan = dict()
        for source_id, steps in candidate_steps.items():
            plan[source_id] = dict()
            for edge_id, target_id in steps:
                step_id = edge_id + target_id
                plan[source_id][step_id] = await self.step_to_kps(source_id, edge_id, target_id)

        self.validate_traversable(plan)

        return plan

    async def step_to_kps(self, source_id, edge_id, target_id):
        """Find KP endpoint(s) that enable step."""
        source = self.query_nodes_by_id[source_id]
        edge = self.query_edges_by_id[edge_id.split('-')[1]]
        target = self.query_nodes_by_id[target_id]
        async with httpx.AsyncClient() as client:
            responses = await asyncio.gather(
                expand_bl(source["type"]),
                expand_bl(target["type"]),
                expand_bl(edge["type"])
            )
            source_types, target_types, edge_types = responses

        if source_id == edge['source_id']:
            edge_types = [f'-{edge_type}->' for edge_type in edge_types]
        else:
            edge_types = [f'<-{edge_type}-' for edge_type in edge_types]
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{KPREGISTRY_URL}/search',
                json={
                    'source_type': source_types,
                    'target_type': target_types,
                    'edge_type': edge_types,
                }
            )
            assert response.status_code < 300
        return response.json()


async def generate_plan(query_graph):
    """Generate a query execution plan."""
    return await Planner(query_graph).plan()


async def expand_bl(concept):
    """Return lineage of biolink concept."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f'https://bl-lookup-sri.renci.org/bl/{concept}/lineage?version=latest'
        )
    if response.status_code >= 300:
        return [concept]
    return response.json()
