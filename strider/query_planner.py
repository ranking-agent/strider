"""Query planner."""
import asyncio
from collections import defaultdict

from fastapi import HTTPException
import httpx

from strider.biolink_model import BiolinkModel

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
                edge_id, _ = step_id.split('/')
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
                status_code=404,
                detail=f'The query is not traversable. The following nodes/edges cannot be reached: {", ".join(missing)}'
            )

    async def plan(self):
        """Generate a query execution plan."""
        # get candidate steps
        # i.e. steps we could imagine taking through the qgraph
        candidate_steps = defaultdict(list)
        for edge in self.query_edges:
            candidate_steps[edge['source_id']].append(edge['id'])
            candidate_steps[edge['target_id']].append(edge['id'])

        # evaluate which candidates are realizable
        plan = dict()
        for source_id, edge_ids in candidate_steps.items():
            plan[source_id] = dict()
            for edge_id in edge_ids:
                step_id = f'{edge_id}/{self.query_edges_by_id[edge_id]["target_id"]}'
                plan[source_id][step_id] = await self.step_to_kps(source_id, edge_id)

        self.validate_traversable(plan)

        return plan

    async def step_to_kps(self, source_id, edge_id):
        """Find KP endpoint(s) that enable step."""
        source = self.query_nodes_by_id[source_id]
        edge = self.query_edges_by_id[edge_id]
        target = self.query_nodes_by_id[edge['target_id']]
        async with httpx.AsyncClient() as client:
            responses = await asyncio.gather(
                client.get(
                    f'https://bl-lookup-sri.renci.org/bl/{source["type"]}/lineage?version=latest'
                ),
                client.get(
                    f'https://bl-lookup-sri.renci.org/bl/{target["type"]}/lineage?version=latest'
                ),
                client.get(
                    f'https://bl-lookup-sri.renci.org/bl/{edge["type"]}/lineage?version=latest'
                )
            )
            source_types, target_types, edge_types = (response.json() for response in responses)
            response = await client.get(
                f'http://localhost:4983/search',
                params={
                    'source_type': source_types,
                    'target_type': target_types,
                    'edge_type': edge_types
                }
            )
            assert response.status_code < 300
        return response.json()


async def generate_plan(query_graph):
    """Generate a query execution plan."""
    return await Planner(query_graph).plan()
