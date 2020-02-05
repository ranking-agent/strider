"""Query planner."""
from collections import defaultdict
import sqlite3

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
        self.conn = sqlite3.connect('kps.db')

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context."""
        self.conn.close()

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
            raise RuntimeError(f'The query is not traversable. The following nodes/edges cannot be reached: {", ".join(missing)}')

    def plan(self):
        """Generate a query execution plan."""
        # get candidate steps
        # i.e. steps we could imagine taking through the qgraph
        candidate_steps = defaultdict(list)
        for edge in self.query_edges:
            candidate_steps[edge['source_id']].append(edge['id'])
            candidate_steps[edge['target_id']].append(edge['id'])

        # evaluate which candidates are realizable
        plan = {
            source_id: {f'{edge_id}/{self.query_edges_by_id[edge_id]["target_id"]}': self.step_to_kps(source_id, edge_id) for edge_id in edge_ids}
            for source_id, edge_ids in candidate_steps.items()
        }

        self.validate_traversable(plan)

        return plan

    def step_to_kps(self, source_id, edge_id):
        """Find KP endpoint(s) that enable step."""
        source = self.query_nodes_by_id[source_id]
        edge = self.query_edges_by_id[edge_id]
        target = self.query_nodes_by_id[edge['target_id']]
        # source_types = BLM.get_lineage(source['type'])
        source_types = (source['type'],)
        source_bindings = ', '.join('?' for _ in range(len(source_types)))
        # target_types = BLM.get_lineage(target['type'])
        target_types = (target['type'],)
        target_bindings = ', '.join('?' for _ in range(len(target_types)))
        # edge_types = BLM.get_lineage(edge['type'])
        edge_types = (edge['type'],)
        edge_bindings = ', '.join('?' for _ in range(len(edge_types)))
        c = self.conn.cursor()
        c.execute(f'''
            SELECT url FROM knowledge_providers
            WHERE source_type in ({source_bindings})
            AND edge_type in ({edge_bindings})
            AND target_type in ({target_bindings})
            ''', list(source_types) + list(edge_types) + list(target_types))
        return [row[0] for row in c.fetchall()]


def generate_plan(query_graph):
    """Generate a query execution plan."""
    with Planner(query_graph) as planner:
        return planner.plan()
