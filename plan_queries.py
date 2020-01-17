"""Convert biolink model to graph."""
from collections import defaultdict
from biolink_model import BiolinkModel


BLM = BiolinkModel('https://raw.githubusercontent.com/biolink/biolink-model/master/biolink-model.yaml')

# define edges provided by KPs
pedges = []
pedges.append({
    'source': 'disease',
    # 'target': 'gene',
    'target': 'gene or gene product',
    'predicate': 'affects',
})
pedges.append({
    'source': 'gene',
    'target': 'biological process or activity',
    # 'target': 'biological process',
    'predicate': 'associated_with',
})
pedges.append({
    'source': 'biological process',
    'target': 'disease',
    'predicate': 'associated_with',
})

# define user query
query_graph = {
    'nodes': [
        {
            'id': 'n00',
            'type': 'disease',
            'curie': 'MONDO:0005737',
        },
        {
            'id': 'n01',
            'type': 'gene',
        },
        {
            'id': 'n02',
            'type': 'biological process',
        },
        {
            'id': 'n03',
            'type': 'disease or phenotypic feature',
        },
    ],
    'edges': [
        {
            'id': 'e01',
            'source_id': 'n00',
            'target_id': 'n01',
        },
        {
            'id': 'e12',
            'source_id': 'n01',
            'target_id': 'n02',
        },
        {
            'id': 'e23',
            'source_id': 'n02',
            'target_id': 'n03',
        },
    ],
}

# helper things
query_nodes_by_id = {
    node['id']: node
    for node in query_graph['nodes']
}

# get candidate steps
# i.e. steps we could imagine taking through the qgraph
candidate_steps = defaultdict(list)
for edge in query_graph['edges']:
    candidate_steps[edge['source_id']].append(edge['target_id'])
    candidate_steps[edge['target_id']].append(edge['source_id'])


def step_to_kps(source_id, target_id):
    """Find KP endpoint(s) that enable step."""
    source = query_nodes_by_id[source_id]
    target = query_nodes_by_id[target_id]
    return [pedge for pedge in pedges if (
        BLM.compatible(pedge['source'], source['type']) and
        BLM.compatible(pedge['target'], target['type'])
    )]


# evaluate which candidates are realizable
plan = {
    source_id: {target_id: step_to_kps(source_id, target_id) for target_id in target_ids}
    for source_id, target_ids in candidate_steps.items()
}

# check that query is traversable via KP edges
# initialize starting nodes
to_visit = {
    node['id']
    for node in query_graph['nodes']
    if 'curie' in node
}
visited = set()
while to_visit:
    source_id = to_visit.pop()
    # don't visit this node again
    visited.add(source_id)
    # remember to visit target nodes
    target_ids = {target_id for target_id, kps in plan[source_id].items() if kps}
    to_visit |= (target_ids - visited)

print(plan)
if visited != set(node['id'] for node in query_graph['nodes']):
    raise RuntimeError('We did not traverse the entire query.')
