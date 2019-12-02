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

# identify pedges matching qedges
pqedges_by_qedge = defaultdict(list)
for qedge in query_graph['edges']:
    qedge_source = query_nodes_by_id[qedge['source_id']]
    qedge_target = query_nodes_by_id[qedge['target_id']]
    for pedge in pedges:
        if (BLM.compatible(pedge['source'], qedge_source['type']) and
            BLM.compatible(pedge['target'], qedge_target['type'])):
            # pedge and qedge are aligned
            pqedges_by_qedge[qedge['id']].append({
                'source': qedge_source['id'],
                'target': qedge_target['id'],
                'details': pedge,
            })
        if (BLM.compatible(pedge['source'], qedge_target['type']) and
            BLM.compatible(pedge['target'], qedge_source['type'])):
            # pedge and qedge are inverted
            pqedges_by_qedge[qedge['id']].append({
                'source': qedge_target['id'],
                'target': qedge_source['id'],
                'details': pedge,
            })

pedges = [x for value in pqedges_by_qedge.values() for x in value]
pqedges_by_source = defaultdict(list)
for edge in pedges:
    pqedges_by_source[edge['source']].append(edge)

# check that query is traversable via KP edges
# initialize starting nodes
to_visit = {
    node['id']
    for node in query_graph['nodes']
    if 'curie' in node
}
visited = set()
while to_visit:
    node = query_nodes_by_id[to_visit.pop()]
    # don't visit this node again
    visited.add(node['id'])
    # remember to visit target nodes
    targets = {edge['target'] for edge in pqedges_by_source[node['id']]}
    to_visit |= (targets - visited)

if visited != set(node['id'] for node in query_graph['nodes']):
    raise RuntimeError('We did not traverse the entire query.')
print(pqedges_by_source)
