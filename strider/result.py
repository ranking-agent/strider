"""Result."""


class ValidationError(Exception):
    """Invalid node or edge."""


class Result():  # pylint: disable=too-few-public-methods
    """Result."""

    def __init__(self, result, qgraph, kgraph, bmt):
        """Initialize."""
        # TODO: use _all_ of the bound elements
        self.edges = {
            qg_id: kgraph['edges'][bound_edges[0]['kg_id']]
            for qg_id, bound_edges in result['edge_bindings'].items()
        }
        self.nodes = {
            qg_id: kgraph['nodes'][bound_nodes[0]['kg_id']]
            for qg_id, bound_nodes in result['node_bindings'].items()
        }
        self.bmt = bmt
        self.validate(qgraph)

    def validate(self, qgraph):
        """Validate against query."""
        for qid, edge in self.edges.items():
            edge_spec = qgraph['edges'][qid]
            self._validate(edge, edge_spec)
        for qid, node in self.nodes.items():
            target_spec = qgraph['nodes'][qid]
            self._validate(node, target_spec)

    def _validate(self, element, spec):
        """Validate a node against a query-node specification."""
        for key, value in spec.items():
            if value is None:
                continue
            if key == 'curie':
                if element['id'] != value:
                    raise ValidationError(f'{element["id"]} != {value}')
            elif key == 'type':
                if isinstance(element['type'], str):
                    lineage = (
                        self.bmt.ancestors(value)
                        + self.bmt.descendents(value)
                        + [value]
                    )
                    if element['type'] not in lineage:
                        raise ValidationError(
                            f'{element["type"]} not in {lineage}'
                        )
                elif isinstance(element['type'], list):
                    if value not in element['type']:
                        raise ValidationError(
                            f'{value} not in {element["type"]}'
                        )
                else:
                    raise ValueError('Type must be a str or list')
            elif key not in ['id', 'source_id', 'target_id']:
                if element[key] != value:
                    raise ValidationError(f'{element[key]} != {value}')
        return True
