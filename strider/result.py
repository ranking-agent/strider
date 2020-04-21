"""Result."""


class ValidationError(Exception):
    """Invalid node or edge."""


class Result():
    """Result."""

    def __init__(self, result, response, bmt):
        """Initialize."""
        knodes = response['knowledge_graph']['nodes']
        kedges = response['knowledge_graph']['edges']
        self.edges = {
            binding['qg_id']: kedges[binding['kg_id']]
            for binding in result['edge_bindings']
        }
        self.nodes = {
            binding['qg_id']: knodes[binding['kg_id']]
            for binding in result['node_bindings']
        }
        self.bmt = bmt

    async def validate(self, query):
        """Validate against query."""
        for qid, edge in self.edges.items():
            edge_spec = query.qgraph['edges'][qid]
            await self._validate(edge, edge_spec)
        for qid, node in self.nodes.items():
            target_spec = query.qgraph['nodes'][qid]
            await self._validate(node, target_spec)

    async def _validate(self, element, spec):
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
