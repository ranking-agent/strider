"""Query object."""
import json


async def create_query(uid, redis):
    """Create Query."""
    query = Query(uid)
    await query._init(redis)  # pylint: disable=protected-access
    return query


class Query():
    """Query."""

    def __init__(self, uid):
        """Initialize."""
        self.uid = uid
        self.qgraph = None

    async def _init(self, redis):
        """Asynchronously initialize qgraph."""
        self.qgraph = json.loads(await redis.get(f'{self.uid}_qgraph'))

    async def get_qgraph(self, redis):
        """Get qgraph."""
        qgraph = {
            'nodes': {},
            'edges': {},
        }
        for value in await redis.hvals(f'{self.uid}_slots'):
            value = json.loads(value)
            if 'source_id' in value:
                qgraph['edges'][value['id']] = value
            else:
                qgraph['nodes'][value['id']] = value
        return qgraph
