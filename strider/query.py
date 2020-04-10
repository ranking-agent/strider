"""Query object."""
import json


async def create_query(uid, redis):
    """Create Query."""
    query = Query(uid)
    await query._init(redis)  # pylint: disable=protected-access
    return query


class Query():  # pylint: disable=too-few-public-methods
    """Query."""

    def __init__(self, uid):
        """Initialize."""
        self.uid = uid
        self.qgraph = None
        self.options = None

    async def _init(self, redis):
        """Asynchronously initialize qgraph."""
        self.qgraph = json.loads(await redis.get(f'{self.uid}_qgraph'))
        self.options = {
            key: json.loads(value)
            for key, value in (await redis.hgetall(
                f'{self.uid}_options'
            )).items()
        }
