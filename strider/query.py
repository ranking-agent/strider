"""Query object."""
import json


async def create_query(uid, redis):
    """Create Query."""
    query = Query(uid, redis)
    await query._init()  # pylint: disable=protected-access
    return query


class Query():
    """Query."""

    def __init__(self, uid, redis):
        """Initialize."""
        self.uid = uid
        self.qgraph = None
        self.options = None
        self.redis = redis

    async def _init(self):
        """Asynchronously initialize qgraph."""
        self.qgraph = json.loads(await self.redis.get(f'{self.uid}_qgraph'))
        self.options = {
            key: json.loads(value)
            for key, value in (await self.redis.hgetall(
                f'{self.uid}_options'
            )).items()
        }

    async def is_done(self, job_id):
        """Return boolean indicating if a job is completed."""
        return (
            await self.redis.exists(f'{self.uid}_done')
            and await self.redis.sismember(f'{self.uid}_done', job_id)
        )

    async def finish(self, job_id):
        """Mark a job as completed."""
        await self.redis.sadd(f'{self.uid}_done', job_id)

    async def get_priority(self, job_id):
        """Get priority for job."""
        return min(255, int(float(await self.redis.hget(
            f'{self.uid}_priorities',
            job_id
        ))))

    async def get_steps(self, qid, kid):
        """Get steps for query graph node id."""
        job_id = f'({qid}:{kid})'

        # never process the same job twice
        if await self.is_done(job_id):
            return None, qid, kid, dict()
        await self.finish(job_id)

        priority = await self.get_priority(job_id)

        steps_string = await self.redis.hget(
            f'{self.uid}_plan',
            qid,
            encoding='utf-8',
        )
        try:
            steps = json.loads(steps_string)
        except TypeError:
            steps = dict()
        return priority, qid, kid, steps

    async def get_start_time(self):
        """Get start time."""
        return float(await self.redis.get(
            f'{self.uid}_starttime'
        ))

    async def update_priority(self, job_id, incr):
        """Update priority."""
        await self.redis.hincrbyfloat(
            f'{self.uid}_priorities',
            job_id,
            incr,
        )
