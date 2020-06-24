"""Query object."""
from collections import defaultdict
import json
import time
import uuid

from strider.query_planner import generate_plan


async def create_query(qgraph):
    """Create Query."""
    query = Query(qgraph)
    await query._init()  # pylint: disable=protected-access
    return query


class Query():
    """Query."""

    def __init__(self, qgraph, **options):
        """Initialize."""
        self.uid = str(uuid.uuid4())
        self.qgraph = qgraph
        self.options = options
        self.done = defaultdict(bool)
        self.priorities = defaultdict(int)
        self.start_time = time.time()
        self.plan = None

    async def _init(self):
        """Generate query plan."""
        self.plan = await generate_plan(self.qgraph)

    async def is_done(self, job_id):
        """Return boolean indicating if a job is completed."""
        return self.done[job_id]

    async def finish(self, job_id):
        """Mark a job as completed."""
        self.done[job_id] = True

    async def get_priority(self, job_id):
        """Get priority for job."""
        return self.priorities[job_id]

    async def get_steps(self, qid, kid):
        """Get steps for query graph node id."""
        job_id = f'({qid}:{kid})'

        # never process the same job twice
        if self.done[job_id]:
            return dict()
        await self.finish(job_id)

        return self.plan[qid]

    async def get_start_time(self):
        """Get start time."""
        return self.start_time

    async def update_priority(self, job_id, incr):
        """Update priority."""
        self.priorities[job_id] += incr
