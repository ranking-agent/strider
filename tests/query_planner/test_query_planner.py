import json

import pytest

from tests.helpers.context import with_registry_overlay

with open("tests/query_planner/ex1_kps.json", "r") as f:
    kps = json.load(f)


@pytest.mark.asyncio
@with_registry_overlay("registry", kps)
async def test_planner():
    pass
