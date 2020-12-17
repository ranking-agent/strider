from pathlib import Path
import os
import json

import pytest

from tests.helpers.context import with_registry_overlay

cwd = Path(__file__).parent

with open(cwd / "ex1_kps.json", "r") as f:
    kps = json.load(f)

print(__file__)


@pytest.mark.asyncio
@with_registry_overlay("registry", kps)
async def test_planner():
    pass
