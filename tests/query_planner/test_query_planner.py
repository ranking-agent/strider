from pathlib import Path
import os
import json

import pytest

from tests.helpers.context import with_registry_overlay

from strider.config import settings

# Switch settings before importing strider things
registry_host = "registry"
settings.kpregistry_url = f"http://{registry_host}"

from strider.query_planner import generate_plan


cwd = Path(__file__).parent


with open(cwd / "ex1_kps.json", "r") as f:
    kps = json.load(f)


@pytest.mark.asyncio
@with_registry_overlay(registry_host, kps)
async def test_ex1():

    with open(cwd / "ex1_qg.json", "r") as f:
        qg = json.load(f)

    plans = await generate_plan(qg)

    assert plans

    # We should have two valid plans
    assert len(plans) == 2
