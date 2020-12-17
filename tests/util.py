"""Test utilities."""
from contextlib import asynccontextmanager, AsyncExitStack
from functools import partial, wraps

import aiosqlite
from asgiar import ASGIAR
from fastapi import FastAPI
import httpx
from kp_registry.routers.kps import registry_router
from simple_kp.testing import kp_overlay
from simple_kp._types import CURIEMap
import small_kg

from tests.normalizer import norm_router


def with_context(context, *args_, **kwargs_):
    """Turn context manager into decorator."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with context(*args_, **kwargs_):
                await func(*args, **kwargs)
        return wrapper
    return decorator


@asynccontextmanager
async def registry_overlay(host, kps):
    """Registry server context manager."""
    async with AsyncExitStack() as stack:
        app = FastAPI()
        connection = await stack.enter_async_context(
            aiosqlite.connect(":memory:")
        )
        app.include_router(registry_router(connection))
        await stack.enter_async_context(
            ASGIAR(app, host=host)
        )
        # Register KPs passed to the function
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{host}/kps",
                json=kps
            )
            response.raise_for_status()

        yield


@asynccontextmanager
async def norm_overlay(host):
    """Normalizer server context manager."""
    async with AsyncExitStack() as stack:
        app = FastAPI()
        app.include_router(norm_router())
        await stack.enter_async_context(
            ASGIAR(app, host=host)
        )
        yield


@asynccontextmanager
async def translator_overlay(origins: list[tuple[str, CURIEMap]]):
    """Registry + KPs context manager."""
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(
            norm_overlay("normalizer")
        )
        kps = dict()
        for origin, curie_prefixes in origins:
            await stack.enter_async_context(
                kp_overlay(
                    origin,
                    curie_prefixes=curie_prefixes,
                    nodes_file=getattr(small_kg, origin).nodes_file,
                    edges_file=getattr(small_kg, origin).edges_file,
                )
            )

            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{origin}/ops")
                response.raise_for_status()
                operations = response.json()
                response = await client.get(f"http://{origin}/metadata")
                response.raise_for_status()
                metadata = response.json()
            node_types = {
                node_type
                for operation in operations
                for node_type in (
                    operation["source_type"],
                    operation["target_type"],
                )
            }
            kps[origin] = {
                "url": f"http://{origin}/query",
                "operations": operations,
                "details": {
                    "preferred_prefixes": metadata["curie_prefixes"]
                }
            }

        # Start registry context using KPs constructed above
        await stack.enter_async_context(
            registry_overlay("registry", kps)
        )

        yield


with_kp_overlay = partial(with_context, kp_overlay)
with_registry_overlay = partial(with_context, registry_overlay)
with_translator_overlay = partial(with_context, translator_overlay)
