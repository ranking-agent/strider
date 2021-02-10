"""Test utilities."""
from contextlib import asynccontextmanager, AsyncExitStack
from functools import partial, wraps
from urllib.parse import urlparse

import aiosqlite
from asgiar import ASGIAR
from fastapi import FastAPI, Response
import httpx
from kp_registry.routers.kps import registry_router
from simple_kp.testing import kp_overlay
from simple_kp._types import CURIEMap
import small_kg

from .normalizer import norm_router


def url_to_host(url):
    # TODO modify ASGIAR to accept a URL instead of a host
    return urlparse(url).netloc


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
async def registry_overlay(url, kps):
    """Registry server context manager."""
    async with AsyncExitStack() as stack:
        app = FastAPI()
        connection = await stack.enter_async_context(
            aiosqlite.connect(":memory:")
        )
        app.include_router(registry_router(connection))
        await stack.enter_async_context(
            ASGIAR(app, host=url_to_host(url))
        )
        # Register KPs passed to the function
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{url}/kps",
                json=kps
            )
            response.raise_for_status()

        yield


@asynccontextmanager
async def norm_overlay(
    url: str,
    normalizer_data: dict[str, dict] = {},
):
    """Normalizer server context manager."""
    async with AsyncExitStack() as stack:
        app = FastAPI()
        app.include_router(norm_router(
            synset_mappings=normalizer_data.get('synset_mappings', None),
            category_mappings=normalizer_data.get('category_mappings', None),
        ))
        await stack.enter_async_context(
            ASGIAR(app, host=url_to_host(url))
        )
        yield


@asynccontextmanager
async def response_overlay(url, response: Response):
    """
    Create a router that returns the specified
    response for all routes
    """
    async with AsyncExitStack() as stack:
        app = FastAPI()

        # pylint: disable=unused-variable disable=unused-argument
        @app.api_route('/{path:path}', methods=["GET", "POST", "PUT", "DELETE"])
        async def all_paths(path):
            return response

        await stack.enter_async_context(
            ASGIAR(app, host=url_to_host(url))
        )
        yield


@asynccontextmanager
async def translator_overlay(
        registry_url: str,
        normalizer_url: str,
        kp_data: dict[str, str] = {},
        normalizer_data: dict[str, dict] = {},
):
    """Registry + KPs + Normalizer context manager."""
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(
            norm_overlay(normalizer_url, normalizer_data)
        )
        kps = dict()
        for host, data_string in kp_data.items():
            await stack.enter_async_context(
                kp_overlay(
                    host,
                    data=data_string,
                )
            )

            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://{host}/ops")
                response.raise_for_status()
                operations = response.json()
                response = await client.get(f"http://{host}/metadata")
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
            kps[host] = {
                "url": f"http://{host}/query",
                "operations": operations,
                "details": {
                    "preferred_prefixes": metadata["curie_prefixes"]
                }
            }

        # Start registry context using KPs constructed above
        await stack.enter_async_context(
            registry_overlay(registry_url, kps)
        )

        yield


with_kp_overlay = partial(with_context, kp_overlay)
with_registry_overlay = partial(with_context, registry_overlay)
with_translator_overlay = partial(with_context, translator_overlay)
with_norm_overlay = partial(with_context, norm_overlay)
with_response_overlay = partial(with_context, response_overlay)
