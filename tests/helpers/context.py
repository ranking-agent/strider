"""Test utilities."""
from contextlib import asynccontextmanager, AsyncExitStack
from functools import partial, wraps
from urllib.parse import urlparse

import aiosqlite
from asgiar import ASGIAR
from fastapi import FastAPI, Response
import httpx
from kp_registry.routers.kps import registry_router
from binder.testing import kp_overlay

from .normalizer import norm_router
from .utils import normalizer_data_from_string

callback_results = {}

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
    normalizer_data: str = "",
):
    """Normalizer server context manager."""

    normalizer_data_dict = normalizer_data_from_string(normalizer_data)

    async with AsyncExitStack() as stack:
        app = FastAPI()
        app.include_router(norm_router(
            synset_mappings=normalizer_data_dict['synset_mappings'],
            category_mappings=normalizer_data_dict['category_mappings'],
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
            ASGIAR(app, url=url)
        )
        yield


@asynccontextmanager
async def translator_overlay(
        registry_url: str,
        normalizer_url: str,
        kp_data: dict[str, str] = {},
        normalizer_data: str = "",
):
    """Registry + KPs + Normalizer context manager."""
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(
            norm_overlay(
                normalizer_url,
                normalizer_data,
            )
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
                response = await client.get(f"http://{host}/meta_knowledge_graph")
                response.raise_for_status()
                metakg = response.json()
            kps[host] = {
                "url": f"http://{host}/query",
                "operations": [
                    {
                        "subject_category": edge["subject"],
                        "predicate": edge["predicate"],
                        "object_category": edge["object"],
                    }
                    for edge in metakg["edges"]
                ],
                "details": {
                    "preferred_prefixes": {
                        category: value["id_prefixes"]
                        for category, value in metakg["nodes"].items()
                    }
                }
            }

        # Start registry context using KPs constructed above
        await stack.enter_async_context(
            registry_overlay(registry_url, kps)
        )

        yield

@asynccontextmanager
async def callback_overlay(url,):
    """
    Create a router that is able to recieve a POST request,
    save the information, and then provide that response again
    with a GET request
    """
    async with AsyncExitStack() as stack:
        app = FastAPI()

        # pylint: disable=unused-variable disable=unused-argument
        @app.post('/{path:path}')
        async def save_response(results: dict):
            global callback_results
            callback_results = results

        @app.get('/{path:path}')
        async def get_response():
            global callback_results
            return callback_results

        await stack.enter_async_context(
            ASGIAR(app, host=url_to_host(url))
        )
        yield



with_kp_overlay = partial(with_context, kp_overlay)
with_registry_overlay = partial(with_context, registry_overlay)
with_translator_overlay = partial(with_context, translator_overlay)
with_norm_overlay = partial(with_context, norm_overlay)
with_response_overlay = partial(with_context, response_overlay)
with_callback_overlay = partial(with_context, callback_overlay)
