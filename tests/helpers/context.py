"""Test utilities."""

import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
from functools import partial, wraps
from typing import Optional
from urllib.parse import urlparse

from asgiar import ASGIAR
from fastapi import FastAPI, Response
import httpx

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
async def norm_overlay(
    url: str,
    normalizer_data: str = "",
):
    """Normalizer server context manager."""

    normalizer_data_dict = normalizer_data_from_string(normalizer_data)

    async with AsyncExitStack() as stack:
        app = FastAPI()
        app.include_router(
            norm_router(
                synset_mappings=normalizer_data_dict["synset_mappings"],
                category_mappings=normalizer_data_dict["category_mappings"],
                ic=normalizer_data_dict["information_content"],
            )
        )
        await stack.enter_async_context(ASGIAR(app, host=url_to_host(url)))
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
        @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
        async def all_paths(path):
            return response

        await stack.enter_async_context(ASGIAR(app, url=url))
        yield


@asynccontextmanager
async def translator_overlay(
    normalizer_url: str,
    response_url: str,
    response: Response,
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
        await stack.enter_async_context(
            response_overlay(
                response_url,
                response,
            )
        )

        yield


@asynccontextmanager
async def callback_overlay(url: str, queue: Optional[asyncio.Queue] = None):
    """
    Create a router that is able to recieve a POST request,
    save the information, and then provide that response again
    with a GET request
    """
    async with AsyncExitStack() as stack:
        app = FastAPI()

        # pylint: disable=unused-variable disable=unused-argument
        @app.post("/{path:path}")
        async def save_response(results: dict):
            if queue is not None:
                await queue.put(results)

        await stack.enter_async_context(ASGIAR(app, host=url_to_host(url)))
        yield


with_translator_overlay = partial(with_context, translator_overlay)
with_norm_overlay = partial(with_context, norm_overlay)
with_response_overlay = partial(with_context, response_overlay)
