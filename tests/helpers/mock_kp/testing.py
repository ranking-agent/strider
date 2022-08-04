"""Testing utilities.

Utilities for testing _other_ apps using simple-kp.
"""
import aiosqlite
from asgiar import ASGIAR
from fastapi import FastAPI


from .build_db import add_data_from_string
from .router import kp_router

from ._contextlib import AsyncExitStack, asynccontextmanager


@asynccontextmanager
async def kp_app(**kwargs):
    """KP context manager."""
    app = FastAPI()

    async with aiosqlite.connect(":memory:") as connection:
        # add data to sqlite
        await add_data_from_string(connection, **kwargs)

        # add kp to app
        app.include_router(kp_router(connection, name=kwargs.get("name")))
        yield app


@asynccontextmanager
async def kp_overlay(host, **kwargs):
    """KP(s) server context manager."""
    async with AsyncExitStack() as stack:
        app = await stack.enter_async_context(kp_app(**kwargs, name=host))

        await stack.enter_async_context(ASGIAR(app, host=host))
        yield
