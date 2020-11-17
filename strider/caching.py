"""Caching/locking."""
import asyncio
from functools import wraps


def make_key(args, kwargs):
    """Generate hash from arguments.

    Based on:
    https://github.com/python/cpython/blob/cd3c2bdd5d53db7fe1d546543d32000070916552/Lib/functools.py#L448
    """
    key = args
    if kwargs:
        key += (object(),)
        for item in kwargs.items():
            key += item
    return hash(key)


def async_cache(fcn):
    """Cache decorator."""
    if not asyncio.iscoroutinefunction(fcn):
        raise ValueError("Function is not asynchronous")
    cache = {}

    @wraps(fcn)
    async def cache_wrapper(*args, **kwargs):
        """Wrap function."""
        key = make_key(args, kwargs)
        if key not in cache:
            cache[key] = await fcn(*args, **kwargs)
        return cache[key]

    return cache_wrapper


def async_locking_cache(fcn):
    """Cache decorator."""
    if not asyncio.iscoroutinefunction(fcn):
        raise ValueError("Function is not asynchronous")
    cache = {}

    @wraps(fcn)
    async def cache_wrapper(*args, **kwargs):
        """Wrap function."""
        key = make_key(args, kwargs)
        if key not in cache:
            cache[key] = asyncio.create_task(fcn(*args, **kwargs))
        return await cache[key]

    return cache_wrapper
