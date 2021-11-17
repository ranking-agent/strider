"""Caching/locking."""
import asyncio
from collections import OrderedDict, namedtuple
from functools import wraps
import json


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


def async_locking_query_cache(fcn, maxsize=32):
    """Cache decorator.

    Taken in part from:
    https://github.com/aio-libs/async-lru/blob/master/async_lru.py
    """
    if not asyncio.iscoroutinefunction(fcn):
        raise ValueError("Function is not asynchronous")

    @wraps(fcn)
    async def wrapper(*args, **kwargs):
        """Wrap function."""
        key = json.dumps((args, kwargs))
        # check cache
        task = wrapper.cache.get(key)
        if task is None:
            # put lock on query into cache
            _cache_miss(key)
            wrapper.cache[key] = asyncio.create_task(fcn(*args, **kwargs))
        else:
            # query already in cache, wait for result
            _cache_hit(key)
        if maxsize is not None and len(wrapper.cache) > maxsize:
            # remove least recently used (lru) query
            wrapper.cache.popitem(last=False)
        return await wrapper.cache[key]

    def cache_clear():
        wrapper.hits = wrapper.misses = 0
        wrapper.cache = OrderedDict()

    _Cache_Info = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])

    def cache_info():
        return _Cache_Info(
            wrapper.hits,
            wrapper.misses,
            maxsize,
            len(wrapper.cache)
        )

    def _cache_touch(key):
        try:
            wrapper.cache.move_to_end(key)
        except KeyError:
            pass

    def _cache_hit(key):
        wrapper.hits += 1
        _cache_touch(key)

    def _cache_miss(key):
        wrapper.misses += 1
        _cache_touch(key)

    cache_clear()
    wrapper.cache_clear = cache_clear
    wrapper.cache_info = cache_info

    return wrapper
