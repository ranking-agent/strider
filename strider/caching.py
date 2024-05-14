"""Caching/locking."""

import asyncio
from collections import OrderedDict, namedtuple
from functools import wraps
import gzip
import json
import redis.asyncio as aioredis

from strider.config import settings


onehop_redis_pool = aioredis.BlockingConnectionPool(
    host=settings.redis_host,
    port=settings.redis_port,
    db=0,
    password=settings.redis_password,
    max_connections=10,
    timeout=600,
)
kp_redis_pool = aioredis.BlockingConnectionPool(
    host=settings.redis_host,
    port=settings.redis_port,
    db=1,
    password=settings.redis_password,
    max_connections=10,
    timeout=600,
)
post_request_redis_pool = aioredis.BlockingConnectionPool(
    host=settings.redis_host,
    port=settings.redis_port,
    db=2,
    password=settings.redis_password,
    max_connections=10,
    timeout=600,
)


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


def async_locking_cache(fcn, maxsize=32):
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
        return _Cache_Info(wrapper.hits, wrapper.misses, maxsize, len(wrapper.cache))

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


async def get_kp_onehop(kp_id, onehop):
    """Get onehop from cache if saved."""
    try:
        client = await aioredis.Redis(
            connection_pool=onehop_redis_pool,
        )
        response = await client.get(f"{kp_id}:{json.dumps(onehop)}")
        await client.close()
        if response is not None:
            response = json.loads(gzip.decompress(response))
    except Exception:
        # failed to get kp onehop
        response = None
    return response


async def save_kp_onehop(kp_id, onehop, response):
    """Cache a kp onehop."""
    try:
        client = await aioredis.Redis(
            connection_pool=onehop_redis_pool,
        )
        key = f"{kp_id}:{json.dumps(onehop)}"
        await client.setex(
            key, settings.redis_expiration, gzip.compress(json.dumps(response).encode())
        )
        await client.close()
    except Exception:
        # failed to save kp onehop
        pass


async def save_kp_registry(kps):
    """Cache a registry of all kps."""
    try:
        client = await aioredis.Redis(connection_pool=kp_redis_pool)
        await client.set("kps", gzip.compress(json.dumps(kps).encode()))
        await client.close()
    except Exception:
        # failed to save kp registry
        pass


async def get_kp_registry():
    """Get the registry of kps from cache."""
    try:
        client = await aioredis.Redis(
            connection_pool=kp_redis_pool,
        )
        response = await client.get("kps")
        await client.close()
        if response is not None:
            response = json.loads(gzip.decompress(response))
    except Exception:
        # failed to get kp registry
        response = None
    return response


async def get_registry_lock():
    """Lock registry lookup so only one worker will retrieve."""
    client = await aioredis.Redis(connection_pool=kp_redis_pool)
    locked = await client.get("locked")
    if locked is None:
        await client.setex("locked", 360, 1)
        await client.close()
        return True
    await client.close()
    return False


async def remove_registry_lock():
    """Remove lock from registry."""
    try:
        client = await aioredis.Redis(connection_pool=kp_redis_pool)
        await client.delete("locked")
        await client.close()
    except Exception:
        # failed to remove registry lock
        pass


async def save_post_request(url, request, response):
    """Save response from post request in cache."""
    try:
        client = await aioredis.Redis(
            connection_pool=post_request_redis_pool,
        )
        key = f'{{"{url}":{json.dumps(request)}}}'
        await client.setex(
            key, settings.redis_expiration, gzip.compress(json.dumps(response).encode())
        )
        await client.close()
    except Exception:
        # failed to save post request
        pass


async def get_post_response(url, request):
    """Get post response from cache."""
    try:
        client = await aioredis.Redis(
            connection_pool=post_request_redis_pool,
        )
        response = await client.get(f'{{"{url}":{json.dumps(request)}}}')
        await client.close()
        if response is not None:
            response = json.loads(gzip.decompress(response))
    except Exception:
        # failed to get post response
        response = None
    return response


async def clear_cache():
    """Clear one-hop redis cache."""
    client = await aioredis.Redis(connection_pool=onehop_redis_pool)
    await client.flushdb()
