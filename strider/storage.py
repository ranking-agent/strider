"""Storage."""
from abc import ABC
import os
import json
import logging

from .config import settings

if settings.redis_url == "redis://fakeredis:6379/0":
    import fakeredis
    r = fakeredis.FakeRedis(
        encoding="utf-8",
        decode_responses=True,
    )
else:
    import redis
    r = redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )


def mapd(f, d):
    """ Map function over dictionary values """
    return {k: f(v) for k, v in d.items()}


class RedisValue(ABC):
    """Redis value."""

    def __init__(self, key: str):
        self.key = key

    def expire(self, when: int):
        r.expire(self.key, when)


class RedisHash(RedisValue):
    """Redis hash."""

    def get(self):
        v = r.hgetall(self.key)
        return mapd(json.loads, v)

    def set(self, v: dict):
        r.delete(self.key)
        if not len(v):
            return
        r.hset(
            self.key,
            mapping=mapd(json.dumps, v)
        )

    def get_val(self):
        v = r.hget(self.key)
        return json.load(v)

    def merge(self, v: dict):
        if not len(v):
            return
        r.hset(
            self.key,
            mapping=mapd(json.dumps, v)
        )


class RedisList(RedisValue):
    """Redis list."""

    def get(self):
        v = r.lrange(self.key, 0, -1)
        return map(json.loads, v)

    def set(self, v: list[any]):
        # Clear
        r.delete(self.key)
        r.push(**map(json.dumps, v))

    def append(self, v):
        r.lpush(self.key, json.dumps(v))


class RedisGraph():
    """Redis graph."""

    def __init__(self, key: str):
        self.nodes = RedisHash(key + ':nodes')
        self.edges = RedisHash(key + ':edges')

    def expire(self, when: int):
        self.nodes.expire(when)
        self.edges.expire(when)

    def get(self):
        return dict(
            nodes=self.nodes.get(),
            edges=self.edges.get()
        )

    def set(self, v):
        self.nodes.set(v['nodes'])
        self.edges.set(v['edges'])


class RedisLogHandler(logging.Handler):
    """Redis log handler."""

    def __init__(self, key: str, *args, **kwargs):
        self.store = RedisList(key)
        super().__init__(*args, **kwargs)

    def emit(self, record):
        log_entry = self.format(record)
        self.store.append(log_entry)
