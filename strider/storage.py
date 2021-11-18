"""Storage."""
from abc import ABC
import os
import json
import logging
from typing import Optional

import redis

from .config import settings


def get_client() -> redis.Redis:
    """Create a Redis client."""
    return redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )


def mapd(f, d):
    """Map function over dictionary values"""
    return {k: f(v) for k, v in d.items()}


class RedisValue(ABC):
    """Redis value."""

    def __init__(self, key: str, client: Optional[redis.Redis] = None):
        self.client = client or get_client()
        self.key = key

    def expire(self, when: int):
        self.client.expire(self.key, when)


class RedisHash(RedisValue):
    """Redis hash."""

    def get(self):
        v = self.client.hgetall(self.key)
        return mapd(json.loads, v)

    def set(self, v: dict):
        self.client.delete(self.key)
        if not len(v):
            return
        self.client.hset(self.key, mapping=mapd(json.dumps, v))

    def get_val(self):
        v = self.client.hget(self.key)
        return json.load(v)

    def merge(self, v: dict):
        if not len(v):
            return
        self.client.hset(self.key, mapping=mapd(json.dumps, v))


class RedisList(RedisValue):
    """Redis list."""

    def get(self):
        v = self.client.lrange(self.key, 0, -1)
        return map(json.loads, v)

    def set(self, v: list[any]):
        # Clear
        self.client.delete(self.key)
        self.client.push(**map(json.dumps, v))

    def append(self, v):
        self.client.lpush(self.key, json.dumps(v))


class RedisGraph:
    """Redis graph."""

    def __init__(self, key: str, client: Optional[redis.Redis] = None):
        self.nodes = RedisHash(key + ":nodes", client)
        self.edges = RedisHash(key + ":edges", client)

    def expire(self, when: int):
        self.nodes.expire(when)
        self.edges.expire(when)

    def get(self):
        return dict(nodes=self.nodes.get(), edges=self.edges.get())

    def set(self, v):
        self.nodes.set(v["nodes"])
        self.edges.set(v["edges"])


class RedisLogHandler(logging.Handler):
    """Redis log handler."""

    def __init__(self, key: str, client: Optional[redis.Redis] = None, **kwargs):
        self.store = RedisList(key, client)
        super().__init__(**kwargs)

    def emit(self, record):
        log_entry = self.format(record)
        self.store.append(log_entry)
