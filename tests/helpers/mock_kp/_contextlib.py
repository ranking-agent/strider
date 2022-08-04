"""Monkey-patch contextlib."""
from contextlib import *
from contextlib import _AsyncGeneratorContextManager
from functools import wraps


def _call(self, func):
    @wraps(func)
    async def inner(*args, **kwds):
        async with self.__class__(self.func, self.args, self.kwds):
            return await func(*args, **kwds)

    return inner


_AsyncGeneratorContextManager.__call__ = _call
