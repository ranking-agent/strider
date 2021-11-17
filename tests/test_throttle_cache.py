"""Test Strider TRAPI Throttle LRU Caching."""
import json
import pytest

from strider.caching import async_locking_cache


@pytest.mark.asyncio
async def test_async_query_caching():
    """Test simple query caching."""
    @async_locking_cache
    async def test_query(query):
        return "Test"
    await test_query({"query": "test"})
    assert test_query.cache_info().hits == 0
    await test_query({"query": "test2"})
    assert test_query.cache_info().hits == 0
    await test_query({"query": "test"})
    assert test_query.cache_info().hits == 1


@pytest.mark.asyncio
async def test_cache_function_error():
    """Test cache needs async function."""
    with pytest.raises(ValueError):
        @async_locking_cache
        def test_query(query):
            return "Test"


@pytest.mark.asyncio
async def test_caching_maxsize():
    """Test caching maxsize parameter."""
    async def test_query(query):
        return "Test"
    test_query = async_locking_cache(test_query, maxsize=1)
    await test_query({"query": "test"})
    assert test_query.cache_info().hits == 0
    await test_query({"query": "test2"})
    assert test_query.cache_info().hits == 0
    await test_query({"query": "test"})
    assert test_query.cache_info().hits == 0
    await test_query({"query": "test"})
    assert test_query.cache_info().hits == 1


@pytest.mark.asyncio
async def test_cache_locking():
    """Test identical queries use the same asyncio task."""
    @async_locking_cache
    async def test_query(query):
        return "Test"
    # queries 1 and 2 should use same task
    query1 = {"query": "test"}
    task1_coro = test_query(query1)
    query2 = {"query": "test"}
    task2_coro = test_query(query2)
    # query 3 should have different task name
    query3 = {"query": "test3"}
    # cache isn't updated until the function is awaited.
    await task1_coro
    await test_query(query3)
    task1 = test_query.cache.get(json.dumps(((query1,), {})))
    task2 = test_query.cache.get(json.dumps(((query2,), {})))
    task3 = test_query.cache.get(json.dumps(((query3,), {})))
    assert task1.get_name() == task2.get_name()
    assert task1.get_name() != task3.get_name()
    await task2_coro
