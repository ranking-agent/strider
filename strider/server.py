"""Simple ReasonerStdAPI server."""
from typing import List, Dict

from fastapi import FastAPI
import httpx

from strider.models import Query, Message
from strider.setup_query import execute_query
from strider.neo4j import HttpInterface


def setup_tools():
    """Set up RabbitMQ and Neo4j."""
    # add strider exchange to RabbitMQ
    r = httpx.put(
        r'http://localhost:15672/api/exchanges/%2f/strider',
        json={"type": "direct", "durable": True},
        auth=('guest', 'guest'),
    )
    assert r.status_code < 300
    # add jobs queue to RabbitMQ
    r = httpx.put(
        r'http://localhost:15672/api/queues/%2f/jobs',
        json={"durable": True, "arguments": {"x-max-priority": 255}},
        auth=('guest', 'guest'),
    )
    assert r.status_code < 300
    # add results queue to RabbitMQ
    r = httpx.put(
        r'http://localhost:15672/api/queues/%2f/results',
        json={"durable": True},
        auth=('guest', 'guest'),
    )
    assert r.status_code < 300

    neo4j = HttpInterface(
        url=f'http://localhost:7474',
    )
    neo4j.run('MATCH (n) DETACH DELETE n')


setup_tools()

app = FastAPI(
    title='"Reasoner Standard API"',
    description='Boilerplate Translator KP/ARA interface',
    version='1.0.0',
)


# @app.post('/query', response_model=Message, tags=['query'])
# async def answer_query(query: Query) -> Message:
@app.post('/query', response_model=str, tags=['query'])
async def answer_query(query: Query) -> str:
    """Answer biomedical question."""
    query_id = await execute_query(query.message.query_graph.dict())
    return query_id


predicates_type = Dict[str, Dict[str, List[str]]]


@app.get('/predicates', response_model=predicates_type, tags=['predicates'])
async def get_predicates() -> predicates_type:
    """Get available predicates."""
    return {}
