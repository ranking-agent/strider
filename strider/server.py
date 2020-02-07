"""Simple ReasonerStdAPI server."""
from typing import List, Dict

from fastapi import FastAPI

from strider.models import Query, Message
from strider.setup_query import execute_query

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
