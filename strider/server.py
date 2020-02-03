"""Simple ReasonerStdAPI server."""
from typing import List, Dict

from fastapi import FastAPI

from strider.models import Query, Message

app = FastAPI(
    title='"Reasoner Standard API"',
    description='Boilerplate Translator KP/ARA interface',
    version='1.0.0',
)


@app.post('/query', response_model=Message, tags=['query'])
async def answer_query(query: Query) -> Message:
    """Answer biomedical question."""
    return query.message


predicates_type = Dict[str, Dict[str, List[str]]]


@app.get('/predicates', response_model=predicates_type, tags=['predicates'])
async def get_predicates() -> predicates_type:
    """Get available predicates."""
    return {}
