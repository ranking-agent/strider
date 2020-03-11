"""Simple ReasonerStdAPI server."""
import sqlite3

import aiosqlite
from fastapi import Depends, FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware

from strider.models import Query, Message
from strider.setup_query import execute_query

app = FastAPI(
    title='Strider/ARAGORN/Ranking Agent',
    description='Translator Autonomous Relay Agent',
    version='1.0.0',
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_db():
    """Get SQLite connection."""
    async with aiosqlite.connect('results.db') as db:
        yield db


# @app.post('/query', response_model=Message, tags=['query'])
# async def answer_query(query: Query) -> Message:
@app.post('/query', response_model=str, tags=['query'])
async def answer_query(
    query: Query,
    support: bool = True,
) -> str:
    """Answer biomedical question."""
    query_id = await execute_query(query.message.query_graph.dict(), support=support)
    return query_id


@app.get('/results')
async def get_results(
        query_id: str,
        t0: float = None,
        limit: int = None,
        offset: int = 0,
        db=Depends(get_db),
):
    """Get results for a query."""
    # get column names
    statement = f'PRAGMA table_info("{query_id}")'
    cursor = await db.execute(
        statement,
    )
    results = await cursor.fetchall()
    columns = [row[1] for row in results]

    # get result rows
    statement = f'SELECT * FROM "{query_id}"'
    if t0 is not None:
        statement += f' WHERE _timestamp >= {t0}'
    statement += ' ORDER BY _timestamp ASC'
    if t0 is None:
        if limit is not None:
            statement += f' LIMIT {limit}'
        if offset:
            statement += f' OFFSET {offset}'
    try:
        cursor = await db.execute(
            statement,
        )
    except sqlite3.OperationalError as err:
        if 'no such table' in str(err):
            raise HTTPException(400, str(err))
        raise err
    results = await cursor.fetchall()

    # zip 'em up
    return [
        dict(zip(columns, row))
        for row in results
    ]
