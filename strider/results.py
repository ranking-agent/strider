"""Results interface."""
import sqlite3

import aiosqlite
from fastapi import HTTPException


def get_db(*args, **kwargs):
    """Get dependable."""
    async def db_dependable():
        """Get SQLite connection."""
        async with Database(*args, **kwargs) as database:
            yield database
    return db_dependable


class Database():
    """Asynchronous sqlite database context manager."""

    def __init__(self, database=':memory:'):
        """Initialize."""
        self.database = database
        self.connection = None

    async def __aenter__(self):
        """Enter async context."""
        self.connection = await aiosqlite.connect(self.database)
        self.connection.row_factory = sqlite3.Row
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit async context."""
        # Set self.database to None first to avoid using a closing connection
        tmp_connection = self.connection
        self.connection = None
        await tmp_connection.close()
        return

    @staticmethod
    def _require_connection(method):
        """Wrap method."""
        async def wrapper(self, *args, **kwargs):
            """Check that a connection is available."""
            if not self.connection:
                raise RuntimeError('No open SQLite connection.')
            return await method(self, *args, **kwargs)
        return wrapper

    require_connection = _require_connection.__func__

    @require_connection
    async def get_columns(self, query_id):
        """Get columns (qnode/qedge ids)."""
        statement = f'PRAGMA table_info("{query_id}")'
        async with self.connection.execute(statement) as cursor:
            columns = [row[1] for row in await cursor.fetchall()]
        return columns

    @require_connection
    async def execute(self, statement):
        """Execute statement on database.

        Return results.
        """
        try:
            cursor = await self.connection.execute(statement)
            await self.connection.commit()
        except sqlite3.OperationalError as err:
            if 'no such table' in str(err):
                raise HTTPException(400, str(err))
            raise err
        return await cursor.fetchall()

    @require_connection
    async def executemany(self, statement, data):
        """Execute statement on database.

        Return results.
        """
        try:
            cursor = await self.connection.executemany(statement, data)
            await self.connection.commit()
        except sqlite3.OperationalError as err:
            if 'no such table' in str(err):
                raise HTTPException(400, str(err))
            raise err
        return await cursor.fetchall()
