"""Results interface."""
import sqlite3

import aiosqlite
from fastapi import HTTPException


async def get_db():
    """Get SQLite connection."""
    async with Results() as database:
        yield database


class Results():
    """Results store."""

    def __init__(self):
        """Initialize."""
        self.database = None

    async def __aenter__(self):
        """Enter async context."""
        self.database = await aiosqlite.connect('results.db')
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Exit async context."""
        # Set self.database to None first to avoid using a closing connection
        tmp_database = self.database
        self.database = None
        await tmp_database.close()
        return

    @staticmethod
    def _require_connection(method):
        """Wrap method."""
        async def wrapper(self, *args, **kwargs):
            """Check that a connection is available."""
            if not self.database:
                raise RuntimeError('No open SQLite connection.')
            return method(self, *args, **kwargs)
        return wrapper

    require_connection = _require_connection.__func__

    @require_connection
    async def get_columns(self, query_id):
        """Get columns (qnode/qedge ids)."""
        statement = f'PRAGMA table_info("{query_id}")'
        async with self.database.execute(statement) as cursor:
            columns = [row[1] for row in await cursor.fetchall()]
        return columns

    @require_connection
    async def execute(self, statement):
        """Execute statement on database.

        Return results.
        """
        try:
            cursor = await self.database.execute(statement)
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
            cursor = await self.database.executemany(statement, data)
        except sqlite3.OperationalError as err:
            if 'no such table' in str(err):
                raise HTTPException(400, str(err))
            raise err
        return await cursor.fetchall()
