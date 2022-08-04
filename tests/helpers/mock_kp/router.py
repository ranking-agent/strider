"""FastAPI router."""
import logging
from typing import Union

import aiosqlite
from fastapi import Depends, APIRouter, HTTPException
from reasoner_pydantic import Query, Response

from .build_db import add_data
from .engine import KnowledgeProvider

LOGGER = logging.getLogger(__name__)


def get_kp(
    database_file: Union[str, aiosqlite.Connection],
    **kwargs,
):
    """Get KP dependable."""

    async def kp_dependable():
        """Get knowledge provider."""
        async with KnowledgeProvider(database_file, **kwargs) as kp:
            yield kp

    return kp_dependable


def kp_router(
    database_file: Union[str, aiosqlite.Connection] = ":memory:",
    **kwargs,
):
    """Add KP to server."""
    router = APIRouter()

    @router.post("/query", response_model=Response)
    async def answer_question(
        query: Query,
    ) -> Response:
        """Get results for query graph."""
        query = query.dict(exclude_unset=True)
        workflow = query.get("workflow", [{"id": "lookup"}])
        if len(workflow) > 1:
            raise HTTPException(400, "Binder does not support workflows of length >1")
        operation = workflow[0]
        qgraph = query["message"]["query_graph"]
        if operation["id"] == "lookup":
            async with KnowledgeProvider(database_file, **kwargs) as kp:
                kgraph, results = await kp.get_results(qgraph)
        elif operation["id"] == "bind":
            kgraph = query["message"]["knowledge_graph"]
            knodes = [
                {
                    "id": knode_id,
                    "category": knode.get("categories", ["biolink:NamedThing"])[0],
                }
                for knode_id, knode in kgraph["nodes"].items()
            ]
            kedges = [
                {
                    "id": kedge_id,
                    "subject": kedge["subject"],
                    "predicate": kedge["predicate"],
                    "object": kedge["object"],
                }
                for kedge_id, kedge in kgraph["edges"].items()
            ]

            async with KnowledgeProvider(":memory:", **kwargs) as kp:
                await add_data(kp.db, knodes, kedges)
                kgraph, results = await kp.get_results(qgraph)
        else:
            raise HTTPException(400, f"Unsupported operation {operation}")

        response = {
            "message": {
                "knowledge_graph": kgraph,
                "results": results,
                "query_graph": qgraph,
            }
        }
        return Response.parse_obj(response)

    @router.get("/meta_knowledge_graph")
    async def get_metakg(
        kp: KnowledgeProvider = Depends(get_kp(database_file)),
    ):
        """Get meta knowledge graph."""
        meta_kg = {
            "edges": [
                {
                    "subject": op["subject_category"],
                    "predicate": op["predicate"],
                    "object": op["object_category"],
                }
                for op in await kp.get_operations()
            ],
            "nodes": {
                category: {"id_prefixes": data}
                for category, data in (await kp.get_curie_prefixes()).items()
            },
        }
        return meta_kg

    return router
