"""LOTR node normalizer."""
import csv
from typing import List

from fastapi import APIRouter, FastAPI, Query
from pydantic.main import BaseModel


def norm_router(synset_mappings: dict[str, list], category_mappings: dict[str, list]):
    """Generate node-normalization router."""
    router = APIRouter()

    def normalize_one(curie):
        """Get normalizer response for CURIE."""
        if curie not in synset_mappings:
            return None
        return {
            "equivalent_identifiers": [
                {"identifier": synonym} for synonym in synset_mappings.get(curie, [])
            ],
            "type": category_mappings.get(curie, []),
        }

    @router.get("/get_normalized_nodes")
    async def normalize_get(
        curies: List[str] = Query(
            ...,
            alias="curie",
            example=["MONDO:0005737"],
        ),
    ):
        """Return synset for each curie."""
        return {curie: normalize_one(curie) for curie in curies}

    class CurieList(BaseModel):
        curies: list[str]

    @router.post("/get_normalized_nodes")
    async def normalize_post(
        curie_list: CurieList,
    ):
        """Return synset for each curie."""
        return {curie: normalize_one(curie) for curie in curie_list.curies}

    return router


if __name__ == "__main__":
    import uvicorn

    app = FastAPI()
    app.include_router(norm_router())
    uvicorn.run(app, host="0.0.0.0", port=8000)
