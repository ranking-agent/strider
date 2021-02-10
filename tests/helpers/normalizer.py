"""LOTR node normalizer."""
import csv
from typing import List

from fastapi import APIRouter, FastAPI, Query
from small_kg import synonyms_file


def generate_category_mappings():
    """Generate CURIE -> category mappings."""
    with open(synonyms_file, newline="") as csvfile:
        reader = csv.reader(
            csvfile,
            delimiter=',',
        )
        data = list(reader)
    node_category_map = {
        row[1]: [row[0]]
        for row in data
    }
    synsets = [row[1:] for row in data]
    return {
        curie: node_category_map[synset[0]]
        for synset in synsets
        for curie in synset
    }


def generate_synset_mappings():
    """Generate CURIE -> synset mappings."""
    with open(synonyms_file, newline="") as csvfile:
        reader = csv.reader(
            csvfile,
            delimiter=',',
        )
        synsets = list(reader)
    return {
        term: synset
        for synset in synsets
        for term in synset
    }


def norm_router(
        synset_mappings: dict[str, list] = None,
        category_mappings: dict[str, list] = None
):
    """Generate node-normalization router."""
    router = APIRouter()

    if not synset_mappings:
        synset_mappings = generate_synset_mappings()
    if not category_mappings:
        category_mappings = generate_category_mappings()

    def normalize_one(curie):
        """Get normalizer response for CURIE."""
        if curie not in synset_mappings:
            return None
        return {
            "equivalent_identifiers": [
                {
                    "identifier": synonym
                }
                for synonym in synset_mappings.get(curie, [])
            ],
            "type": category_mappings.get(curie, []),
        }

    @router.get("/get_normalized_nodes")
    async def normalize(
            curies: List[str] = Query(
                ...,
                alias="curie",
                example=["MONDO:0005737"],
            ),
    ):
        """Return synset for each curie."""
        return {
            curie: normalize_one(curie)
            for curie in curies
        }

    return router


if __name__ == "__main__":
    import uvicorn

    app = FastAPI()
    app.include_router(norm_router())
    uvicorn.run(app, host="0.0.0.0", port=8000)
