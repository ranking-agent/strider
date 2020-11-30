"""LOTR node normalizer."""
import csv
from typing import List

from fastapi import APIRouter, FastAPI, Query
from small_kg import nodes_file, synonyms_file


def generate_category_mappings():
    """Generate CURIE -> category mappings."""
    with open(synonyms_file, newline="") as csvfile:
        reader = csv.reader(
            csvfile,
            delimiter=',',
        )
        synsets = list(reader)
    with open(nodes_file, newline="") as csvfile:
        reader = csv.reader(
            csvfile,
            delimiter=',',
        )
        nodes = list(reader)
    node_category_map = {
        node[0]: [node[1]]
        for node in nodes
    }
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


def norm_router():
    """Generate node-normalization router."""
    router = APIRouter()

    synset_mappings = generate_synset_mappings()
    category_mappings = generate_category_mappings()

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
            curie: {
                "equivalent_identifiers": [
                    {
                        "identifier": synonym
                    }
                    for synonym in synset_mappings[curie]
                ],
                "type": category_mappings[curie],
            }
            for curie in curies
        }

    return router


if __name__ == "__main__":
    import uvicorn

    app = FastAPI()
    app.include_router(norm_router())
    uvicorn.run(app, host="0.0.0.0", port=8000)
