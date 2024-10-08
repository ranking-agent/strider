"""Node Normalizer Utilities."""

from collections import namedtuple
import httpx
import logging
import uuid

from reasoner_pydantic import Message

from .utils import (
    StriderRequestError,
    post_json,
    get_curies,
)
from .config import settings


Entity = namedtuple(
    "Entity", ["categories", "identifiers", "information_content", "preferred_curie"]
)


class Normalizer:
    """Node Normalizer."""

    def __init__(
        self,
        logger: logging.Logger,
    ):
        """Initialize."""
        self.logger = logger
        self.curie_map = dict()

    async def get_types(self, curies):
        """Get types for a given curie"""

        try:
            results = await post_json(
                f"{settings.normalizer_url}/get_normalized_nodes",
                {
                    "curies": curies,
                    "conflate": True,
                    "drug_chemical_conflate": True,
                },
                self.logger,
                "Node Normalizer",
            )
        except StriderRequestError:
            return []

        types = []
        for c in curies:
            if results.get(c) is None:
                self.logger.warning(f"Normalizer knows nothing about {c}")
                continue
            types.extend(results[c]["type"])
        return types

    async def load_message(self, message: Message):
        """Load map for concepts in message."""
        curies = get_curies(message)
        await self.load_curies(*curies)

    async def load_curies(self, *curies: list[str]):
        """Load CURIES into map."""
        # get all curie synonyms
        try:
            response = await post_json(
                f"{settings.normalizer_url}/get_normalized_nodes",
                {
                    "curies": curies,
                    "conflate": True,
                    "drug_chemical_conflate": True,
                },
                self.logger,
                "Node Normalizer",
            )
        except StriderRequestError:
            return
        except Exception as e:
            self.logger.error(e)

        entities = [
            Entity(
                entity["type"],
                [synonym["identifier"] for synonym in entity["equivalent_identifiers"]],
                entity.get("information_content", 101),
                entity["id"]["identifier"],
            )
            for entity in response.values()
            if entity
        ]
        self.curie_map |= {
            curie: entity for entity in entities for curie in entity.identifiers
        }

    def __getitem__(self, curie: str):
        """Get preferred curie."""
        return self.curie_map[curie]

    def map(
        self,
        curies: list[str],
        prefixes: dict[str, list[str]],
    ):
        """Generate CURIE map."""
        return {curie: self.map_curie(curie, prefixes) for curie in curies}

    def map_curie(
        self,
        curie: str,
        prefixes: dict[str, list[str]],
    ) -> str:
        """Map a single CURIE to the list of preferred equivalent CURIES.

        1. Find the most-preferred prefix for which the provided CURIE has synonyms.
        2. Return all synonymous CURIEs that have that prefix.
        """
        try:
            categories, identifiers, _, preferred_curie = self.curie_map[curie]
        except KeyError:
            return [curie]
        # Gather the preferred prefixes for each category, deduplicating while retaining order
        prefixes = list(
            dict.fromkeys(
                prefix
                for category in categories
                for prefix in prefixes.get(category, [])
            )
        )
        if not prefixes:
            # There are no preferred prefixes for these categories - use the prefixes that Biolink prefers
            self.logger.debug(
                "[{}] Cannot find preferred prefixes for at least one of: {}".format(
                    getattr(self.logger, "context", ""),
                    categories,
                )
            )
            prefixes = identifiers[0].split(":")[0]

        # Find CURIEs beginning with the most-preferred prefix
        nn_preferred_prefix = preferred_curie.split(":")[0]
        if nn_preferred_prefix in prefixes:
            return [preferred_curie]
        # prefixes aren't necessarily ordered by preference
        for prefix in prefixes:
            curies = [_curie for _curie in identifiers if _curie.startswith(prefix)]
            if curies:
                return curies

        # There is no equivalent CURIE with any of the acceptable prefixes - return the original CURIE
        self.logger.debug(
            "[{}] Cannot find identifier in {} with a preferred prefix in {}".format(
                getattr(self.logger, "context", ""),
                identifiers,
                prefixes,
            ),
        )
        return [curie]

    async def get_mcq_uuid(self, curies: list[str]) -> str:
        """Get the MCQ uuid from NN."""
        response = {}
        try:
            async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
                self.logger.debug("Sending request to NN for MCQ setid.")
                res = await client.get(
                    f"{settings.normalizer_url}/get_setid",
                    params={
                        "curie": curies,
                        "conflation": [
                            "GeneProtein",
                            "DrugChemical",
                        ],
                    },
                )
                res.raise_for_status()
                response = res.json()
        except Exception as e:
            self.logger.error(f"Normalizer MCQ setid failed with: {e}")

        return response.get("setid", f"uuid:unknown-{str(uuid.uuid4())}")
