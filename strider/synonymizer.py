"""Compatibility utilities."""
from collections import namedtuple
import logging

from reasoner_pydantic import Message

from .util import (
    StriderRequestError,
    post_json,
)
from .trapi import get_curies
from .config import settings


Entity = namedtuple("Entity", ["categories", "identifiers"])


class Synonymizer:
    """CURIE synonymizer."""

    def __init__(
        self,
        logger: logging.Logger,
    ):
        """Initialize."""
        self.logger = logger
        self._data = dict()

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
                {"curies": curies},
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
            )
            for entity in response.values()
            if entity
        ]
        self._data |= {
            curie: entity for entity in entities for curie in entity.identifiers
        }

    def __getitem__(self, curie: str):
        """Get preferred curie."""
        return self._data[curie]

    def map(
        self,
        curies: list[str],
        prefixes: dict[str, list[str]],
        logger: logging.Logger = None,
    ):
        """Generate CURIE map."""
        if not logger:
            logger = self.logger
        return {
            curie: self.map_curie(curie, self._data, prefixes, logger)
            for curie in curies
        }

    def map_curie(
        self,
        curie: str,
        data: dict[str, Entity],
        prefixes: dict[str, list[str]],
        logger: logging.Logger = None,
    ) -> str:
        """Map a single CURIE to the list of preferred equivalent CURIES.

        1. Find the most-preferred prefix for which the provided CURIE has synonyms.
        2. Return all synonymous CURIEs that have that prefix.
        """
        try:
            categories, identifiers = data[curie]
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
            logger.debug(
                "[{}] Cannot find preferred prefixes for at least one of: {}".format(
                    getattr(logger, "context", ""),
                    categories,
                )
            )
            prefixes = identifiers[0].split(":")[0]

        # Find CURIEs beginning with the most-preferred prefix
        for prefix in prefixes:
            curies = [_curie for _curie in identifiers if _curie.startswith(prefix)]
            if curies:
                return curies

        # There is no equivalent CURIE with any of the acceptable prefixes - return the original CURIE
        logger.debug(
            "[{}] Cannot find identifier in {} with a preferred prefix in {}".format(
                getattr(logger, "context", ""),
                identifiers,
                prefixes,
            ),
        )
        return [curie]
