"""Compatibility utilities."""
from collections import namedtuple
from functools import cache
import logging
import os

import httpx
from reasoner_converter.upgrading import upgrade_BiolinkEntity
from reasoner_pydantic import Message

from .util import post_json
from .trapi import apply_curie_map, get_curies
from .config import settings


class KnowledgePortal():
    """Knowledge portal."""

    def __init__(
            self,
            logger: logging.Logger = None,
    ):
        """Initialize."""
        if not logger:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.synonymizer = Synonymizer(self.logger)

    async def map_prefixes(
            self,
            message: Message,
            prefixes: dict[str, list[str]],
    ) -> Message:
        """Map prefixes."""
        await self.synonymizer.load_message(message)
        curie_map = self.synonymizer.map(prefixes)
        return apply_curie_map(message, curie_map)

    async def map_curie(
        self,
        curie: str,
        prefixes: dict[str, list[str]],
    ) -> str:
        """Map CURIE."""
        await self.synonymizer.load_curies(curie)
        curie_map = self.synonymizer.map(prefixes)
        return curie_map.get(curie, curie)

    async def fetch(
            self,
            url: str,
            request: dict,
            input_prefixes: dict = None,
            output_prefixes: dict = None,
    ):
        """Wrap fetch with CURIE mapping(s)."""
        request['message'] = await self.map_prefixes(request['message'], input_prefixes)

        response = await post_json(url, request)
        message = response["message"]

        message = await self.map_prefixes(message, output_prefixes)

        add_source(message, url)

        return message


def add_source(message: Message, source: str):
    """Add source annotation to kedges."""
    for kedge in message["knowledge_graph"]["edges"].values():
        kedge["attributes"] = [dict(
            name="provenance",
            type="MetaInformation",
            value=source,
        )]


Entity = namedtuple("Entity", ["categories", "identifiers"])


class Synonymizer():
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
        url_base = f"{settings.normalizer_url}/get_normalized_nodes"
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.get(
                url_base,
                params={"curie": list(curies)},
            )
        if response.status_code >= 300:
            try:
                response_json = response.json()
                if not response_json:
                    raise ValueError()
                detail = response_json['detail']
            except (ValueError, KeyError):
                self.logger.warning(
                    f"Failed to contact normalizer, \
                    status code {response.status_code}. \
                    Results may be incomplete.")
                return
            self.logger.warning(
                f"Error from normalizer: {detail}")
            return

        entities = [
            Entity(
                [
                    upgrade_BiolinkEntity(category)
                    for category in entity["type"]
                ],
                [
                    synonym["identifier"]
                    for synonym in entity["equivalent_identifiers"]
                ],
            )
            for entity in response.json().values() if entity
        ]
        self._data |= {
            curie: entity
            for entity in entities
            for curie in entity.identifiers
        }

    def __getitem__(self, curie: str):
        """Get preferred curie."""
        return self._data[curie]

    def map(self, prefixes: dict[str, list[str]]):
        """Generate CURIE map."""
        return CURIEMap(self, prefixes, self.logger)


class CURIEMap():
    """CURIE map."""

    def __init__(
            self,
            lookup: Synonymizer,
            prefixes: dict[str, list[str]],
            logger: logging.Logger
    ):
        """Initialize."""
        self.prefixes = prefixes
        self.lookup = lookup
        self.logger = logger

    @ cache
    def __getitem__(self, curie):
        """Get preferred curie."""
        categories, identifiers = self.lookup[curie]
        try:
            prefixes = next(
                self.prefixes[category]
                for category in categories
                if category in self.prefixes
            )
        except StopIteration:
            # no preferred prefixes for these categories
            self.logger.warning(
                "Cannot find preferred prefixes for categories")
            return curie

        # get CURIE with preferred prefix
        try:
            return next(
                curie
                for prefix in prefixes
                for curie in identifiers
                if curie.startswith(prefix)
            )
        except StopIteration:
            # no preferred curie with these prefixes
            self.logger.warning(
                "Cannot find identifier with a preferred prefix")
            return curie

    def get(self, *args):
        """Get item."""
        try:
            return self.__getitem__(args[0])
        except KeyError:
            if len(args) > 1:
                return args[1]
            raise
