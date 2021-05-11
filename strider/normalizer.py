import logging
from strider.util import post_json
import httpx


class Normalizer():
    def __init__(
            self,
            url,
            logger: logging.Logger = None,
    ):
        self.url = url
        if not logger:
            logger = logging.getLogger(__name__)
        self.logger = logger

    async def get_types(self, curies):
        """Get types for a given curie"""

        results = await post_json(
            f"{self.url}/get_normalized_nodes",
            {"curies" : curies},
            self.logger, "Node Normalizer"
        )
        if not results:
            return []

        types = []
        for c in curies:
            if results[c] is None:
                self.logger.warning(f"Normalizer knows nothing about {c}")
                continue
            types.extend(results[c]['type'])
        return types
