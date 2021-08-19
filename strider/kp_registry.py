"""KP registry."""
from itertools import chain
import logging
from strider.util import StriderRequestError, post_json
from typing import Union

import httpx

from strider.util import WBMT


class Registry():
    """KP registry."""

    def __init__(
            self,
            url,
            logger: logging.Logger = None,
    ):
        """Initialize."""
        self.url = url
        if not logger:
            logger = logging.getLogger(__name__)
        self.logger = logger

    async def __aenter__(self):
        """Enter context."""
        return self

    async def __aexit__(self, *args):
        """Exit context."""

    async def setup(self):
        """Set up database table."""

    async def get_all(self):
        """Get all KPs."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f'{self.url}/kps',
            )
            assert response.status_code < 300
        return response.json()

    async def get_one(self, url):
        """Get a specific KP."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f'{self.url}/kps/{url}',
            )
            assert response.status_code < 300
        provider = response.json()
        return provider[5]['details']

    async def add(self, **kps):
        """Add KP(s)."""
        # kps = {
        #     kp.name: await kp.get_operations()
        #     for kp in kps
        # }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{self.url}/kps',
                json=kps
            )
            assert response.status_code < 300

    async def delete_one(self, url):
        """Delete a specific KP."""
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f'{self.url}/kps/{url}',
            )
            assert response.status_code < 300

    async def search(
            self,
            subject_categories: Union[str, list[str]] = None,
            predicates: Union[str, list[str]] = None,
            object_categories: Union[str, list[str]] = None,
            allowlist=None, denylist=None,
    ):
        """Search for KPs matching a pattern."""
        if isinstance(subject_categories, str):
            subject_categories = [subject_categories]
        if isinstance(predicates, str):
            predicates = [predicates]
        if isinstance(object_categories, str):
            object_categories = [object_categories]
        subject_categories = [desc for cat in subject_categories for desc in WBMT.get_descendants(cat)]
        predicates = [desc for pred in predicates for desc in WBMT.get_descendants(pred)]
        inverse_predicates = [
            desc
            for pred in predicates
            if (inverse := WBMT.predicate_inverse(pred))
            for desc in WBMT.get_descendants(inverse)
        ]
        object_categories = [desc for cat in object_categories for desc in WBMT.get_descendants(cat)]

        try:
            response = await post_json(
                f'{self.url}/search',
                {
                    'subject_category': subject_categories,
                    'object_category': object_categories,
                    'predicate': predicates,
                },
                self.logger, "KP Registry"
            )
        except StriderRequestError:
            return {}
        if inverse_predicates:
            try:
                inverse_response = await post_json(
                    f'{self.url}/search',
                    {
                        'subject_category': object_categories,
                        'object_category': subject_categories,
                        'predicate': inverse_predicates,
                    },
                    self.logger, "KP Registry"
                )
            except StriderRequestError:
                return {}
        else:
            inverse_response = dict()

        return {
            kpid: details
            for kpid, details in chain(*(response.items(), inverse_response.items()))
            if (
                (allowlist is None or kpid in allowlist)
                and (denylist is None or kpid not in denylist)
            )
        }

    async def delete_all(self):
        """Delete all KPs."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f'{self.url}/clear',
            )
            assert response.status_code < 300
        return response.json()
