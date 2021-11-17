"""Test KP registry."""
import pytest

from strider.kp_registry import Registry

from .helpers.context import with_registry_overlay
from .helpers.utils import kps_from_string


@pytest.mark.asyncio
@with_registry_overlay(
    "http://test",
    kps_from_string(
        """
    kp0 biolink:Drug biolink:treats biolink:Disease
    """
    ),
)
async def test_search_no_inverse():
    """Test search with no predicate inverses."""
    registry = Registry("http://test")
    kps = await registry.search(
        subject_categories=["biolink:Disease"],
        predicates=["biolink:no_inverse"],
        object_categories=["biolink:Drug"],
    )
    assert not kps
