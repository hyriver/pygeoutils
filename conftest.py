"""Configuration for pytest."""

import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace):
    """Add pygeoutils namespace for doctest."""
    import pygeoutils as geoutils

    doctest_namespace["geoutils"] = geoutils
