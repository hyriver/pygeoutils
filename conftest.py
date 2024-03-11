"""Configuration for pytest."""

from __future__ import annotations

import pytest
from shapely import Polygon

from pygeoogc import WMS, ServiceURL

DEF_CRS = 4326
GEO_NAT = Polygon(
    [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
)


@pytest.fixture(autouse=True)
def _add_standard_imports(doctest_namespace):
    """Add pygeoutils namespace for doctest."""
    import pygeoutils as geoutils

    doctest_namespace["geoutils"] = geoutils


@pytest.fixture()
def wms_resp():
    """Return a WMS response."""
    return WMS(
        ServiceURL().wms.mrlc,
        layers="nlcd_tcc_conus_2011_v2021-4",
        outformat="image/geotiff",
        crs=DEF_CRS,
        validation=False,
        ssl=False,
    ).getmap_bybox(
        GEO_NAT.bounds,
        1e3,
        box_crs=DEF_CRS,
    )


@pytest.fixture()
def gtiff_list():
    """Return a WMS response."""
    return WMS(
        ServiceURL().wms.mrlc,
        layers="NLCD_2019_Land_Cover_Science_Product_L48",
        outformat="image/geotiff",
        crs=DEF_CRS,
        validation=False,
        ssl=False,
    ).getmap_bybox(
        GEO_NAT.bounds,
        1e3,
        box_crs=DEF_CRS,
        tiff_dir="cache",
    )
