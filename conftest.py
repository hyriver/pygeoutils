"""Configuration for pytest."""

import pytest
from pygeoogc import WMS, ServiceURL
from shapely.geometry import Polygon

DEF_CRS = "epsg:4326"
GEO_NAT = Polygon(
    [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
)


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace):
    """Add pygeoutils namespace for doctest."""
    import pygeoutils as geoutils

    doctest_namespace["geoutils"] = geoutils


@pytest.fixture()
def wms_resp():
    """Return a WMS response."""
    wms = WMS(
        ServiceURL().wms.mrlc,
        layers="NLCD_2011_Tree_Canopy_L48",
        outformat="image/geotiff",
        crs=DEF_CRS,
    )
    return wms.getmap_bybox(
        GEO_NAT.bounds,
        1e3,
        box_crs=DEF_CRS,
    )


@pytest.fixture()
def cover_resp():
    """Return a WMS response."""
    wms = WMS(
        ServiceURL().wms.mrlc,
        layers="NLCD_2019_Land_Cover_Science_Product_L48",
        outformat="image/geotiff",
        crs=DEF_CRS,
    )
    return wms.getmap_bybox(
        GEO_NAT.bounds,
        1e3,
        box_crs=DEF_CRS,
    )
