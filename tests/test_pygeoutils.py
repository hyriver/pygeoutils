"""Tests for PyGeoUtils"""
import io

import pytest
from pygeoogc import WFS, WMS
from shapely.geometry import Polygon

import pygeoutils as geoutils
from pygeoutils import MatchCRS

DEF_CRS = "epsg:4326"


@pytest.fixture
def geometry_nat():
    return Polygon(
        [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
    )


@pytest.fixture
def geometry_urb():
    return Polygon(
        [
            [-118.72, 34.118],
            [-118.31, 34.118],
            [-118.31, 34.518],
            [-118.72, 34.518],
            [-118.72, 34.118],
        ]
    )


@pytest.mark.flaky(max_runs=3)
def test_gtiff2array(geometry_nat):
    url_wms = "https://www.fws.gov/wetlands/arcgis/services/Wetlands_Raster/ImageServer/WMSServer"
    wms = WMS(url_wms, layers="0", outformat="image/tiff", crs="epsg:3857",)
    r_dict = wms.getmap_bybox(geometry_nat.bounds, 1e3, box_crs=DEF_CRS,)
    geoutils.gtiff2xarray(r_dict, geometry_nat.bounds, DEF_CRS)
    wetlands = geoutils.gtiff2xarray(r_dict, geometry_nat, DEF_CRS)

    assert abs(wetlands.isel(band=0).mean().values.item() - 16.542) < 1e-3


@pytest.mark.flaky(max_runs=3)
def test_json2geodf(geometry_urb):
    url_wfs = "https://hazards.fema.gov/gis/nfhl/services/public/NFHL/MapServer/WFSServer"

    wfs = WFS(
        url_wfs,
        layer="public_NFHL:Base_Flood_Elevations",
        outformat="esrigeojson",
        crs="epsg:4269",
        version="2.0.0",
    )
    bbox = geometry_urb.bounds
    r = wfs.getfeature_bybox(bbox, box_crs=DEF_CRS)
    flood = geoutils.json2geodf([r.json(), r.json()], "epsg:4269", DEF_CRS)
    flood = flood.drop_duplicates()

    assert abs(flood["ELEV"].sum() - 630417.6) < 1e-1


def test_ring():
    ring = {
        "rings": [
            [
                [-97.06138, 32.837],
                [-97.06133, 32.836],
                [-97.06124, 32.834],
                [-97.06127, 32.832],
                [-97.06138, 32.837],
            ],
            [[-97.06326, 32.759], [-97.06298, 32.755], [-97.06153, 32.749], [-97.06326, 32.759]],
        ],
        "spatialReference": {"wkid": 4326},
    }
    _ring = geoutils.arcgis2geojson(ring)
    res = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [
                    [-97.06138, 32.837],
                    [-97.06127, 32.832],
                    [-97.06124, 32.834],
                    [-97.06133, 32.836],
                    [-97.06138, 32.837],
                ]
            ],
            [[[-97.06326, 32.759], [-97.06298, 32.755], [-97.06153, 32.749], [-97.06326, 32.759]]],
        ],
    }
    assert _ring == res


def test_point():
    point = {"x": -118.15, "y": 33.80, "z": 10.0, "spatialReference": {"wkid": 4326}}
    _point = geoutils.arcgis2geojson(point)
    res = {"type": "Point", "coordinates": [-118.15, 33.8, 10.0]}
    assert _point == res


def test_multipoint():
    mpoint = {
        "hasZ": "true",
        "points": [
            [-97.06138, 32.837, 35.0],
            [-97.06133, 32.836, 35.1],
            [-97.06124, 32.834, 35.2],
        ],
        "spatialReference": {"wkid": 4326},
    }
    _mpoint = geoutils.arcgis2geojson(mpoint)
    res = {
        "type": "MultiPoint",
        "coordinates": [
            [-97.06138, 32.837, 35.0],
            [-97.06133, 32.836, 35.1],
            [-97.06124, 32.834, 35.2],
        ],
    }
    assert _mpoint == res


def test_path():
    path = {
        "hasM": "true",
        "paths": [
            [
                [-97.06138, 32.837, 5],
                [-97.06133, 32.836, 6],
                [-97.06124, 32.834, 7],
                [-97.06127, 32.832, 8],
            ],
            [[-97.06326, 32.759], [-97.06298, 32.755]],
        ],
        "spatialReference": {"wkid": 4326},
    }
    _path = geoutils.arcgis2geojson(path)
    res = {
        "type": "MultiLineString",
        "coordinates": [
            [
                [-97.06138, 32.837, 5],
                [-97.06133, 32.836, 6],
                [-97.06124, 32.834, 7],
                [-97.06127, 32.832, 8],
            ],
            [[-97.06326, 32.759], [-97.06298, 32.755]],
        ],
    }
    assert _path == res


def test_matchcrs(geometry_urb):
    bounds = geometry_urb.bounds
    crs = "epsg:2149"
    points = ((bounds[0], bounds[2]), (bounds[1], bounds[3]))
    geom = MatchCRS.geometry(geometry_urb, DEF_CRS, crs)
    bbox = MatchCRS.bounds(geometry_urb.bounds, DEF_CRS, crs)
    coords = MatchCRS.coords(points, DEF_CRS, crs)
    assert (
        abs(geom.area - 2475726907.644) < 1e-3
        and abs(bbox[0] - (-3654031.190)) < 1e-3
        and abs(coords[0][-1] == (-2877067.244)) < 1e-3
    )


def test_show_versions():
    f = io.StringIO()
    geoutils.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
