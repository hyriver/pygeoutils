"""Tests for PyGeoUtils"""
import io

from pygeoogc import ArcGISRESTful, ServiceURL
from shapely.geometry import Polygon

import pygeoutils as geoutils

DEF_CRS = "epsg:4326"
ALT_CRS = "epsg:4269"
GEO_URB = Polygon(
    [
        [-118.72, 34.118],
        [-118.31, 34.118],
        [-118.31, 34.518],
        [-118.72, 34.518],
        [-118.72, 34.118],
    ]
)

GEO_NAT = Polygon(
    [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
)
SMALL = 1e-3


def test_json2geodf():
    geom = [
        (-97.06138, 32.837),
        (-97.06133, 32.836),
        (-97.06124, 32.834),
        (-97.06127, 32.832),
    ]

    service = ArcGISRESTful(ServiceURL().restful.nhdplus_epa, 2, outformat="json", crs=ALT_CRS)
    service.oids_bygeom(
        geom, geo_crs=ALT_CRS, sql_clause="FTYPE NOT IN (420,428,566)", distance=1500
    )
    rjosn = service.get_features(return_m=True)
    flw = geoutils.json2geodf(rjosn * 2, ALT_CRS, DEF_CRS)

    assert abs(flw.LENGTHKM.sum() - 8.917 * 2) < SMALL


def test_gtiff2array(wms_resp):
    canopy_box = geoutils.gtiff2xarray(wms_resp, GEO_NAT.bounds, DEF_CRS)
    canopy = geoutils.gtiff2xarray(wms_resp, GEO_NAT, DEF_CRS)

    mask = canopy > 60
    vec = geoutils.xarray2geodf(canopy, "float32", mask)

    expected = 72.042
    assert (
        abs(canopy_box.mean().values.item() - expected) < SMALL
        and abs(canopy.mean().values.item() - expected) < SMALL
        and isinstance(canopy.attrs["scales"], tuple)
        and isinstance(canopy.attrs["offsets"], tuple)
        and vec.shape[0] == 1174
    )


def test_envelope():
    resp = """
    {
      "attributes": {"OBJECTID": 5141},
      "geometry": {
      "xmin" : -109.55, "ymin" : 25.76, "xmax" : -86.39, "ymax" : 49.94,
      "zmin" : -12.0, "zmax" : 13.3,
      "spatialReference" : {"wkid" : 4326}}
    }"""

    expected = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-86.39, 49.94],
                    [-109.55, 49.94],
                    [-109.55, 25.76],
                    [-86.39, 25.76],
                    [-86.39, 49.94],
                ]
            ],
        },
        "properties": {"OBJECTID": 5141},
        "id": 5141,
    }

    assert geoutils.arcgis2geojson(resp) == expected


def test_attr_none():
    resp = """
    {
      "geometry": {
      "xmin" : -109.55, "ymin" : 25.76, "xmax" : -86.39, "ymax" : 49.94,
      "zmin" : -12.0, "zmax" : 13.3,
      "spatialReference" : {"wkid" : 4326}}
    }"""
    expected = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-86.39, 49.94],
                    [-109.55, 49.94],
                    [-109.55, 25.76],
                    [-86.39, 25.76],
                    [-86.39, 49.94],
                ]
            ],
        },
        "properties": None,
    }
    assert geoutils.arcgis2geojson(resp) == expected


def test_geometry_none():
    resp = '{"attributes": {"OBJECTID": 5141}}'
    expected = {"type": "Feature", "geometry": None, "properties": {"OBJECTID": 5141}, "id": 5141}
    assert geoutils.arcgis2geojson(resp) == expected


def test_no_id():
    resp = '{"attributes": {"OBJECTIDs": 5141}}'
    expected = {"type": "Feature", "geometry": None, "properties": {"OBJECTIDs": 5141}}
    assert geoutils.arcgis2geojson(resp) == expected


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


def test_show_versions():
    f = io.StringIO()
    geoutils.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
