"""Tests for PyGeoUtils"""
import io

import geopandas as gpd
from pygeoogc import ArcGISRESTful, ServiceURL
from shapely.geometry import LineString, Point, Polygon

import pygeoutils as geoutils
from pygeoutils import Coordinates, GeoBSpline

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


def test_break_line():
    crs_proj = "epsg:3857"
    lines = gpd.GeoSeries([LineString([[0, 0], [2, 2]])], crs=crs_proj)
    pt = Point(1, 1)
    points = gpd.GeoDataFrame({"direction": ["down"]}, geometry=[pt], crs=crs_proj, index=[0])
    lb_wo_tol = geoutils.break_lines(lines, points)
    pt = Point(0.5, 1.5)
    points = gpd.GeoDataFrame({"direction": ["up"]}, geometry=[pt], crs=crs_proj, index=[0])
    lb_w_tol = geoutils.break_lines(lines, points, tol=0.5)
    assert abs(lb_wo_tol.length.sum() - lb_w_tol.length.sum()) < SMALL

    lines = gpd.GeoDataFrame({"id": [0]}, geometry=[LineString([[0, 0], [2, 2]])], crs=crs_proj)
    lb_w_df = geoutils.break_lines(lines, points, tol=0.5)
    assert abs(lb_wo_tol.length.sum() - lb_w_df.length.sum()) < SMALL


def test_snap():
    crs_proj = "epsg:3857"
    lines = gpd.GeoSeries([LineString([[0, 0], [2, 2]])], crs=crs_proj)
    points = gpd.GeoSeries([Point(0.5, 1.5)], crs=crs_proj)
    pts = geoutils.snap2nearest(lines, points, tol=0.5)
    assert pts.geom_equals(gpd.GeoSeries([Point(1, 1)], crs=crs_proj)).sum() == 1


def test_coords():
    c = Coordinates([460, 20, -30], [80, 200, 10])
    assert c.points.x.tolist() == [100.0, -30.0]


def test_bspline():
    xl, yl = zip(
        *[
            (-97.06138, 32.837),
            (-97.06133, 32.836),
            (-97.06124, 32.834),
            (-97.06127, 32.832),
        ]
    )
    pts = gpd.GeoSeries(gpd.points_from_xy(xl, yl, crs="epsg:4326")).to_crs("epsg:3857")
    sp = GeoBSpline(pts, 10).spline
    assert (
        len(sp.x) == 10
        and abs(sum(sp.y) - 38734230.680) < SMALL
        and abs(sp.phi.max() - (-1.527)) < SMALL
        and abs(sp.radius.min() - 9618.943) < SMALL
    )


def test_json2geodf():
    geom = [
        (-97.06138, 32.837),
        (-97.06133, 32.836),
        (-97.06124, 32.834),
        (-97.06127, 32.832),
    ]

    service = ArcGISRESTful(ServiceURL().restful.nhdplus_epa, 2, outformat="json", crs=ALT_CRS)
    oids = service.oids_bygeom(
        geom, geo_crs=ALT_CRS, sql_clause="FTYPE NOT IN (420,428,566)", distance=1500
    )
    rjosn = service.get_features(oids, return_m=True)
    flw = geoutils.json2geodf(rjosn * 2, ALT_CRS, DEF_CRS)

    assert abs(flw.LENGTHKM.sum() - 8.917 * 2) < SMALL


def test_gtiff2array(wms_resp, cover_resp):
    canopy_box = geoutils.gtiff2xarray(wms_resp, GEO_NAT.bounds, DEF_CRS)
    canopy = geoutils.gtiff2xarray(wms_resp, GEO_NAT, DEF_CRS, drop=False)
    cover = geoutils.gtiff2xarray(cover_resp, GEO_NAT, DEF_CRS)

    mask = canopy > 60
    vec = geoutils.xarray2geodf(canopy, "float32", mask)

    expected = 72.2235
    assert (
        abs(canopy_box.mean().values.item() - expected) < SMALL
        and abs(canopy.mean().values.item() - expected) < SMALL
        and vec.shape[0] == 1014
        and cover.rio.nodata == 0
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


def test_unsupported_geo():
    resp = """
    {
        "geometry": {
            "hasM": true,
            "curveRings": [
                [
                [11, 11, 1],
                [10, 10, 2],
                [10, 11, 3],
                [11, 11, 4],
                {
                    "b": [
                    [15, 15, 2],
                    [10, 17],
                    [18, 20]
                    ]
                },
                [11, 11, 4]
                ],
                [
                [15, 15, 1],
                {
                    "c": [
                    [20, 16, 3],
                    [20, 14]
                    ]
                },
                [15, 15, 3]
                ]
            ],
            "rings":[
                [
                [11, 11, 1],
                [10, 10, 2],
                [10, 11, 3],
                [11, 11, 4]
                ],
                [
                [15, 15, 1],
                [11, 11, 2],
                [12, 15.5],
                [15.4, 17.3],
                [15, 15, 3]
                ],
                [
                [20, 16 ,1],
                [20, 14],
                [17.6, 12.5],
                [15, 15, 2],
                [20, 16, 3]
                ]
            ]
        }
    }"""
    expected = {
        "type": "Feature",
        "geometry": None,
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
