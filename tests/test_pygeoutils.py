"""Tests for PyGeoUtils"""
import io

import geopandas as gpd
import numpy as np
from pygeoogc import ArcGISRESTful, ServiceURL
from scipy.interpolate import make_interp_spline
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

import pygeoutils as geoutils
from pygeoutils import Coordinates, GeoBSpline

DEF_CRS = 4326
ALT_CRS = 4269
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


def assert_close(a: float, b: float, rtol: float = 1e-3) -> bool:
    assert np.isclose(a, b, rtol=rtol).all()


def test_geom_list():
    glist = geoutils.geometry_list(GEO_URB)
    assert len(glist) == 1
    glist = geoutils.geometry_list(GEO_URB.bounds)
    assert len(glist) == 1
    glist = geoutils.geometry_list(MultiPolygon([GEO_URB, GEO_NAT]))
    assert len(glist) == 2


def test_break_line():
    crs_proj = 3857
    lines = gpd.GeoSeries([LineString([[0, 0], [2, 2]])], crs=crs_proj)
    pt = Point(1, 1)
    points = gpd.GeoDataFrame({"direction": ["down"]}, geometry=[pt], crs=crs_proj, index=[0])
    lb_wo_tol = geoutils.break_lines(lines, points)
    pt = Point(0.5, 1.5)
    points = gpd.GeoDataFrame({"direction": ["up"]}, geometry=[pt], crs=crs_proj, index=[0])
    lb_w_tol = geoutils.break_lines(lines, points, tol=0.5)
    assert_close(lb_wo_tol.length.sum(), lb_w_tol.length.sum())

    lines = gpd.GeoDataFrame({"id": [0]}, geometry=[LineString([[0, 0], [2, 2]])], crs=crs_proj)
    lb_w_df = geoutils.break_lines(lines, points, tol=0.5)
    assert_close(lb_wo_tol.length.sum(), lb_w_df.length.sum())


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
    pts = gpd.GeoSeries(gpd.points_from_xy(xl, yl, crs=4326)).to_crs(3857)
    sp = GeoBSpline(pts, 10).spline
    assert len(sp.x) == 10
    assert_close(sum(sp.y), 38734230.680)
    assert_close(sp.phi.mean(), -1.552)
    assert_close(sp.radius.mean(), 75849.471)


def test_curvature():
    theta = np.linspace(0, 2 * np.pi, 20)
    rad = 10
    x = np.cos(theta) * rad
    y = np.sin(theta) * rad
    bspl = make_interp_spline(theta, np.c_[x, y], k=3)
    konts = np.linspace(0, 2 * np.pi, 100)
    phi, curvature, radius = geoutils.bspline_curvature(bspl, konts)

    # Curvature of a circle is 1/radius
    assert_close(np.mean(curvature).round(1), 1 / rad)
    assert_close(np.mean(radius).round(), rad)
    assert_close(np.mean(phi).round(), 0)

    x = np.linspace(0, 4 * np.pi, 100)
    y = np.sin(x)
    distances = np.hypot(np.diff(x), np.diff(y))
    theta = np.insert(np.cumsum(distances), 0, 0)
    bspl = make_interp_spline(theta, np.c_[x, y], k=3)

    konts = np.linspace(theta.min(), theta.max(), 200)
    _, curvature, _ = geoutils.bspline_curvature(bspl, konts)

    # Curvatures of the sine are negative at peaks and positive at troughs
    p = 50
    for i in range(4):
        assert all(np.sign(curvature)[i * p : (i + 1) * p] == (-1) ** (i + 1))


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

    assert_close(flw["lengthkm"].sum(), 8.917 * 2)


def test_gtiff2array(wms_resp, cover_resp):
    canopy_box = geoutils.gtiff2xarray(wms_resp, GEO_NAT.bounds, DEF_CRS)
    canopy = geoutils.gtiff2xarray(wms_resp, GEO_NAT, DEF_CRS, drop=False)
    cover = geoutils.gtiff2xarray(cover_resp, GEO_NAT, DEF_CRS)
    expected = 71.9547

    assert_close(canopy_box.mean().values.item(), expected)
    assert_close(canopy.mean().values.item(), expected)
    assert cover.rio.nodata == 0


def test_xarray_geodf(wms_resp):
    canopy = geoutils.gtiff2xarray(wms_resp, GEO_NAT, DEF_CRS, drop=False)

    mask = canopy > 60
    vec = geoutils.xarray2geodf(canopy, "float32", mask)
    ras = geoutils.geodf2xarray(vec, 1e3)
    ras_col = geoutils.geodf2xarray(vec, 1e3, attr_col=canopy.name, fill=np.nan)

    assert vec.shape[0] == 1081
    assert ras.sum().compute().item() == 1326
    assert_close(ras_col.mean().compute().item(), 79.6538)


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


def test_nested():
    gdf = gpd.GeoSeries(
        [box(6, 6, 8, 8), box(3, 3, 4, 4), box(2, 2, 5, 5), box(5, 5, 6, 6), box(6, 6, 7, 7)],
        crs=5070,
    )
    assert geoutils.nested_polygons(gdf) == {2: [1], 0: [4]}


def test_coords():
    coords = (1, 2)
    clist = geoutils.coords_list(coords)
    assert isinstance(clist, list) and all(len(c) == 2 for c in clist)

    coords = ((1, 2), (1, 2, 4))
    clist = geoutils.coords_list(coords)
    assert isinstance(clist, list) and all(len(c) == 2 for c in clist)

    coords = np.array(((1, 2), (1, 3)))
    clist = geoutils.coords_list(coords)
    assert isinstance(clist, list) and all(len(c) == 2 for c in clist)


def test_mp2p():
    gdf = gpd.GeoDataFrame(
        geometry=[MultiPolygon([box(6, 6, 8, 8), box(6, 6, 6.01, 6.01)]), box(2, 2, 5, 5)],
        crs=5070,
    )
    gdf = geoutils.multi2poly(gdf)
    assert (gdf.geometry.geom_type == "Polygon").all() and gdf.shape[0] == 2

    gdf = gpd.GeoDataFrame(
        geometry=[MultiPolygon([box(3, 3, 5, 5), box(6, 6, 8, 8)]), box(2, 2, 5, 5)],
        crs=5070,
    )
    gdf = geoutils.multi2poly(gdf)
    assert (gdf.geometry.geom_type == "MultiPolygon").sum() == 1

    gdf = gpd.GeoDataFrame(
        geometry=[
            MultiPolygon([box(3, 3, 5, 5), box(6, 6, 8, 8)]),
            box(8, 8, 9, 9),
            MultiPolygon([box(2, 2, 5, 5)]),
        ],
        crs=5070,
    )
    gdf = geoutils.multi2poly(gdf)
    assert (gdf.geometry.geom_type == "Polygon").sum() == 2


def test_show_versions():
    f = io.StringIO()
    geoutils.show_versions(file=f)
    assert "SYS INFO" in f.getvalue()
