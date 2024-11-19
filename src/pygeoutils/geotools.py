"""Some utilities for manipulating GeoSpatial data."""

from __future__ import annotations

import contextlib
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

import cytoolz.curried as tlz
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pyproj
import shapely
from shapely import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, ops

from pygeoutils.exceptions import (
    InputRangeError,
    InputTypeError,
    InputValueError,
    MatchingCRSError,
    MissingColumnError,
)

FloatArray = npt.NDArray[np.float64]

if TYPE_CHECKING:
    from collections.abc import Sequence

    POLYTYPE = Union[Polygon, MultiPolygon, tuple[float, float, float, float]]
    GTYPE = Union[
        Point,
        MultiPoint,
        Polygon,
        MultiPolygon,
        LineString,
        MultiLineString,
        "tuple[float, float, float, float]",
        "list[float]",
        "list[tuple[float, float]]",
    ]
    GDFTYPE = TypeVar("GDFTYPE", gpd.GeoDataFrame, gpd.GeoSeries)
    CRSTYPE = Union[int, str, pyproj.CRS]
    GEOM = TypeVar(
        "GEOM",
        Point,
        MultiPoint,
        Polygon,
        MultiPolygon,
        LineString,
        MultiLineString,
        "tuple[float, float, float, float]",
        "list[float]",
        "list[tuple[float, float]]",
    )
    NUMBER = Union[int, float, np.number]  # pyright: ignore[reportMissingTypeArgument]

__all__ = [
    "Coordinates",
    "break_lines",
    "coords_list",
    "geo2polygon",
    "geometry_list",
    "geometry_reproject",
    "multi2poly",
    "nested_polygons",
    "query_indices",
    "snap2nearest",
]


@dataclass
class Coordinates:
    """Generate validated and normalized coordinates in WGS84.

    Parameters
    ----------
    lon : float or list of floats
        Longitude(s) in decimal degrees.
    lat : float or list of floats
        Latitude(s) in decimal degrees.
    bounds : tuple of length 4, optional
        The bounding box to check of the input coordinates fall within.
        Defaults to WGS84 bounds.

    Examples
    --------
    >>> c = Coordinates([460, 20, -30], [80, 200, 10])
    >>> c.points.x.tolist()
    [100.0, -30.0]
    """

    lon: NUMBER | Sequence[NUMBER]
    lat: NUMBER | Sequence[NUMBER]
    bounds: tuple[float, float, float, float] | None = None

    @staticmethod
    def __box_geo(bounds: tuple[float, float, float, float] | None) -> Polygon:
        """Get EPSG:4326 CRS."""
        wgs84_bounds = pyproj.CRS(4326).area_of_use.bounds  # pyright: ignore[reportOptionalMemberAccess]
        if bounds is None:
            return shapely.box(*wgs84_bounds)

        if len(bounds) != 4:
            raise InputTypeError("bounds", "tuple of length 4")

        try:
            bbox = shapely.box(*bounds)
        except (TypeError, AttributeError, ValueError) as ex:
            raise InputTypeError("bounds", "tuple of length 4") from ex

        if not bbox.within(shapely.box(*wgs84_bounds)):
            raise InputRangeError("bounds", "within EPSG:4326")
        return bbox

    @staticmethod
    def __validate(pts: gpd.GeoSeries, bbox: Polygon) -> gpd.GeoSeries:
        """Create a ``geopandas.GeoSeries`` from valid coords within a bounding box."""
        return pts[pts.sindex.query(bbox)].sort_index()  # pyright: ignore[reportReturnType]

    def __post_init__(self) -> None:
        """Normalize the longitude value within [-180, 180)."""
        if isinstance(self.lon, (int, float, np.number)):
            _lon = np.array([self.lon], "f8")
        else:
            _lon = np.array(self.lon, "f8")

        if isinstance(self.lat, (int, float, np.number)):
            lat = np.array([self.lat], "f8")
        else:
            lat = np.array(self.lat, "f8")

        lon = np.mod(np.mod(_lon, 360.0) + 540.0, 360.0) - 180.0
        pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326)
        self._points = self.__validate(pts, self.__box_geo(self.bounds))

    @property
    def points(self) -> gpd.GeoSeries:
        """Get validate coordinate as a ``geopandas.GeoSeries``."""
        return self._points


def geometry_reproject(geom: GEOM, in_crs: CRSTYPE, out_crs: CRSTYPE) -> GEOM:
    """Reproject a geometry to another CRS.

    Parameters
    ----------
    geom : list or tuple or any shapely.GeometryType
        Input geometry could be a list of coordinates such as ``[(x1, y1), ...]``,
        a bounding box like so ``(xmin, ymin, xmax, ymax)``, or any valid ``shapely``'s
        geometry such as ``Polygon``, ``MultiPolygon``, etc..
    in_crs : str, int, or pyproj.CRS
        Spatial reference of the input geometry
    out_crs : str, int, or pyproj.CRS
        Target spatial reference

    Returns
    -------
    same type as the input geometry
        Transformed geometry in the target CRS.

    Examples
    --------
    >>> from shapely import Point
    >>> point = Point(-7766049.665, 5691929.739)
    >>> geometry_reproject(point, 3857, 4326).xy
    (array('d', [-69.7636111130079]), array('d', [45.44549114818127]))
    >>> bbox = (-7766049.665, 5691929.739, -7763049.665, 5696929.739)
    >>> geometry_reproject(bbox, 3857, 4326)
    (-69.7636111130079, 45.44549114818127, -69.73666165448431, 45.47699468552394)
    >>> coords = [(-7766049.665, 5691929.739)]
    >>> geometry_reproject(coords, 3857, 4326)
    [(-69.7636111130079, 45.44549114818127)]
    """
    project = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True).transform

    if isinstance(
        geom,
        (
            Polygon,
            LineString,
            MultiLineString,
            MultiPolygon,
            Point,
            MultiPoint,
        ),
    ):
        if pyproj.CRS(in_crs) == pyproj.CRS(out_crs):
            return geom
        return ops.transform(project, geom)

    with contextlib.suppress(TypeError, AttributeError, ValueError):
        if len(geom) > 4:
            raise TypeError
        if pyproj.CRS(in_crs) == pyproj.CRS(out_crs):
            bbox = shapely.box(*geom)  # pyright: ignore[reportArgumentType,reportCallIssue]
        else:
            bbox = ops.transform(project, shapely.box(*geom))  # pyright: ignore[reportArgumentType,reportCallIssue]
            bbox = cast("Polygon", bbox)
        return tuple(float(p) for p in bbox.bounds)  # pyright: ignore[reportReturnType]

    with contextlib.suppress(TypeError, AttributeError, ValueError):
        if pyproj.CRS(in_crs) == pyproj.CRS(out_crs):
            point = Point(geom)
        else:
            point = ops.transform(project, Point(geom))
        return [(float(point.x), float(point.y))]  # pyright: ignore[reportReturnType]

    with contextlib.suppress(TypeError, AttributeError, ValueError):
        if pyproj.CRS(in_crs) == pyproj.CRS(out_crs):
            mp = MultiPoint(geom)  # pyright: ignore[reportArgumentType]
        else:
            mp = ops.transform(project, MultiPoint(geom))  # pyright: ignore[reportArgumentType]
        return [(float(p.x), float(p.y)) for p in mp.geoms]  # pyright: ignore[reportReturnType]

    gtypes = " ".join(
        (
            "a list of coordinates such as [(x1, y1), ...],",
            "a bounding box like so (xmin, ymin, xmax, ymax),",
            "or any valid shapely's geometry.",
        )
    )
    raise InputTypeError("geom", gtypes)


def geo2polygon(
    geometry: POLYTYPE,
    geo_crs: CRSTYPE | None = None,
    crs: CRSTYPE | None = None,
) -> Polygon | MultiPolygon:
    """Convert a geometry to a Shapely's Polygon and transform to any CRS.

    Parameters
    ----------
    geometry : Polygon or tuple of length 4
        Polygon or bounding box (west, south, east, north).
    geo_crs : int, str, or pyproj.CRS, optional
        Spatial reference of the input geometry, defaults to ``None``.
    crs : int, str, or pyproj.CRS
        Target spatial reference, defaults to ``None``.

    Returns
    -------
    shapely.Polygon or shapely.MultiPolygon
        A (Multi)Polygon in the target CRS, if different from the input CRS.
    """
    if isinstance(geometry, (Polygon, MultiPolygon)):
        geom = geometry
    elif isinstance(geometry, (tuple, list)) and len(geometry) == 4:
        geom = shapely.box(*geometry)  # pyright: ignore[reportArgumentType]
    else:
        raise InputTypeError("geometry", "(Multi)Polygon or tuple of length 4")

    if geo_crs is not None and crs is not None:
        geom = geometry_reproject(geom, geo_crs, crs)  # pyright: ignore[reportArgumentType]
    if not geom.is_valid:
        geom = geom.buffer(0.0)
        geom = cast("Polygon | MultiPolygon", geom)
    return geom


def snap2nearest(lines_gdf: GDFTYPE, points_gdf: GDFTYPE, tol: float) -> GDFTYPE:
    """Find the nearest points on a line to a set of points.

    Parameters
    ----------
    lines_gdf : geopandas.GeoDataFrame or geopandas.GeoSeries
        Lines.
    points_gdf : geopandas.GeoDataFrame or geopandas.GeoSeries
        Points to snap to lines.
    tol : float, optional
        Tolerance for snapping points to the nearest lines in meters.
        It must be greater than 0.0.

    Returns
    -------
    geopandas.GeoDataFrame or geopandas.GeoSeries
        Points snapped to lines.
    """
    if lines_gdf.crs != points_gdf.crs:
        raise MatchingCRSError

    if isinstance(points_gdf, gpd.GeoSeries):
        pts = points_gdf.to_frame("geometry").reset_index()
    else:
        pts = points_gdf.copy()

    pts = cast("gpd.GeoDataFrame", pts)
    cols = list(pts.columns)
    cols.remove("geometry")
    pts_idx, ln_idx = lines_gdf.sindex.query(pts.buffer(tol))
    merged_idx = tlz.merge_with(list, ({p: f} for p, f in zip(pts_idx, ln_idx)))
    _pts = {
        pi: (
            *pts.iloc[pi][cols],
            ops.nearest_points(
                shapely.unary_union(lines_gdf.iloc[fi].geometry), pts.iloc[pi].geometry
            )[0],
        )
        for pi, fi in merged_idx.items()
    }
    pts = gpd.GeoDataFrame.from_dict(_pts, orient="index")
    pts.columns = [*cols, "geometry"]
    pts = pts.set_geometry("geometry", crs=points_gdf.crs)
    pts = cast("gpd.GeoDataFrame", pts)

    if isinstance(points_gdf, gpd.GeoSeries):
        return pts.geometry
    return pts


def break_lines(lines: GDFTYPE, points: gpd.GeoDataFrame, tol: float = 0.0) -> GDFTYPE:
    """Break lines at specified points at given direction.

    Parameters
    ----------
    lines : geopandas.GeoDataFrame
        Lines to break at intersection points.
    points : geopandas.GeoDataFrame
        Points to break lines at. It must contain a column named ``direction``
        with values ``up`` or ``down``. This column is used to determine which
        part of the lines to keep, i.e., upstream or downstream of points.
    tol : float, optional
        Tolerance for snapping points to the nearest lines in meters.
        The default is 0.0.

    Returns
    -------
    geopandas.GeoDataFrame
        Original lines except for the parts that have been broken at the specified
        points.
    """
    if "direction" not in points.columns:
        raise MissingColumnError(["direction"])

    if (points.direction == "up").sum() + (points.direction == "down").sum() != len(points):
        raise InputValueError("direction", ["up", "down"])

    if not lines.geom_type.isin(["LineString", "MultiLineString"]).all():
        raise InputTypeError("geometry", "LineString or MultiLineString")

    crs_proj = lines.crs
    if tol > 0.0:
        points = snap2nearest(lines, points, tol)  # pyright: ignore[reportAssignmentType,reportArgumentType]

    mlines = lines.geom_type == "MultiLineString"
    if mlines.any():
        lines.loc[mlines, "geometry"] = lines.loc[mlines, "geometry"].apply(lambda g: list(g.geoms))
        lines = lines.explode("geometry").set_crs(crs_proj)  # pyright: ignore[reportAssignmentType,reportArgumentType]

    lines = lines.reset_index(drop=True)  # pyright: ignore[reportAssignmentType]
    points = points.reset_index(drop=True)  # pyright: ignore[reportAssignmentType]
    pts_idx, flw_idx = lines.sindex.query(points.geometry, predicate="intersects")
    if len(pts_idx) == 0:
        msg = "No intersection between lines and points"
        raise ValueError(msg)

    flw_geom = lines.iloc[flw_idx].geometry
    pts_geom = points.iloc[pts_idx].geometry
    pts_dir = points.iloc[pts_idx].direction
    idx = lines.iloc[flw_idx].index
    broken_lines = gpd.GeoSeries(
        [
            ops.substring(fl, *((0, fl.project(pt)) if d == "up" else (fl.project(pt), fl.length)))
            for fl, pt, d in zip(flw_geom, pts_geom, pts_dir)
        ],
        crs=crs_proj,
        index=idx,
    )
    return gpd.GeoDataFrame(  # pyright: ignore[reportReturnType]
        lines.loc[idx].drop(columns="geometry"), geometry=broken_lines, crs=crs_proj
    ).to_crs(lines.crs)


def geometry_list(geometry: GTYPE) -> list[LineString | Point | Polygon]:
    """Convert input geometry to a list of Polygons, Points, or LineStrings.

    Parameters
    ----------
    geometry : Polygon or MultiPolygon or tuple of length 4 or list of tuples of length 2 or 3
        Input geometry could be a ``(Multi)Polygon``, ``(Multi)LineString``,
        ``(Multi)Point``, a tuple/list of length 4 (west, south, east, north),
        or a list of tuples of length 2 or 3.

    Returns
    -------
    list
        A list of Polygons, Points, or LineStrings.
    """
    if isinstance(geometry, (Polygon, LineString, Point)):
        return [geometry]

    if isinstance(geometry, (MultiPolygon, MultiLineString, MultiPoint)):
        return list(geometry.geoms)

    if (
        isinstance(geometry, (tuple, list, np.ndarray))
        and len(geometry) == 4
        and all(isinstance(i, (float, int, np.number)) for i in geometry)
    ):
        return [shapely.box(*geometry)]  # pyright: ignore[reportArgumentType,reportCallIssue]

    with contextlib.suppress(TypeError, AttributeError):
        return list(MultiPoint(geometry).geoms)  # pyright: ignore[reportArgumentType]

    valid_geoms = (
        "Polygon",
        "MultiPolygon",
        "tuple/list of length 4",
        "list of tuples of length 2 or 3",
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
    )
    raise InputTypeError("geometry", ", ".join(valid_geoms))


def query_indices(
    tree_gdf: gpd.GeoDataFrame | gpd.GeoSeries,
    input_gdf: gpd.GeoDataFrame | gpd.GeoSeries,
    predicate: str = "intersects",
) -> dict[Any, list[Any]]:
    """Find the indices of the input_geo that intersect with the tree_geo.

    Parameters
    ----------
    tree_gdf : geopandas.GeoDataFrame or geopandas.GeoSeries
        The tree geodataframe.
    input_gdf : geopandas.GeoDataFrame or geopandas.GeoSeries
        The input geodataframe.
    predicate : str, optional
        The predicate to use for the query operation, defaults to ``intesects``.

    Returns
    -------
    dict
        A dictionary of the indices of the ``input_gdf`` that intersect with the
        ``tree_gdf``. Keys are the index of ``input_gdf`` and values are a list
        of indices of the intersecting ``tree_gdf``.
    """
    if input_gdf.crs != tree_gdf.crs:
        raise MatchingCRSError

    in_iloc, tr_iloc = tree_gdf.sindex.query(input_gdf.geometry, predicate=predicate)
    idx_dict = defaultdict(set)
    for ii, it in zip(input_gdf.iloc[in_iloc].index, tree_gdf.iloc[tr_iloc].index):
        idx_dict[ii].add(it)
    return {k: list(v) for k, v in idx_dict.items()}


def nested_polygons(gdf: gpd.GeoDataFrame | gpd.GeoSeries) -> dict[int | str, list[int | str]]:
    """Get nested polygons in a GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or geopandas.GeoSeries
        A GeoDataFrame or GeoSeries with (multi)polygons.

    Returns
    -------
    dict
        A dictionary where keys are indices of larger polygons and
        values are a list of indices of smaller polygons that are
        contained within the larger polygons.
    """
    if not gdf.geom_type.str.contains("Polygon").all():
        raise InputTypeError("gdf", "dataframe with (Multi)Polygons")

    centroid = gdf.centroid
    nested_idx = query_indices(centroid, gdf, "contains")
    nested_idx = {k: list(set(v).difference({k})) for k, v in nested_idx.items()}
    nested_idx = {k: v for k, v in nested_idx.items() if v}
    nidx = {tuple({*v, k}) for k, v in nested_idx.items()}
    area = gdf.area
    nested_keys = [area.loc[list(i)].idxmax() for i in nidx]
    nested_idx = {k: v for k, v in nested_idx.items() if k in nested_keys}
    return nested_idx


def coords_list(
    coords: tuple[float, float] | list[tuple[float, float]] | FloatArray,
) -> list[tuple[float, float]]:
    """Convert a single coordinate or list of coordinates to a list of coordinates.

    Parameters
    ----------
    coords : tuple of list of tuple
        Input coordinates

    Returns
    -------
    list of tuple
        List of coordinates as ``[(x1, y1), ...]``.
    """
    try:
        points = MultiPoint(coords)  # pyright: ignore[reportArgumentType]
    except (ValueError, TypeError, AttributeError):
        try:
            point = Point(coords)
        except (ValueError, TypeError, AttributeError) as ex:
            raise InputTypeError("coords", "tuple or list of tuples") from ex
        else:
            return [(float(point.x), float(point.y))]
    else:
        return [(float(p.x), float(p.y)) for p in points.geoms]


def _get_area_range(mp: MultiPolygon) -> float:
    """Get the range of areas of polygons in a multipolygon."""
    if np.isclose(mp.area, 0.0):
        return 0.0
    return np.ptp([g.area for g in mp.geoms]) / mp.area


def _get_largest(mp: MultiPolygon) -> Polygon:
    """Get the largest polygon from a multipolygon."""
    argmax = np.argmax([g.area for g in mp.geoms])
    return Polygon(mp.geoms[argmax].exterior)  # pyright: ignore[reportOptionalMemberAccess]


def multi2poly(gdf: GDFTYPE) -> GDFTYPE:
    """Convert multipolygons to polygon and fill holes, if any.

    Notes
    -----
    This function tries to convert multipolygons to polygons by
    first checking if multiploygons can be directly converted using
    their exterior boundaries. If not, will try to remove very small
    sub-polygons that their area is less than 1% of the total area
    of the multipolygon. If this fails, the original multipolygon will
    be returned.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or geopandas.GeoSeries
        A GeoDataFrame or GeoSeries with (multi)polygons. This will be
        more accurate if the CRS is projected.

    Returns
    -------
    geopandas.GeoDataFrame or geopandas.GeoSeries
        A GeoDataFrame or GeoSeries with polygons (and multipolygons).
    """
    if not isinstance(gdf, (gpd.GeoDataFrame, gpd.GeoSeries)):
        raise InputTypeError("gdf", "GeoDataFrame or GeoSeries")

    gdf_prj = cast("GDFTYPE", gdf.copy())
    if isinstance(gdf_prj, gpd.GeoSeries):
        gdf_prj = gpd.GeoDataFrame(gdf_prj.to_frame("geometry"))

    mp_idx = gdf_prj[gdf_prj.geom_type == "MultiPolygon"].index
    if mp_idx.size > 0:
        geo_mp = gdf_prj.loc[mp_idx, "geometry"]
        idx = {i: g.geoms[0] for i, g in geo_mp.geometry.items() if len(g.geoms) == 1}
        gdf_prj.loc[list(idx), "geometry"] = list(idx.values())
        if len(idx) < len(geo_mp):
            mp_idx = [i for i, g in geo_mp.items() if _get_area_range(g) >= 0.99]
            if mp_idx:
                gdf_prj.loc[mp_idx, "geometry"] = [
                    _get_largest(g) for g in gdf_prj.loc[mp_idx, "geometry"]
                ]

    if isinstance(gdf, gpd.GeoSeries):
        return gdf_prj.geometry

    return gdf_prj
