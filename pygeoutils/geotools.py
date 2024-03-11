"""Some utilities for manipulating GeoSpatial data."""

# pyright: reportGeneralTypeIssues=false
from __future__ import annotations

import contextlib
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, Tuple, TypeVar, Union, cast

import cytoolz.curried as tlz
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pyproj
import scipy.interpolate as sci
import shapely
from shapely import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, ops

from pygeoutils.exceptions import (
    InputRangeError,
    InputTypeError,
    InputValueError,
    MatchingCRSError,
    MissingColumnError,
)

BOX_ORD = "(west, south, east, north)"
FloatArray = npt.NDArray[np.float64]

if TYPE_CHECKING:
    GTYPE = Union[Polygon, MultiPolygon, Tuple[float, float, float, float]]
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
    "snap2nearest",
    "break_lines",
    "geo2polygon",
    "geometry_list",
    "Coordinates",
    "query_indices",
    "nested_polygons",
    "coords_list",
    "multi2poly",
    "GeoSpline",
    "make_spline",
    "spline_linestring",
    "smooth_linestring",
    "line_curvature",
    "spline_curvature",
    "geometry_reproject",
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
            bbox = shapely.box(*geom)  # pyright: ignore[reportArgumentType]
        else:
            bbox = cast("Polygon", ops.transform(project, shapely.box(*geom)))  # pyright: ignore[reportArgumentType]
        return tuple(float(p) for p in bbox.bounds)  # pyright: ignore[reportReturnType]

    with contextlib.suppress(TypeError, AttributeError, ValueError):
        if pyproj.CRS(in_crs) == pyproj.CRS(out_crs):
            point = Point(geom)
        else:
            point = cast("Point", ops.transform(project, Point(geom)))
        return [(float(point.x), float(point.y))]  # pyright: ignore[reportReturnType]

    with contextlib.suppress(TypeError, AttributeError, ValueError):
        if pyproj.CRS(in_crs) == pyproj.CRS(out_crs):
            mp = MultiPoint(geom)
        else:
            mp = cast("MultiPoint", ops.transform(project, MultiPoint(geom)))
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
    geometry: GTYPE,
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
        geom = geometry_reproject(geom, geo_crs, crs)
    if not geom.is_valid:
        geom = geom.buffer(0.0)
        geom = cast("Polygon | MultiPolygon", geom)
    return geom


@dataclass
class Spline:
    """Provide attributes of an interpolated Spline.

    Attributes
    ----------
    x : numpy.ndarray
        The x-coordinates of the interpolated points.
    y : numpy.ndarray
        The y-coordinates of the interpolated points.
    phi : numpy.ndarray
        Angle of the tangent of the Spline curve.
    curvature : numpy.ndarray
        Curvature of the Spline curve.
    radius : numpy.ndarray
        Radius of curvature of the Spline.
    distance : numpy.ndarray
        Total distance of each point along the Spline from the start point.
    line : shapely.LineString
        The Spline as a shapely.LineString.
    """

    x: FloatArray
    y: FloatArray
    phi: FloatArray
    curvature: FloatArray
    radius: FloatArray
    distance: FloatArray

    @property
    def line(self) -> LineString:
        """Convert the Spline to shapely.LineString."""
        return LineString(zip(self.x, self.y))


def spline_curvature(
    spline_x: sci.UnivariateSpline, spline_y: sci.UnivariateSpline, konts: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Compute the curvature of a Spline curve.

    Notes
    -----
    The formula for the curvature of a Spline curve is:

    .. math::

        \kappa = \frac{\dot{x}\ddot{y} - \ddot{x}\dot{y}}{(\dot{x}^2 + \dot{y}^2)^{3/2}}

    where :math:`\dot{x}` and :math:`\dot{y}` are the first derivatives of the
    Spline curve and :math:`\ddot{x}` and :math:`\ddot{y}` are the second
    derivatives of the Spline curve. Also, the radius of curvature is:

    .. math::

        \rho = \frac{1}{|\kappa|}

    Parameters
    ----------
    spline_x : scipy.interpolate.UnivariateSpline
        Spline curve for the x-coordinates of the points.
    spline_y : scipy.interpolate.UnivariateSpline
        Spline curve for the y-coordinates of the points.
    konts : numpy.ndarray
        Knots along the Spline curve to compute the curvature at. The knots
        must be strictly increasing.

    Returns
    -------
    phi : numpy.ndarray
        Angle of the tangent of the Spline curve.
    curvature : numpy.ndarray
        Curvature of the Spline curve.
    radius : numpy.ndarray
        Radius of curvature of the Spline curve.
    """
    if not isinstance(spline_x, sci.UnivariateSpline) or not isinstance(
        spline_y, sci.UnivariateSpline
    ):
        raise InputTypeError("spline_x/y", "scipy.interpolate.UnivariateSpline")

    dx = spline_x.derivative(1)(konts)
    dx = cast("FloatArray", dx)
    # Impose symmetric boundary conditions
    dx[0], dx[-1] = dx[1], dx[-2]
    dy = spline_y.derivative(1)(konts)
    dy = cast("FloatArray", dy)
    dy[0], dy[-1] = dy[1], dy[-2]

    phi = np.arctan2(dy, dx)

    # Check if the spline degree is high enough to compute the second derivative
    if spline_x._data[5] > 2:  # pyright: ignore[reportPrivateUsage]
        ddx = spline_x.derivative(2)(konts)
        ddx = cast("FloatArray", ddx)
        ddx[0], ddx[-1] = ddx[1], ddx[-2]
        ddy = spline_y.derivative(2)(konts)
        ddy = cast("FloatArray", ddy)
        ddy[0], ddy[-1] = ddy[1], ddy[-2]
    else:
        ddx = np.zeros_like(dx)
        ddy = np.zeros_like(dy)
    denom = np.float_power(np.square(dx) + np.square(dy), 1.5)
    denom = np.where(denom == 0, 1, denom)
    curvature = (dx * ddy - ddx * dy) / denom
    with np.errstate(divide="ignore"):
        radius = np.reciprocal(np.abs(curvature))
    return phi, curvature, radius


def line_curvature(line: LineString) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Compute the curvature of a Spline curve.

    Notes
    -----
    The formula for the curvature of a Spline curve is:

    .. math::

        \kappa = \frac{\dot{x}\ddot{y} - \ddot{x}\dot{y}}{(\dot{x}^2 + \dot{y}^2)^{3/2}}

    where :math:`\dot{x}` and :math:`\dot{y}` are the first derivatives of the
    Spline curve and :math:`\ddot{x}` and :math:`\ddot{y}` are the second
    derivatives of the Spline curve. Also, the radius of curvature is:

    .. math::

        \rho = \frac{1}{|\kappa|}

    Parameters
    ----------
    line : shapely.LineString
        Line to compute the curvature at.

    Returns
    -------
    phi : numpy.ndarray
        Angle of the tangent of the Spline curve.
    curvature : numpy.ndarray
        Curvature of the Spline curve.
    radius : numpy.ndarray
        Radius of curvature of the Spline curve.
    """
    if not isinstance(line, LineString):
        raise InputTypeError("line", "shapely.LineString")

    x, y = shapely.get_coordinates(line).T
    dx = np.diff(x)
    # Impose symmetric boundary conditions
    dx[0], dx[-1] = dx[1], dx[-2]
    dy = np.diff(y)
    dy[0], dy[-1] = dy[1], dy[-2]
    phi = np.arctan2(dy, dx)

    ddx = np.diff(dx)
    ddx = np.insert(ddx, 0, ddx[0])
    ddx[-1] = ddx[-2]
    ddy = np.diff(dy)
    ddy = np.insert(ddy, 0, ddy[0])
    ddy[-1] = ddy[-2]

    denom = np.float_power(np.square(dx) + np.square(dy), 1.5)
    denom = np.where(denom == 0, 1, denom)
    curvature = (dx * ddy - ddx * dy) / denom
    with np.errstate(divide="ignore"):
        radius = np.reciprocal(np.abs(curvature))
    return phi, curvature, radius


def make_spline(
    x: FloatArray, y: FloatArray, n_pts: int, k: int = 3, s: float | None = None
) -> Spline:
    """Create a parametric spline from a set of points.

    Parameters
    ----------
    x : numpy.ndarray
        x-coordinates of the points.
    y : numpy.ndarray
        y-coordinates of the points.
    n_pts : int
        Number of points in the output spline curve.
    k : int, optional
        Degree of the smoothing spline. Must be
        1 <= ``k`` <= 5. Default to 3 which is a cubic spline.
    s : float or None, optional
        Smoothing factor is used for determining the number of knots.
        This arg controls the tradeoff between closeness and smoothness of fit.
        Larger ``s`` means more smoothing while smaller values of ``s`` indicates
        less smoothing. If None (default), smoothing is done with all data points.

    Returns
    -------
    :class:`Spline`
        A Spline object with ``x``, ``y``, ``phi``, ``radius``, ``distance``,
        and ``line`` attributes. The ``line`` attribute returns the Spline
        as a ``shapely.LineString``.
    """
    k = np.clip(k, 1, x.size - 1)
    konts = np.hypot(np.diff(x), np.diff(y)).cumsum()
    konts = np.insert(konts, 0, 0)
    konts /= konts[-1]

    spl_x = sci.UnivariateSpline(konts, x, k=k, s=s, check_finite=True)
    spl_y = sci.UnivariateSpline(konts, y, k=k, s=s, check_finite=True)

    konts = np.linspace(0, 1, n_pts)
    x_sp = spl_x(konts)
    y_sp = spl_y(konts)
    x_sp = cast("FloatArray", x_sp)
    y_sp = cast("FloatArray", y_sp)
    phi_sp, curv_sp, rad_sp = spline_curvature(spl_x, spl_y, konts)
    d_sp = np.hypot(np.diff(x_sp), np.diff(y_sp)).cumsum()
    d_sp = np.insert(d_sp, 0, 0)
    if n_pts < 3:
        idx = np.r_[:n_pts]
        return Spline(x_sp[idx], y_sp[idx], phi_sp[idx], curv_sp[idx], rad_sp[idx], d_sp[idx])

    return Spline(x_sp, y_sp, phi_sp, curv_sp, rad_sp, d_sp)


class GeoSpline:
    """Create a parametric spline from a GeoDataFrame of points.

    Parameters
    ----------
    points : geopandas.GeoDataFrame or geopandas.GeoSeries
        Input points as a ``GeoDataFrame`` or ``GeoSeries``. The results
        will be more accurate if the CRS is projected.
    npts_sp : int
        Number of points in the output spline curve.
    degree : int, optional
        Degree of the smoothing spline. Must be
        1 <= ``degree`` <= 5. Default to 3 which is a cubic spline.
    smoothing : float or None, optional
        Smoothing factor is used for determining the number of knots.
        This arg controls the tradeoff between closeness and smoothness of fit.
        Larger ``smoothing`` means more smoothing while smaller values of
        ``smoothing`` indicates less smoothing. If None (default), smoothing
        is done with all points.

    Examples
    --------
    >>> import geopandas as gpd
    >>> xl, yl = zip(
    ...     *[
    ...         (-97.06138, 32.837),
    ...         (-97.06133, 32.836),
    ...         (-97.06124, 32.834),
    ...         (-97.06127, 32.832),
    ...     ]
    ... )
    >>> pts = gpd.GeoSeries(gpd.points_from_xy(xl, yl, crs=4326))
    >>> sp = GeoSpline(pts.to_crs(3857), 5).spline
    >>> pts_sp = gpd.GeoSeries(gpd.points_from_xy(sp.x, sp.y, crs=3857))
    >>> pts_sp = pts_sp.to_crs(4326)
    >>> list(zip(pts_sp.x, pts_sp.y))
    [(-97.06138, 32.837),
    (-97.06132, 32.83575),
    (-97.06126, 32.83450),
    (-97.06123, 32.83325),
    (-97.06127, 32.83200)]
    """

    def __init__(
        self, points: GDFTYPE, n_pts: int, degree: int = 3, smoothing: float | None = None
    ) -> None:
        self.degree = degree
        if not (1 <= degree <= 5):
            raise InputRangeError("degree", "1 <= degree <= 5")

        self.smoothing = smoothing

        if any(points.geom_type != "Point"):
            raise InputTypeError("points.geom_type", "Point")
        self.points = points

        if n_pts < 1:
            raise InputRangeError("n_pts", ">= 1")
        self.n_pts = n_pts

        self.x_ln = points.geometry.x.to_numpy("f8")
        self.y_ln = points.geometry.y.to_numpy("f8")
        self.npts_ln = self.x_ln.size
        if self.npts_ln < self.degree:
            raise InputRangeError("degree", f"< {self.npts_ln}")
        self._spline = make_spline(self.x_ln, self.y_ln, self.n_pts, self.degree, self.smoothing)

    @property
    def spline(self) -> Spline:
        """Get the spline as a ``Spline`` object."""
        return self._spline


def spline_linestring(
    line: LineString | MultiLineString,
    crs: CRSTYPE,
    n_pts: int,
    degree: int = 3,
    smoothing: float | None = None,
) -> Spline:
    """Generate a parametric spline from a LineString.

    Parameters
    ----------
    line : shapely.LineString, shapely.MultiLineString
        Line to smooth. Note that if ``line`` is ``MultiLineString``
        it will be merged into a single ``LineString``. If the merge
        fails, an exception will be raised.
    crs : int, str, or pyproj.CRS
        CRS of the input line. It must be a projected CRS.
    n_pts : int
        Number of points in the output spline curve.
    degree : int, optional
        Degree of the smoothing spline. Must be
        1 <= ``degree`` <= 5. Default to 3 which is a cubic spline.
    smoothing : float or None, optional
        Smoothing factor is used for determining the number of knots.
        This arg controls the tradeoff between closeness and smoothness of fit.
        Larger ``smoothing`` means more smoothing while smaller values of
        ``smoothing`` indicates less smoothing. If None (default), smoothing
        is done with all points.

    Returns
    -------
    :class:`Spline`
        A :class:`Spline` object with ``x``, ``y``, ``phi``, ``radius``,
        ``distance``, and ``line`` attributes. The ``line`` attribute
        returns the Spline as a shapely.LineString.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import shapely
    >>> line = shapely.LineString(
    ...     [
    ...         (-97.06138, 32.837),
    ...         (-97.06133, 32.836),
    ...         (-97.06124, 32.834),
    ...         (-97.06127, 32.832),
    ...     ]
    ... )
    >>> sp = spline_linestring(line, 4326, 5)
    >>> list(zip(*sp.line.xy))
    [(-97.06138, 32.837),
    (-97.06132, 32.83575),
    (-97.06126, 32.83450),
    (-97.06123, 32.83325),
    (-97.06127, 32.83200)]
    """
    if isinstance(line, MultiLineString):
        line = shapely.line_merge(line)

    if not isinstance(line, LineString):
        raise InputTypeError("line", "LineString")

    points = gpd.GeoSeries([Point(xy) for xy in zip(*line.xy)], crs=crs)
    return GeoSpline(points, n_pts, degree, smoothing).spline


def smooth_linestring(
    line: LineString, smoothing: float | None = None, npts: int | None = None
) -> LineString:
    """Smooth a LineString using ``UnivariateSpline`` from ``scipy``.

    Parameters
    ----------
    line : shapely.LineString
        Centerline to be smoothed.
    smoothing : float or None, optional
        Smoothing factor is used for determining the number of knots.
        This arg controls the tradeoff between closeness and smoothness of fit.
        Larger ``smoothing`` means more smoothing while smaller values of
        ``smoothing`` indicates less smoothing. If None (default), smoothing
        is done with all points.
    npts : int, optional
        Number of points in the output smoothed line. Defaults to 5 times
        the number of points in the input line.

    Returns
    -------
    shapely.LineString
        Smoothed line with uniform spacing.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import shapely
    >>> line = shapely.LineString(
    ...     [
    ...         (-97.06138, 32.837),
    ...         (-97.06133, 32.836),
    ...         (-97.06124, 32.834),
    ...         (-97.06127, 32.832),
    ...     ]
    ... )
    >>> line_smooth = smooth_linestring(line, 4326, 5)
    >>> list(zip(*line_smooth.xy))
    [(-97.06138, 32.837),
    (-97.06132, 32.83575),
    (-97.06126, 32.83450),
    (-97.06123, 32.83325),
    (-97.06127, 32.83200)]
    """
    if isinstance(line, MultiLineString):
        line = shapely.line_merge(line)

    if not isinstance(line, LineString):
        raise InputTypeError("line", "LineString")

    x, y = shapely.get_coordinates(line).T
    konts = np.hypot(np.diff(x), np.diff(y)).cumsum()
    konts = np.insert(konts, 0, 0)
    konts /= konts[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        spl_x = sci.UnivariateSpline(konts, x, k=3, s=smoothing)
        spl_y = sci.UnivariateSpline(konts, y, k=3, s=smoothing)
    _npts = npts if npts is not None else 5 * len(x)
    konts = np.linspace(0, 1, _npts)
    return LineString(np.c_[spl_x(konts), spl_y(konts)])


def snap2nearest(lines: GDFTYPE, points: GDFTYPE, tol: float) -> GDFTYPE:
    """Find the nearest points on a line to a set of points.

    Parameters
    ----------
    lines : geopandas.GeoDataFrame or geopandas.GeoSeries
        Lines.
    points : geopandas.GeoDataFrame or geopandas.GeoSeries
        Points to snap to lines.
    tol : float, optional
        Tolerance for snapping points to the nearest lines in meters.
        It must be greater than 0.0.

    Returns
    -------
    geopandas.GeoDataFrame or geopandas.GeoSeries
        Points snapped to lines.
    """
    if lines.crs != points.crs:
        raise MatchingCRSError

    if isinstance(points, gpd.GeoSeries):
        pts = points.to_frame("geometry").reset_index()
    else:
        pts = points.copy()

    pts = cast("gpd.GeoDataFrame", pts)
    cols = list(pts.columns)
    cols.remove("geometry")
    pts_idx, ln_idx = lines.sindex.query(pts.buffer(tol))
    merged_idx = tlz.merge_with(list, ({p: f} for p, f in zip(pts_idx, ln_idx)))
    _pts = {
        pi: (
            *pts.iloc[pi][cols],
            ops.nearest_points(lines.iloc[fi].geometry.unary_union, pts.iloc[pi].geometry)[0],
        )
        for pi, fi in merged_idx.items()
    }
    pts = gpd.GeoDataFrame.from_dict(_pts, orient="index")
    pts.columns = [*cols, "geometry"]
    pts = pts.set_geometry("geometry", crs=points.crs)
    pts = cast("gpd.GeoDataFrame", pts)

    if isinstance(points, gpd.GeoSeries):
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
        points = snap2nearest(lines, points, tol)  # pyright: ignore[reportArgumentType]

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


def geometry_list(geometry: GEOM) -> list[Polygon] | list[Point] | list[LineString]:
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
        return [shapely.box(*geometry)]  # pyright: ignore[reportArgumentType]

    with contextlib.suppress(TypeError, AttributeError):
        return list(MultiPoint(geometry).geoms)

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
        points = MultiPoint(coords)
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
