"""Some utilities for manipulating GeoSpatial data."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, Union, cast

import numpy as np
import scipy.interpolate as sci
import shapely
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from shapely import LineString, MultiLineString, Point

from pygeoutils.exceptions import (
    InputRangeError,
    InputTypeError,
    InputValueError,
)

if TYPE_CHECKING:
    import geopandas as gpd
    import pyproj
    from numpy.typing import NDArray

    GDFTYPE = TypeVar("GDFTYPE", gpd.GeoDataFrame, gpd.GeoSeries)
    CRSTYPE = Union[int, str, pyproj.CRS]
    FloatArray = NDArray[np.float64]

__all__ = [
    "GeoSpline",
    "make_spline",
    "spline_linestring",
    "smooth_linestring",
    "line_curvature",
    "anchored_smoothing",
    "smooth_multilinestring",
    "spline_curvature",
]


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


def line_curvature(
    line: LineString, k: int = 3, s: float | None = None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Compute the curvature of a LineString.

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
    k = np.clip(k, 1, x.size - 1)
    konts = np.hypot(np.diff(x), np.diff(y)).cumsum()
    konts = np.insert(konts, 0, 0)
    konts /= konts[-1]

    spl_x = sci.UnivariateSpline(konts, x, k=k, s=s, check_finite=True)
    spl_y = sci.UnivariateSpline(konts, y, k=k, s=s, check_finite=True)
    return spline_curvature(spl_x, spl_y, konts)


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
    points : geopandas.GeoDataFrame or geopandas.GeoSeries or array-like of shapely.Point
        Input points as a ``GeoDataFrame``, ``GeoSeries``, or array-like of
        ``shapely.Point``. The results will be more accurate if the CRS is projected.
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
        self,
        points: GDFTYPE | NDArray[Point],  # pyright: ignore[reportInvalidTypeForm]
        n_pts: int,
        degree: int = 3,
        smoothing: float | None = None,
    ) -> None:
        self.degree = degree
        if not (1 <= degree <= 5):
            raise InputRangeError("degree", "1 <= degree <= 5")

        self.smoothing = smoothing

        if shapely.get_type_id(points).sum() != 0:
            raise InputTypeError("points", "geometries of type shapely.Point")
        self.points = points

        if n_pts < 1:
            raise InputRangeError("n_pts", ">= 1")
        self.n_pts = n_pts

        self.x_ln, self.y_ln = shapely.get_coordinates(points).T
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

    return GeoSpline(shapely.points(line.coords), n_pts, degree, smoothing).spline


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
        line = shapely.line_merge(line)  # pyright: ignore[reportAssignmentType]

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


def anchored_smoothing(
    line: LineString, npts: int | None = None, sigma: float | None = None
) -> LineString:
    """Fit a cubic spline through a line while anchoring the ends.

    Parameters
    ----------
    line : shapey.LineString
        Line to smooth.
    npts : int, optional
        Number of points for uniform spacing of the generated spline, defaults
        to ``None``, i.e., the number of points along the original line.
    sigma : float, optional
        Standard deviation for Gaussian kernel used for filtering noise in the line
        before fitting the spline. Defaults to ``None``, i.e., no filtering.

    Returns
    -------
    numpy.ndarray
        The fitted cubic spline.
    """
    if not isinstance(line, LineString):
        raise InputTypeError("line", "LineString")

    line_coords = shapely.get_coordinates(line)
    npts_ = npts or line_coords.shape[0]
    x, y = np.asarray(line_coords).T
    # Apply Gaussian filter to smooth out the noise
    if sigma is not None:
        x = gaussian_filter1d(x, sigma)
        y = gaussian_filter1d(y, sigma)

    konts = np.hypot(np.diff(x), np.diff(y)).cumsum()
    konts = np.insert(konts, 0, 0)
    konts /= konts[-1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cs_x = CubicSpline(konts, x, bc_type=((1, 0.0), (1, 0.0)))  # pyright: ignore[reportArgumentType]
        cs_y = CubicSpline(konts, y, bc_type=((1, 0.0), (1, 0.0)))  # pyright: ignore[reportArgumentType]

    konts = np.linspace(0, 1, npts_)
    line_sm = np.c_[cs_x(konts), cs_y(konts)]
    line_sm[0], line_sm[-1] = line_coords[0], line_coords[-1]
    return LineString(line_sm)


def smooth_multilinestring(
    mline: MultiLineString, npts_list: list[int] | None = None, sigma: float | None = None
) -> MultiLineString:
    """Smooth a MultiLineString using a cubic spline.

    Parameters
    ----------
    mline : shapely.MultiLineString
        MultiLineString to smooth.
    npts_list : list of int, optional
        Number of points for uniform spacing of the generated spline, defaults
        to ``None``, i.e., the number of points along each line in the MultiLineString.
    sigma : float, optional
        Standard deviation for Gaussian kernel used for filtering noise in the line
        before fitting the spline. Defaults to ``None``, i.e., no filtering.

    Returns
    -------
    shapely.MultiLineString
        The fitted cubic spline.
    """
    if not isinstance(mline, MultiLineString):
        raise InputTypeError("mline", "MultiLineString")
    npts_list_ = npts_list or [None] * len(mline.geoms)
    if len(npts_list_) != len(mline.geoms):
        raise InputValueError("npts_list", "length of npts_list must match the number of lines")
    return MultiLineString(
        [anchored_smoothing(line, npts, sigma) for line, npts in zip(mline.geoms, npts_list_)]
    )
