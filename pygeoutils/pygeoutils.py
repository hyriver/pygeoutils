"""Some utilities for manipulating GeoSpatial data."""
import contextlib
import logging
import sys
import tempfile
import uuid
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import cytoolz as tlz
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
import rasterio.transform as rio_transform
import rioxarray as rxr
import shapely.geometry as sgeom
import ujson as json
import xarray as xr
from scipy.interpolate import BSpline
from shapely import ops
from shapely.geometry import LineString, MultiPolygon, Polygon

from . import _utils as utils
from ._utils import Attrs
from .exceptions import (
    EmptyResponse,
    InvalidInputRange,
    InvalidInputType,
    InvalidInputValue,
    MissingAttribute,
    MissingColumns,
    MissingCRS,
    UnprojectedCRS,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(""))
logger.handlers = [handler]
logger.propagate = False

DEF_CRS = "epsg:4326"
BOX_ORD = "(west, south, east, north)"
GTYPE = Union[Polygon, MultiPolygon, Tuple[float, float, float, float]]
GDF = TypeVar("GDF", gpd.GeoDataFrame, gpd.GeoSeries)
XD = TypeVar("XD", xr.Dataset, xr.DataArray)

__all__ = [
    "snap2nearest",
    "break_lines",
    "json2geodf",
    "arcgis2geojson",
    "geo2polygon",
    "get_transform",
    "xarray_geomask",
    "gtiff2xarray",
    "xarray2geodf",
    "Coordinates",
    "GeoBSpline",
]


def arcgis2geojson(
    arcgis: Union[str, Dict[str, Any]], id_attr: Optional[str] = None
) -> Dict[str, Any]:
    """Convert ESRIGeoJSON format to GeoJSON.

    Notes
    -----
    Based on `arcgis2geojson <https://github.com/chris48s/arcgis2geojson>`__.

    Parameters
    ----------
    arcgis : str or binary
        The ESRIGeoJSON format str (or binary)
    id_attr : str, optional
        ID of the attribute of interest, defaults to ``None``.

    Returns
    -------
    dict
        A GeoJSON file readable by GeoPandas.
    """
    if isinstance(arcgis, str):
        return utils.convert(json.loads(arcgis), id_attr)

    return utils.convert(arcgis, id_attr)


def json2geodf(
    content: Union[List[Dict[str, Any]], Dict[str, Any]],
    in_crs: Union[str, pyproj.CRS] = DEF_CRS,
    crs: Union[str, pyproj.CRS] = DEF_CRS,
) -> gpd.GeoDataFrame:
    """Create GeoDataFrame from (Geo)JSON.

    Parameters
    ----------
    content : dict or list of dict
        A (Geo)JSON dictionary e.g., response.json() or a list of them.
    in_crs: str or pyproj.CRS
        CRS of the content, defaults to ``epsg:4326``.
    crs: str or pyproj.CRS, optional
        The target CRS of the output GeoDataFrame, defaults to ``epsg:4326``.

    Returns
    -------
    geopandas.GeoDataFrame
        Generated geo-data frame from a GeoJSON
    """
    if not isinstance(content, (list, dict)):
        raise InvalidInputType("content", "list or list of dict ((geo)json)")

    content = content if isinstance(content, list) else [content]
    try:
        geodf = gpd.GeoDataFrame.from_features(next(iter(content)))
    except TypeError:
        content = [arcgis2geojson(c) for c in content]
        geodf = gpd.GeoDataFrame.from_features(content[0])
    except StopIteration as ex:
        raise EmptyResponse from ex

    if len(content) > 1:
        geodf = gpd.GeoDataFrame(pd.concat(gpd.GeoDataFrame.from_features(c) for c in content))

    if "geometry" in geodf and len(geodf) > 0:
        geodf = geodf.set_crs(in_crs)
        if in_crs != crs:
            geodf = geodf.to_crs(crs)

    return geodf


def xarray_geomask(
    ds: XD,
    geometry: Union[Polygon, MultiPolygon],
    crs: Union[str, pyproj.CRS],
    all_touched: bool = False,
    drop: bool = True,
    from_disk: bool = False,
) -> XD:
    """Mask a ``xarray.Dataset`` based on a geometry.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset(array) to be masked
    geometry : Polygon, MultiPolygon
        The geometry to mask the data
    crs: str or pyproj.CRS
        The spatial reference of the input geometry
    all_touched : bool, optional
        Include a pixel in the mask if it touches any of the shapes.
        If False (default), include a pixel only if its center is within one
        of the shapes, or if it is selected by Bresenham's line algorithm.
    drop : bool, optional
        If True, drop the data outside of the extent of the mask geometries.
        Otherwise, it will return the same raster with the data masked.
        Default is True.
    from_disk : bool, optional
         If True, it will clip from disk using rasterio.mask.mask if possible.
         This is beneficial when the size of the data is larger than memory.
         Default is False.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The input dataset with a mask applied (np.nan)
    """
    ds_attrs = ds.attrs
    if isinstance(ds, xr.Dataset):
        da_attrs = {v: ds[v].attrs for v in ds}

    ds = ds.rio.clip([geometry], crs=crs, all_touched=all_touched, drop=drop, from_disk=from_disk)
    ds.rio.update_attrs(ds_attrs, inplace=True)

    if isinstance(ds, xr.Dataset):
        _ = [ds[v].rio.update_attrs(da_attrs[v], inplace=True) for v in ds]
    ds.rio.update_encoding(ds.encoding, inplace=True)
    return ds


def get_gtiff_attrs(
    resp: bytes,
    ds_dims: Optional[Tuple[str, str]] = None,
    driver: Optional[str] = None,
    nodata: Optional[Number] = None,
) -> Attrs:
    """Get nodata, CRS, and dimension names in (vertical, horizontal) order from raster in bytes.

    Parameters
    ----------
    resp : bytes
        Raster response returned from a wed service request such as WMS
    ds_dims : tuple of str, optional
        The names of the vertical and horizontal dimensions (in that order)
        of the target dataset, default to None. If None, dimension names are determined
        from a list of common names.
    driver : str, optional
        A GDAL driver for reading the content, defaults to automatic detection. A list of
        the drivers can be found here: https://gdal.org/drivers/raster/index.html
    nodata : float or int, optional
        The nodata value of the raster, defaults to None, i.e., is determined from the raster.

    Returns
    -------
    Attrs
        No data, CRS, and dimension names for vertical and horizontal directions or
        a list of the existing dimensions if they are not in a list of common names.
    """
    with rio.MemoryFile() as memfile:
        memfile.write(resp)
        with memfile.open(driver=driver) as src:
            r_crs = pyproj.CRS(src.crs)
            _nodata = utils.get_nodata(src) if nodata is None else nodata

            ds = rxr.open_rasterio(src)
            if ds_dims is None:
                ds_dims = utils.get_dim_names(ds)

            valid_dims = list(ds.sizes)
            if ds_dims is None or any(d not in valid_dims for d in ds_dims):
                raise MissingAttribute("ds_dims", valid_dims)
            if isinstance(src.transform, rio.Affine):
                transform = utils.transform2tuple(src.transform)
            else:
                transform = tuple(src.transform)  # type: ignore
    return Attrs(_nodata, r_crs, ds_dims, transform)


def gtiff2xarray(
    r_dict: Dict[str, bytes],
    geometry: Optional[GTYPE] = None,
    geo_crs: Optional[str] = None,
    ds_dims: Optional[Tuple[str, str]] = None,
    driver: Optional[str] = None,
    all_touched: bool = False,
    nodata: Optional[Number] = None,
    drop: bool = True,
) -> Union[xr.DataArray, xr.Dataset]:
    """Convert (Geo)Tiff byte responses to ``xarray.Dataset``.

    Parameters
    ----------
    r_dict : dict
        Dictionary of (Geo)Tiff byte responses where keys are some names that are used
        for naming each responses, and values are bytes.
    geometry : Polygon, MultiPolygon, or tuple, optional
        The geometry to mask the data that should be in the same CRS as the r_dict.
        Defaults to ``None``.
    geo_crs: str or pyproj.CRS, optional
        The spatial reference of the input geometry, defaults to ``None``. This
        argument should be given when ``geometry`` is given.
    ds_dims : tuple of str, optional
        The names of the vertical and horizontal dimensions (in that order)
        of the target dataset, default to None. If None, dimension names are determined
        from a list of common names.
    driver : str, optional
        A GDAL driver for reading the content, defaults to automatic detection. A list of
        the drivers can be found here: https://gdal.org/drivers/raster/index.html
    all_touched : bool, optional
        Include a pixel in the mask if it touches any of the shapes.
        If False (default), include a pixel only if its center is within one
        of the shapes, or if it is selected by Bresenham's line algorithm.
    nodata : float or int, optional
        The nodata value of the raster, defaults to None, i.e., is determined from the raster.
    drop : bool, optional
        If True, drop the data outside of the extent of the mask geometries.
        Otherwise, it will return the same raster with the data masked.
        Default is True.

    Returns
    -------
    xarray.Dataset or xarray.DataAraay
        Parallel (with dask) dataset or dataarray.
    """
    if not isinstance(r_dict, dict):
        raise InvalidInputType("r_dict", "dict", '{"name": bytes}')  # noqa: FS003

    try:
        key1 = next(iter(r_dict.keys()))
    except StopIteration as ex:
        raise EmptyResponse from ex

    if "_dd_" in key1:
        var_name = {lyr: "_".join(lyr.split("_")[:-3]) for lyr in r_dict.keys()}
    else:
        var_name = dict(zip(r_dict, r_dict))

    attrs = get_gtiff_attrs(r_dict[key1], ds_dims, driver, nodata)
    dtypes: Dict[str, type] = {}
    nodata_dict: Dict[str, Number] = {}

    tmp_dir = tempfile.gettempdir()

    def to_dataset(lyr: str, resp: bytes) -> Path:
        with rio.MemoryFile() as memfile:
            memfile.write(resp)
            with memfile.open(driver=driver) as vrt:
                ds = rxr.open_rasterio(vrt)
                with contextlib.suppress(ValueError):
                    ds = ds.squeeze("band", drop=True)
                ds = ds.sortby(attrs.dims[0], ascending=False)
                ds.name = var_name[lyr]
                dtypes[ds.name] = ds.dtype
                nodata_dict[ds.name] = utils.get_nodata(vrt) if nodata is None else nodata
                ds = ds.rio.write_nodata(nodata_dict[ds.name])
                fpath = Path(tmp_dir, f"{uuid.uuid4().hex}.nc")
                ds.to_netcdf(fpath)
                return fpath

    ds = xr.open_mfdataset(
        (to_dataset(lyr, resp) for lyr, resp in r_dict.items()),
        parallel=True,
        decode_coords="all",
    )
    ds = ds.sortby(attrs.dims[0], ascending=False)

    for v in ds:
        ds[v] = ds[v].astype(dtypes[v])

    variables = list(ds)
    if len(variables) == 1:
        ds = ds[variables[0]].copy()
        ds.attrs["crs"] = attrs.crs.to_string()
        ds.attrs["nodatavals"] = (nodata_dict[ds.name],)
        ds = ds.rio.write_nodata(nodata_dict[ds.name])
        ds = ds.rio.write_crs(attrs.crs)
    else:
        ds.attrs["crs"] = attrs.crs.to_string()
        ds = ds.rio.write_crs(attrs.crs)
        for v in variables:
            ds[v].attrs["crs"] = attrs.crs.to_string()
            ds[v].attrs["nodatavals"] = (nodata_dict[v],)
            ds[v] = ds[v].rio.write_nodata(nodata_dict[v])
    if isinstance(geometry, (Polygon, MultiPolygon)):
        if geo_crs is None:
            raise MissingCRS
        return xarray_geomask(ds, geometry, geo_crs, all_touched, drop, from_disk=True)
    return ds


def get_transform(
    ds: Union[xr.Dataset, xr.DataArray],
    ds_dims: Tuple[str, str] = ("y", "x"),
) -> Tuple[rio.Affine, int, int]:
    """Get transform of a ``xarray.Dataset`` or ``xarray.DataArray``.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset(array) to be masked
    ds_dims : tuple, optional
        Names of the coordinames in the dataset, defaults to ``("y", "x")``.
        The order of the dimension names must be (vertical, horizontal).

    Returns
    -------
    rasterio.Affine, int, int
        The affine transform, width, and height
    """
    ydim, xdim = ds_dims
    height, width = ds.sizes[ydim], ds.sizes[xdim]

    left, bottom, right, top = utils.get_bounds(ds, ds_dims)

    x_res = abs(left - right) / width
    y_res = abs(top - bottom) / height

    left -= x_res * 0.5
    right += x_res * 0.5
    top += y_res * 0.5
    bottom -= y_res * 0.5

    transform = rio_transform.from_bounds(left, bottom, right, top, width, height)
    return transform, width, height


def geo2polygon(
    geometry: GTYPE,
    geo_crs: Union[str, pyproj.CRS],
    crs: Union[str, pyproj.CRS],
) -> Polygon:
    """Convert a geometry to a Shapely's Polygon and transform to any CRS.

    Parameters
    ----------
    geometry : Polygon or tuple of length 4
        Polygon or bounding box (west, south, east, north).
    geo_crs: str or pyproj.CRS
        Spatial reference of the input geometry
    crs: str or pyproj.CRS
        Target spatial reference.

    Returns
    -------
    Polygon
        A Polygon in the target CRS.
    """
    if not isinstance(geometry, (Polygon, MultiPolygon, Sequence)):
        raise InvalidInputType("geometry", "Polygon or tuple of length 4")

    if isinstance(geometry, Sequence) and len(geometry) != 4:
        raise InvalidInputType("geometry", "tuple of length 4")

    geom = sgeom.box(*geometry) if isinstance(geometry, Sequence) else geometry
    project = pyproj.Transformer.from_crs(geo_crs, crs, always_xy=True).transform
    geom = ops.transform(project, geom)
    if not geom.is_valid:
        geom = geom.buffer(0.0)
    return geom


def xarray2geodf(
    da: xr.DataArray, dtype: str, mask_da: Optional[xr.DataArray] = None, connectivity: int = 8
) -> gpd.GeoDataFrame:
    """Vectorize a ``xarray.DataArray`` to a ``geopandas.GeoDataFrame``.

    Parameters
    ----------
    da : xarray.DataArray
        The dataarray to vectorize.
    dtype : type
        The data type of the dataarray. Valid types are ``int16``, ``int32``,
        ``uint8``, ``uint16``, and ``float32``.
    mask_da : xarray.DataArray, optional
        The dataarray to use as a mask, defaults to ``None``.
    connectivity : int, optional
        Use 4 or 8 pixel connectivity for grouping pixels into features,
        defaults to 8.

    Returns
    -------
    geopandas.GeoDataFrame
        The vectorized dataarray.
    """
    if not isinstance(da, xr.DataArray):
        raise InvalidInputType("da", "xarray.DataArray")

    if not isinstance(mask_da, (xr.DataArray, type(None))):
        raise InvalidInputType("da", "xarray.DataArray or None")

    valid_types = ["int16", "int32", "uint8", "uint16", "float32"]
    if dtype not in valid_types:
        raise InvalidInputValue("dtype", valid_types)
    _dtype = getattr(np, dtype)

    crs = da.rio.crs if da.attrs.get("crs") is None else da.crs
    if crs is None:
        raise MissingCRS

    mask = None if mask_da is None else mask_da.to_numpy()
    shapes = rio.features.shapes(
        source=da.to_numpy().astype(_dtype),
        transform=da.rio.transform(),
        mask=mask,
        connectivity=connectivity,
    )
    geometry, values = zip(*shapes)
    return gpd.GeoDataFrame(
        data={da.name: _dtype(values)},
        geometry=[sgeom.shape(g) for g in geometry],
        crs=crs,
    )


@dataclass
class Coordinates:
    """Generate validated and normalized coordinates in WGS84.

    Parameters
    ----------
    lon : float or list of floats
        Longitude(s) in decimal degrees.
    lat : float or list of floats
        Latitude(s) in decimal degrees.

    Examples
    --------
    >>> from pygeoutils import Coordinates
    >>> c = Coordinates([460, 20, -30], [80, 200, 10])
    >>> c.points.x.tolist()
    [100.0, -30.0]
    """

    lon: Union[Number, Sequence[Number]]
    lat: Union[Number, Sequence[Number]]

    @property
    def _crs(self) -> pyproj.CRS:
        """Get EPSG:4326 CRS."""
        return pyproj.CRS(4326)

    @property
    def _wgs84(self) -> sgeom.Polygon:
        """Get the WGS84 bounds."""
        return sgeom.box(*self._crs.area_of_use.bounds)

    @staticmethod
    def __validate(pts: gpd.GeoSeries, bbox: sgeom.Polygon) -> gpd.GeoSeries:
        """Create a ``geopandas.GeoSeries`` from valid coords within a bounding box."""
        return pts[pts.sindex.query(bbox)].sort_index()

    def __post_init__(self) -> None:
        """Normalize the longitude value within [-180, 180)."""
        _lon = [self.lon] if isinstance(self.lon, Number) else self.lon
        lat = [self.lat] if isinstance(self.lat, Number) else self.lat
        if not isinstance(_lon, (list, tuple)) and not isinstance(lat, (list, tuple)):
            raise InvalidInputType("lon/lat", "float or list")

        lon = np.mod(np.mod(_lon, 360) + 540, 360) - 180
        pts = gpd.GeoSeries([sgeom.Point(xy) for xy in zip(lon, lat)], crs=self._crs)
        self._points = self.__validate(pts, self._wgs84)

    @property
    def points(self) -> gpd.GeoSeries:
        """Get validate coordinate as a ``geopandas.GeoSeries``."""
        return self._points


@dataclass
class Spline:
    """Provide attributes of an interpolated B-spline.

    Attributes
    ----------
    x : numpy.ndarray
        The x-coordinates of the interpolated points.
    y : numpy.ndarray
        The y-coordinates of the interpolated points.
    phi : numpy.ndarray
        Curvature of the B-spline in radians.
    radius : numpy.ndarray
        Radius of curvature of the B-spline.
    distance : numpy.ndarray
        Total distance of each point along the B-spline from the start point.
    """

    x: np.ndarray
    y: np.ndarray
    phi: np.ndarray
    radius: np.ndarray
    distance: np.ndarray


class GeoBSpline:
    """Create B-spline from a geo-dataframe of points.

    Parameters
    ----------
    points : geopandas.GeoDataFrame or geopandas.GeoSeries
        Input points as a ``GeoDataFrame`` or ``GeoSeries`` in a projected CRS.
    npts_sp : int
        Number of points in the output spline curve.
    degree : int, optional
        Degree of the spline. Should be less than the number of points and
        greater than 1. Default is 3.

    Examples
    --------
    >>> from pygeoutils import GeoBSpline
    >>> import geopandas as gpd
    >>> xl, yl = zip(
    ...     *[
    ...         (-97.06138, 32.837),
    ...         (-97.06133, 32.836),
    ...         (-97.06124, 32.834),
    ...         (-97.06127, 32.832),
    ...     ]
    ... )
    >>> pts = gpd.GeoSeries(gpd.points_from_xy(xl, yl, crs="epsg:4326"))
    >>> sp = GeoBSpline(pts.to_crs("epsg:3857"), 5).spline
    >>> pts_sp = gpd.GeoSeries(gpd.points_from_xy(sp.x, sp.y, crs="epsg:3857"))
    >>> pts_sp = pts_sp.to_crs("epsg:4326")
    >>> list(zip(pts_sp.x, pts_sp.y))
    [(-97.06138, 32.837),
    (-97.06135, 32.83629),
    (-97.06131, 32.83538),
    (-97.06128, 32.83434),
    (-97.06127, 32.83319)]
    """

    @staticmethod
    def __curvature(
        xs: Union[Sequence[float], np.ndarray], ys: Union[Sequence[float], np.ndarray], l_tot: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the curvature of a B-spline curve.

        Notes
        -----
        This function is based on `nldi-xstool <https://code.usgs.gov/wma/nhgf/toolsteam/nldi-xstool>`__.

        Parameters
        ----------
        xs : array_like
            x coordinates of the points.
        ys : array_like
            y coordinates of the points.
        l_tot : float
            Total distance of points along the B-spline from the start point.

        Returns
        -------
        tuple of array_like
            Curvature and radius of curvature.
        """
        size = len(xs)
        dx = np.diff(xs, prepend=xs[0])
        dy = np.diff(ys, prepend=ys[0])
        phi = np.zeros(size) + np.pi * 0.5 * np.sign(dy)
        nonzero = np.nonzero(dx)

        phi[nonzero] = np.arctan2(dy[nonzero], dx[nonzero])
        phi[0] = (2.0 * phi[1]) - phi[2]

        rad = np.zeros(size) + 1.0e8
        scals = l_tot / (dx.size - 1)

        dphi = np.diff(np.abs(phi), prepend=np.abs(phi[0]))
        non_small = np.where(dphi > 1e-4)[0]
        rad[non_small] = scals / dphi[non_small]
        return phi, rad

    def __spline(self, npts_sp: int, degree: int = 3) -> Spline:
        """Create a B-spline curve from a set of points.

        Notes
        -----
        This function is based on https://stackoverflow.com/a/45928473/5797702.

        Parameters
        ----------
        npts_sp : int
            Number of points in the output spline curve.
        degree : int, optional
            Degree of the spline. Should be less than the number of points and
            greater than 1. Default is 3.

        Returns
        -------
        Spline
            A Spline object with ``x``, ``y``, ``phi``, ``radius``,
            and ``distance`` attributes.
        """
        degree = np.clip(degree, 1, self.npts_ln - 1)
        konts = np.clip(np.arange(self.npts_ln + degree + 1) - degree, 0, self.npts_ln - degree)
        spl = BSpline(konts, np.column_stack([self.x_ln, self.y_ln]), degree)

        x_sp, y_sp = spl(np.linspace(0, self.npts_ln - degree, max(npts_sp, 3), endpoint=False)).T
        phi_sp, rad_sp = self.__curvature(x_sp, y_sp, self.l_ln)
        geom = (
            LineString([(x1, y1), (x2, y2)])
            for x1, y1, x2, y2 in zip(x_sp[:-1], y_sp[:-1], x_sp[1:], y_sp[1:])
        )
        d_sp = gpd.GeoSeries(geom, crs=self.crs).length.cumsum().values
        if npts_sp < 3:
            idx = np.r_[:npts_sp]
            return Spline(x_sp[idx], y_sp[idx], phi_sp[idx], rad_sp[idx], d_sp[idx])

        return Spline(x_sp, y_sp, phi_sp, rad_sp, d_sp)

    def __init__(self, points: GDF, npts_sp: int, degree: int = 3) -> None:
        self.degree = degree
        self.crs = points.crs
        if self.crs is None:
            raise MissingCRS

        if not self.crs.is_projected:
            raise InvalidInputType("points.crs", "projected CRS")

        if any(points.geom_type != "Point"):
            raise InvalidInputType("points.geom_type", "Point")
        self.points = points

        if npts_sp < 1:
            raise InvalidInputRange("npts_sp", ">= 1")
        self.npts_sp = npts_sp

        tx, ty = zip(*(g.xy for g in points.geometry))
        self.x_ln = np.array(tx, dtype="f8").squeeze()
        self.y_ln = np.array(ty, dtype="f8").squeeze()
        self.npts_ln = self.x_ln.size
        self.l_ln = LineString(points.geometry).length
        self._spline = self.__spline(npts_sp, degree)

    @property
    def spline(self) -> Spline:
        """Get the spline as a ``Spline`` object."""
        return self._spline


def snap2nearest(lines: GDF, points: GDF, tol: float) -> GDF:
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
    if lines.crs is None or points.crs is None:
        raise MissingCRS

    if not lines.crs.is_projected or not points.crs.is_projected:
        raise UnprojectedCRS

    if isinstance(points, gpd.GeoSeries):
        pts: gpd.GeoDataFrame = points.to_frame("geometry").reset_index()
    else:
        pts = points.copy()

    cols = list(pts.columns)
    cols.remove("geometry")
    pts_idx, ln_idx = lines.sindex.query_bulk(pts.buffer(tol))
    merged_idx = tlz.merge_with(list, ({p: f} for p, f in zip(pts_idx, ln_idx)))
    _pts = {
        pi: (
            *pts.iloc[pi][cols],  # type: ignore[has-type]
            ops.nearest_points(lines.iloc[fi].geometry.unary_union, pts.iloc[pi].geometry)[0],
        )
        for pi, fi in merged_idx.items()
    }
    pts = gpd.GeoDataFrame.from_dict(_pts, orient="index")
    pts.columns = cols + ["geometry"]
    pts = pts.set_geometry("geometry", crs=points.crs)

    if isinstance(points, gpd.GeoSeries):
        return pts.geometry
    return pts


def break_lines(lines: GDF, points: gpd.GeoDataFrame, tol: float = 0.0) -> GDF:
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
    if lines.crs is None or points.crs is None:
        raise MissingCRS

    if "direction" not in points.columns:
        raise MissingColumns(["direction"])

    if (points.direction == "up").sum() + (points.direction == "down").sum() != len(points):
        raise InvalidInputValue("direction", ["up", "down"])

    if not lines.geom_type.isin(["LineString", "MultiLineString"]).all():
        raise InvalidInputType("geometry", "LineString or MultiLineString")

    if lines.crs != points.crs or not lines.crs.is_projected or not points.crs.is_projected:
        crs_proj = "epsg:3857"
        lns = lines.to_crs(crs_proj)
        pts = points.to_crs(crs_proj)
    else:
        crs_proj = lines.crs
        lns = lines.copy()
        pts = points.copy()

    if tol > 0.0:
        pts = snap2nearest(lns, pts, tol)

    mlines = lns.geom_type == "MultiLineString"
    if mlines.any():
        lns.loc[mlines, "geometry"] = lns.loc[mlines, "geometry"].apply(lambda g: list(g.geoms))
        lns = lns.explode("geometry").set_crs(crs_proj)

    pts_idx, flw_idx = lns.sindex.query_bulk(pts.geometry)
    if len(pts_idx) == 0:
        raise ValueError("No intersection between lines and points")  # noqa: TC003

    flw_geom = lns.iloc[flw_idx].geometry
    pts_geom = pts.iloc[pts_idx].geometry
    pts_dir = pts.iloc[pts_idx].direction
    idx = lns.iloc[flw_idx].index
    broken_lines = gpd.GeoSeries(
        [
            ops.substring(fl, *((0, fl.project(pt)) if d == "up" else (fl.project(pt), fl.length)))
            for fl, pt, d in zip(flw_geom, pts_geom, pts_dir)
        ],
        crs=crs_proj,
        index=idx,
    )

    if isinstance(lns, gpd.GeoDataFrame):
        lns.loc[idx, "geometry"] = broken_lines
    else:
        lns.loc[idx] = broken_lines
    return lns.to_crs(lines.crs)
