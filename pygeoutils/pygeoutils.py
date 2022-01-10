"""Some utilities for manipulating GeoSpatial data."""
import contextlib
import logging
import sys
import tempfile
import uuid
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
import rasterio.features as rio_features
import rasterio.transform as rio_transform
import rioxarray as rxr
import shapely.geometry as sgeom
import ujson as json
import xarray as xr
from shapely import ops
from shapely.geometry import MultiPolygon, Polygon

from . import utils
from .exceptions import (
    EmptyResponse,
    InvalidInputType,
    InvalidInputValue,
    MissingAttribute,
    MissingCRS,
)
from .utils import Attrs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(""))
logger.handlers = [handler]
logger.propagate = False

DEF_CRS = "epsg:4326"
BOX_ORD = "(west, south, east, north)"
GTYPE = Union[Polygon, MultiPolygon, Tuple[float, float, float, float]]

__all__ = [
    "json2geodf",
    "arcgis2geojson",
    "geo2polygon",
    "get_transform",
    "xarray_geomask",
    "gtiff2xarray",
    "xarray2geodf",
    "Coordinates",
]


def json2geodf(
    content: Union[List[Dict[str, Any]], Dict[str, Any]],
    in_crs: str = DEF_CRS,
    crs: str = DEF_CRS,
) -> gpd.GeoDataFrame:
    """Create GeoDataFrame from (Geo)JSON.

    Parameters
    ----------
    content : dict or list of dict
        A (Geo)JSON dictionary e.g., response.json() or a list of them.
    in_crs : str
        CRS of the content, defaults to ``epsg:4326``.
    crs : str, optional
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


def gtiff2xarray(
    r_dict: Dict[str, bytes],
    geometry: Optional[GTYPE] = None,
    geo_crs: Optional[str] = None,
    ds_dims: Optional[Tuple[str, str]] = None,
    driver: Optional[str] = None,
    all_touched: bool = False,
    nodata: Union[Number, None] = None,
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
    geo_crs : str, optional
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
                fpath = Path(tmp_dir, f"{uuid.uuid4().hex}.nc")
                ds.to_netcdf(fpath)
                return fpath

    ds = xr.open_mfdataset(
        (to_dataset(lyr, resp) for lyr, resp in r_dict.items()),
        parallel=True,
        decode_coords="all",
    )
    ds = ds.sortby(attrs.dims[0], ascending=False)
    transform, _, _ = get_transform(ds, attrs.dims)
    ds_attrs = {
        "crs": attrs.crs.to_string(),
        "transform": utils.transform2tuple(transform),
        "res": (transform.a, transform.e),
        "bounds": utils.get_bounds(ds, attrs.dims),
    }
    for v in ds:
        ds[v] = ds[v].astype(dtypes[v])

    variables = list(ds)
    if len(variables) == 1:
        ds = ds[variables[0]].copy()
        ds.attrs.update(ds_attrs)
        ds.attrs["nodatavals"] = (nodata_dict[ds.name],)
    else:
        ds.attrs.update(ds_attrs)
        for v in variables:
            ds[v].attrs["nodatavals"] = (nodata_dict[v],)
    if geometry:
        if geo_crs is None:
            raise MissingCRS
        return xarray_geomask(ds, geometry, geo_crs, attrs.dims, all_touched, drop)
    return ds


def xarray_geomask(
    ds: Union[xr.Dataset, xr.DataArray],
    geometry: GTYPE,
    geo_crs: str,
    ds_dims: Optional[Tuple[str, str]] = None,
    all_touched: bool = False,
    drop: bool = True,
) -> Union[xr.Dataset, xr.DataArray]:
    """Mask a ``xarray.Dataset`` based on a geometry.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset(array) to be masked
    geometry : Polygon, MultiPolygon, or tuple of length 4
        The geometry or bounding box to mask the data
    geo_crs : str
        The spatial reference of the input geometry
    ds_dims : tuple of str, optional
        The names of the vertical and horizontal dimensions (in that order)
        of the target dataset, default to None. If None, dimension names are determined
        from a list of common names.
    all_touched : bool, optional
        Include a pixel in the mask if it touches any of the shapes.
        If False (default), include a pixel only if its center is within one
        of the shapes, or if it is selected by Bresenham's line algorithm.
    drop : bool, optional
        If True, drop the data outside of the extent of the mask geometries.
        Otherwise, it will return the same raster with the data masked.
        Default is True.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The input dataset with a mask applied (np.nan)
    """
    if "crs" not in ds.attrs:
        raise MissingAttribute("crs", list(ds.attrs.keys()))
    crs = ds.attrs["crs"]

    if ds_dims is None:
        ds_dims = utils.get_dim_names(ds)

    valid_dims = list(ds.sizes)
    if ds_dims is None or any(d not in valid_dims for d in ds_dims):
        raise MissingAttribute("ds_dims", valid_dims)

    transform, width, height = get_transform(ds, ds_dims)
    _geometry = geo2polygon(geometry, geo_crs, crs)
    attrs = {
        "crs": crs,
        "transform": utils.transform2tuple(transform),
        "bounds": _geometry.bounds,
        "res": (transform.a, transform.e),
    }

    _mask = rio_features.geometry_mask(
        [_geometry], (height, width), transform, invert=True, all_touched=all_touched
    )

    coords = {ds_dims[0]: ds.coords[ds_dims[0]], ds_dims[1]: ds.coords[ds_dims[1]]}
    mask = xr.DataArray(_mask, coords, dims=ds_dims)

    ds_masked = ds.where(mask, drop=drop)
    ds_masked.attrs = ds.attrs
    ds_masked.attrs.update({k: v for k, v in attrs.items() if k in ds_masked.attrs})
    if isinstance(ds_masked, xr.Dataset):
        for v in ds_masked:
            ds_masked[v].attrs = ds[v].attrs
            ds_masked[v].attrs.update(
                {key: val for key, val in attrs.items() if key in ds_masked[v].attrs}
            )
            if "nodatavals" in ds[v].attrs:
                ds_masked[v] = ds_masked[v].fillna(ds[v].attrs["nodatavals"][0])
                ds_masked[v] = ds_masked[v].astype(ds[v].dtype)
    else:
        if "nodatavals" in ds.attrs:
            ds_masked = ds_masked.fillna(ds.attrs["nodatavals"][0])
            ds_masked = ds_masked.astype(ds.dtype)
    return ds_masked


def get_gtiff_attrs(
    resp: bytes,
    ds_dims: Optional[Tuple[str, str]] = None,
    driver: Optional[str] = None,
    nodata: Union[Number, None] = None,
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
    geo_crs: str,
    crs: str,
) -> Polygon:
    """Convert a geometry to a Shapely's Polygon and transform to any CRS.

    Parameters
    ----------
    geometry : Polygon or tuple of length 4
        Polygon or bounding box (west, south, east, north).
    geo_crs : str
        Spatial reference of the input geometry
    crs : str
        Target spatial reference.

    Returns
    -------
    Polygon
        A Polygon in the target CRS.
    """
    if not isinstance(geometry, (Polygon, MultiPolygon, tuple)):
        raise InvalidInputType("geometry", "Polygon or tuple of length 4")

    if isinstance(geometry, tuple) and len(geometry) != 4:
        raise InvalidInputType("geometry", "tuple of length 4")

    geom = sgeom.box(*geometry) if isinstance(geometry, tuple) else geometry
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

    for attr in ["crs", "transform"]:
        if attr not in da.attrs:
            raise MissingAttribute(attr)

    mask = None if mask_da is None else mask_da.to_numpy()
    shapes = rio.features.shapes(
        source=da.to_numpy().astype(_dtype),
        transform=da.transform,
        mask=mask,
        connectivity=connectivity,
    )
    geometry, values = zip(*shapes)
    return gpd.GeoDataFrame(
        data={da.name: _dtype(values)},
        geometry=[sgeom.shape(g) for g in geometry],
        crs=da.crs,
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

    @property
    def points(self) -> gpd.GeoSeries:
        """Get validate coordinate as a ``geopandas.GeoSeries``."""
        return self._points
