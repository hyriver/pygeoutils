"""Some utilities for manipulating GeoSpatial data."""
from __future__ import annotations

import contextlib
import itertools
from typing import TYPE_CHECKING, Any, Tuple, TypeVar, Union, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio.features as rio_features
import rasterio.transform as rio_transform
import rioxarray._io as rxr
import shapely.geometry as sgeom
import ujson as json
import xarray as xr
from pyproj.exceptions import CRSError as ProjCRSError
from rasterio import MemoryFile
from rioxarray.exceptions import OneDimensionalRaster
from shapely.geometry import MultiPolygon, Polygon

from pygeoutils import _utils as utils
from pygeoutils import geotools
from pygeoutils.exceptions import (
    EmptyResponseError,
    InputTypeError,
    InputValueError,
    MissingCRSError,
)

BOX_ORD = "(west, south, east, north)"
NUMBER = Union[int, float, np.number]  # type: ignore
if TYPE_CHECKING:
    GTYPE = Union[Polygon, MultiPolygon, Tuple[float, float, float, float]]
    GDFTYPE = TypeVar("GDFTYPE", gpd.GeoDataFrame, gpd.GeoSeries)
    XD = TypeVar("XD", xr.Dataset, xr.DataArray)
    CRSTYPE = Union[int, str, pyproj.CRS]

__all__ = [
    "json2geodf",
    "arcgis2geojson",
    "xarray_geomask",
    "gtiff2xarray",
    "xarray2geodf",
    "geodf2xarray",
]


def arcgis2geojson(arcgis: str | dict[str, Any], id_attr: str | None = None) -> dict[str, Any]:
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
    content: list[dict[str, Any]] | dict[str, Any],
    in_crs: CRSTYPE = 4326,
    crs: CRSTYPE = 4326,
) -> gpd.GeoDataFrame:
    """Create GeoDataFrame from (Geo)JSON.

    Parameters
    ----------
    content : dict or list of dict
        A (Geo)JSON dictionary e.g., response.json() or a list of them.
    in_crs : int, str, or pyproj.CRS, optional
        CRS of the content, defaults to ``epsg:4326``.
    crs : int, str, or pyproj.CRS, optional
        The target CRS of the output GeoDataFrame, defaults to ``epsg:4326``.

    Returns
    -------
    geopandas.GeoDataFrame
        Generated geo-data frame from a GeoJSON
    """
    if not isinstance(content, (list, dict)):
        raise InputTypeError("content", "list or list of dict ((geo)json)")

    content = content if isinstance(content, list) else [content]
    try:
        geodf = gpd.GeoDataFrame.from_features(next(iter(content)))
    except TypeError:
        content = [arcgis2geojson(c) for c in content]
        geodf = gpd.GeoDataFrame.from_features(content[0])
    except StopIteration as ex:
        raise EmptyResponseError from ex

    if len(content) > 1:
        geodf = gpd.GeoDataFrame(
            pd.concat((gpd.GeoDataFrame.from_features(c) for c in content), ignore_index=True)
        )

    if "geometry" in geodf and not geodf.geometry.is_empty.all():
        geodf = geodf.set_crs(in_crs)
        if in_crs != crs:
            geodf = geodf.to_crs(crs)
    geodf = cast("gpd.GeoDataFrame", geodf)
    return geodf


def xarray_geomask(
    ds: XD,
    geometry: GTYPE,
    crs: CRSTYPE,
    all_touched: bool = False,
    drop: bool = True,
    from_disk: bool = False,
) -> XD:
    """Mask a ``xarray.Dataset`` based on a geometry.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset(array) to be masked
    geometry : Polygon, MultiPolygon, or tuple of length 4
        The geometry to mask the data
    crs : int, str, or pyproj.CRS
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
    da_attrs = {v: ds[v].attrs for v in ds} if isinstance(ds, xr.Dataset) else {}

    if ds.rio.crs is None:
        raise MissingCRSError

    geom = geotools.geo2polygon(geometry, crs, ds.rio.crs)
    ds = utils.xd_write_crs(ds)
    try:
        ds = ds.rio.clip_box(*geom.bounds, auto_expand=True)
        if isinstance(geometry, (Polygon, MultiPolygon)):
            ds = ds.rio.clip([geom], all_touched=all_touched, drop=drop, from_disk=from_disk)
    except OneDimensionalRaster:
        ds = ds.rio.clip([geom], all_touched=True, drop=drop, from_disk=from_disk)

    if drop:
        ds = utils.xd_write_crs(ds)
    ds.rio.update_attrs(ds_attrs, inplace=True)
    if isinstance(ds, xr.Dataset):
        _ = [ds[v].rio.update_attrs(da_attrs[v], inplace=True) for v in ds]
    ds.rio.update_encoding(ds.encoding, inplace=True)
    return ds


def gtiff2xarray(
    r_dict: dict[str, bytes],
    geometry: GTYPE | None = None,
    geo_crs: CRSTYPE | None = None,
    ds_dims: tuple[str, str] | None = None,
    driver: str | None = None,
    all_touched: bool = False,
    nodata: NUMBER | None = None,
    drop: bool = True,
) -> xr.DataArray | xr.Dataset:
    """Convert (Geo)Tiff byte responses to ``xarray.Dataset``.

    Parameters
    ----------
    r_dict : dict
        Dictionary of (Geo)Tiff byte responses where keys are some names that are used
        for naming each responses, and values are bytes.
    geometry : Polygon, MultiPolygon, or tuple, optional
        The geometry to mask the data that should be in the same CRS as the r_dict.
        Defaults to ``None``.
    geo_crs : int, str, or pyproj.CRS, optional
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
        Requested dataset or dataarray.
    """
    if not isinstance(r_dict, dict):
        raise InputTypeError("r_dict", "dict", '{"name": bytes}')  # noqa: FS003

    try:
        key1 = next(iter(r_dict.keys()))
    except StopIteration as ex:
        raise EmptyResponseError from ex

    var_name = dict(zip(r_dict, r_dict))
    if "_dd_" in key1:
        var_name = {lyr: "_".join(lyr.split("_")[:-3]) for lyr in r_dict.keys()}

    attrs = utils.get_gtiff_attrs(r_dict[key1], ds_dims, driver, nodata)
    dtypes: dict[str, type] = {}
    nodata_dict: dict[str, NUMBER] = {}

    def to_dataset(lyr: str, resp: bytes) -> xr.DataArray:
        with MemoryFile() as memfile:
            memfile.write(resp)
            with memfile.open(driver=driver) as vrt:
                ds = cast("xr.DataArray", rxr.open_rasterio(vrt))
                with contextlib.suppress(ValueError, KeyError):
                    ds = ds.squeeze("band", drop=True)
                ds.name = var_name[lyr]
                dtypes[ds.name] = ds.dtype
                nodata_dict[ds.name] = utils.get_nodata(vrt) if nodata is None else nodata
                ds = ds.rio.write_nodata(nodata_dict[ds.name])
                ds = cast("xr.DataArray", ds)
                return ds

    ds = xr.merge(itertools.starmap(to_dataset, r_dict.items()))

    with contextlib.suppress(ValueError, KeyError):
        ds = ds.squeeze("band", drop=True)

    variables = cast("str", list(ds))
    if len(variables) == 1:
        ds = ds[variables[0]].copy()
        ds = ds.astype(dtypes[variables[0]])
        name = cast("str", ds.name)
        ds.attrs["crs"] = attrs.crs.to_string()
        ds.attrs["nodatavals"] = (nodata_dict[name],)
        ds = ds.rio.write_nodata(nodata_dict[name])
        ds = cast("xr.DataArray", ds)
    else:
        ds.attrs["crs"] = attrs.crs.to_string()
        for v in variables:
            ds[v] = ds[v].astype(dtypes[v])
            ds[v].attrs["crs"] = attrs.crs.to_string()
            ds[v].attrs["nodatavals"] = (nodata_dict[v],)
            ds[v] = ds[v].rio.write_nodata(nodata_dict[v])

    ds = utils.xd_write_crs(ds)  # type: ignore
    if geometry is not None:
        if geo_crs is None:
            raise MissingCRSError
        return xarray_geomask(ds, geometry, geo_crs, all_touched, drop)
    return ds


def xarray2geodf(
    da: xr.DataArray, dtype: str, mask_da: xr.DataArray | None = None, connectivity: int = 8
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
        raise InputTypeError("da", "xarray.DataArray")

    if mask_da is not None and not isinstance(mask_da, xr.DataArray):
        raise InputTypeError("da", "xarray.DataArray or None")

    valid_types = ["int16", "int32", "uint8", "uint16", "float32"]
    if dtype not in valid_types:
        raise InputValueError("dtype", valid_types)

    crs = da.rio.crs if da.attrs.get("crs") is None else da.crs
    if crs is None:
        raise MissingCRSError

    mask = None if mask_da is None else mask_da.to_numpy()
    shapes = rio_features.shapes(
        source=da.to_numpy().astype(dtype),
        transform=da.rio.transform(),
        mask=mask,
        connectivity=connectivity,
    )
    geojsons, values = zip(*shapes)
    geojsons = cast("tuple[dict[str, Any], ...]", geojsons)
    return gpd.GeoDataFrame(
        data={str(da.name): np.array(values, dtype)},
        geometry=[sgeom.shape(g) for g in geojsons],
        crs=crs,
    )


def geodf2xarray(
    geodf: GDFTYPE,
    resolution: float,
    attr_col: str | None = None,
    fill: int | float = 0,
    projected_crs: CRSTYPE = 5070,
) -> xr.Dataset:
    """Rasterize a ``geopandas.GeoDataFrame`` to ``xarray.DataArray``.

    Parameters
    ----------
    geodf : geopandas.GeoDataFrame or geopandas.GeoSeries
        GeoDataFrame or GeoSeries to rasterize.
    resolution : float
        Target resolution of the output raster in the ``projected_crs`` unit. Since
        the default ``projected_crs`` is ``EPSG:5070``, the default unit for the
        resolution is meters.
    attr_col : str, optional
        Column name of the attribute to use as variable., defaults to ``None``,
        i.e., the variable will be a boolean mask where 1 indicates the presence of
        a geometry. Also, note that the attribute must be numeric and have one of the
        following ``numpy`` types: ``int16``, ``int32``, ``uint8``, ``uint16``,
        ``uint32``, ``float32``, and ``float64``.
    fill : int or float, optional
        Value to use for filling the missing values (mask) of the output raster,
        defaults to ``0``.
    projected_crs : int, str, or pyproj.CRS, optional
        A projected CRS to use for the output raster, defaults to ``EPSG:5070``.

    Returns
    -------
    xarray.Dataset
        The xarray Dataset with a single variable.
    """
    if not pyproj.CRS(projected_crs).is_projected:
        raise InputTypeError("projected_crs", "a projected CRS")

    gdf = geodf.to_crs(projected_crs) if geodf.crs != pyproj.CRS(projected_crs) else geodf
    gdf = cast("gpd.GeoDataFrame", gdf)
    west, south, east, north = gdf.total_bounds
    width = np.ceil(abs(west - east) / resolution).astype(int)
    height = np.ceil(abs(north - south) / resolution).astype(int)
    affine = rio_transform.from_bounds(west, south, east, north, width, height)

    if attr_col:
        _types = ["int16", "int32", "uint8", "uint16", "uint32", "float32", "float64"]
        valid_types = [np.dtype(t) for t in _types]
        dtype = geodf[attr_col].dtype
        if dtype not in valid_types:
            raise InputTypeError("attr_col", ", ".join(_types))

        ds = xr.DataArray(
            rio_features.rasterize(
                shapes=zip(gdf.geometry, gdf[attr_col]),
                out_shape=(height, width),
                transform=affine,
                dtype=dtype,
                fill=fill,
            ),
            coords={"x": np.linspace(west, east, width), "y": np.linspace(north, south, height)},
            dims=("y", "x"),
            name=attr_col,
        )
    else:
        ds = xr.DataArray(
            rio_features.rasterize(
                shapes=gdf.geometry,
                out_shape=(height, width),
                transform=affine,
            ),
            coords={"x": np.linspace(west, east, width), "y": np.linspace(north, south, height)},
            dims=("y", "x"),
        )
    ds = ds.rio.write_transform(affine)
    ds = ds.rio.write_crs(projected_crs)
    ds = ds.rio.write_coordinate_system()
    return ds


def validate_crs(crs: CRSTYPE) -> str:
    """Validate a CRS.

    Parameters
    ----------
    crs : str, int, or pyproj.CRS
        Input CRS.

    Returns
    -------
    str
        Validated CRS as a string.
    """
    try:
        return pyproj.CRS(crs).to_string()  # type: ignore
    except ProjCRSError as ex:
        raise InputTypeError("crs", "a valid CRS") from ex
