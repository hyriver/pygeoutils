"""Some utilities for manipulating GeoSpatial data."""
import contextlib
import logging
import numbers
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
import rasterio.features as rio_features
import rasterio.transform as rio_transform
import shapely.geometry as sgeom
import ujson as json
import xarray as xr
from shapely import ops
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from .exceptions import EmptyResponse, InvalidInputType, InvalidInputValue, MissingAttribute

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(""))
logger.handlers = [handler]
logger.propagate = False

DEF_CRS = "epsg:4326"
BOX_ORD = "(west, south, east, north)"


__all__ = [
    "json2geodf",
    "arcgis2geojson",
    "geo2polygon",
    "get_transform",
    "xarray_geomask",
    "gtiff2xarray",
    "xarray2geodf",
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
    id_attr : str
        ID of the attribute of interest

    Returns
    -------
    dict
        A GeoJSON file readable by GeoPandas.
    """

    def convert(arcgis: Dict[str, Any], id_attr: Optional[str] = None) -> Dict[str, Any]:
        """Convert an ArcGIS JSON object to a GeoJSON object."""
        geojson: Dict[str, Any] = {}

        if arcgis.get("features") is not None:
            geojson["type"] = "FeatureCollection"
            geojson["features"] = [convert(f, id_attr) for f in arcgis["features"]]

        if isinstance(arcgis.get("x"), numbers.Number) and isinstance(
            arcgis.get("y"), numbers.Number
        ):
            geojson["type"] = "Point"
            geojson["coordinates"] = [arcgis["x"], arcgis["y"]]
            if isinstance(arcgis.get("z"), numbers.Number):
                geojson["coordinates"].append(arcgis["z"])

        if arcgis.get("points") is not None:
            geojson["type"] = "MultiPoint"
            geojson["coordinates"] = arcgis["points"]

        if arcgis.get("paths") is not None:
            if len(arcgis["paths"]) == 1:
                geojson["type"] = "LineString"
                geojson["coordinates"] = arcgis["paths"][0]
            else:
                geojson["type"] = "MultiLineString"
                geojson["coordinates"] = arcgis["paths"]

        if arcgis.get("rings") is not None:
            geojson = _rings2geojson(arcgis["rings"])

        coords = ("xmin", "xmax", "ymin", "ymax")
        if all(isinstance(arcgis.get(c), numbers.Number) for c in coords):
            geojson["type"] = "Polygon"
            geojson["coordinates"] = [
                [
                    [arcgis["xmax"], arcgis["ymax"]],
                    [arcgis["xmin"], arcgis["ymax"]],
                    [arcgis["xmin"], arcgis["ymin"]],
                    [arcgis["xmax"], arcgis["ymin"]],
                    [arcgis["xmax"], arcgis["ymax"]],
                ]
            ]

        if arcgis.get("geometry") is not None or arcgis.get("attributes") is not None:
            geojson["type"] = "Feature"
            if arcgis.get("geometry") is not None:
                geojson["geometry"] = convert(arcgis["geometry"])
            else:
                geojson["geometry"] = None

            if arcgis.get("attributes") is not None:
                geojson["properties"] = arcgis["attributes"]
                keys = [id_attr, "OBJECTID", "FID"] if id_attr else ["OBJECTID", "FID"]
                for key in keys:
                    if isinstance(arcgis["attributes"].get(key), (numbers.Number, str)):
                        geojson["id"] = arcgis["attributes"][key]
                        break
                if "id" not in geojson:
                    logger.warning("No valid id attribute found.")
            else:
                geojson["properties"] = None

        if json.dumps(geojson.get("geometry")) == json.dumps({}):
            geojson["geometry"] = None

        return geojson

    if isinstance(arcgis, str):
        return convert(json.loads(arcgis), id_attr)

    return convert(arcgis, id_attr)


def _rings2geojson(rings: List[List[List[float]]]) -> Dict[str, Any]:
    """Check for holes in the ring and fill them."""
    outer_rings, holes = _get_outer_rings(rings)
    uncontained_holes = _get_uncontained_holes(outer_rings, holes)

    # if we couldn't match any holes using contains we can try intersects...
    while uncontained_holes:
        # pop a hole off out stack
        hole = uncontained_holes.pop()

        # loop over all outer rings and see if any intersect our hole.
        intersects = False
        x = len(outer_rings) - 1
        while x >= 0:
            outer_ring = outer_rings[x][0]
            l1, l2 = LineString(outer_ring), LineString(hole)
            intersects = l1.intersects(l2)
            if intersects:
                # the hole is contained push it into our polygon
                outer_rings[x].append(hole)  # type: ignore
                intersects = True
                break
            x = x - 1

        if not intersects:
            outer_rings.append([hole[::-1]])  # type: ignore

    if len(outer_rings) == 1:
        return {"type": "Polygon", "coordinates": outer_rings[0]}

    return {"type": "MultiPolygon", "coordinates": outer_rings}


def _get_outer_rings(rings: List[List[List[float]]]) -> Tuple[List[List[float]], List[List[float]]]:
    """Get outer rings and holes in a list of rings."""
    outer_rings = []
    holes = []
    for ring in rings:
        if not np.all(np.isclose(ring[0], ring[-1])):
            ring.append(ring[0])

        if len(ring) < 4:
            continue

        total = sum((pt2[0] - pt1[0]) * (pt2[1] + pt1[1]) for pt1, pt2 in zip(ring[:-1], ring[1:]))
        # Clock-wise check
        if total >= 0:
            # wind outer rings counterclockwise for RFC 7946 compliance
            outer_rings.append([ring[::-1]])
        else:
            # wind inner rings clockwise for RFC 7946 compliance
            holes.append(ring[::-1])
    return outer_rings, holes  # type: ignore


def _get_uncontained_holes(
    outer_rings: List[List[float]], holes: List[List[float]]
) -> List[List[float]]:
    """Get all the uncontstrained holes."""
    uncontained_holes = []

    # while there are holes left...
    while holes:
        # pop a hole off out stack
        hole = holes.pop()

        # loop over all outer rings and see if they contain our hole.
        contained = False
        x = len(outer_rings) - 1
        while x >= 0:
            outer_ring = outer_rings[x][0]
            l1, l2 = LineString(outer_ring), LineString(hole)
            p2 = Point(hole[0])
            intersects = l1.intersects(l2)
            contains = l1.contains(p2)
            if not intersects and contains:
                # the hole is contained push it into our polygon
                outer_rings[x].append(hole)  # type: ignore
                contained = True
                break
            x = x - 1

        # ring is not contained in any outer ring
        # sometimes this happens https://github.com/Esri/esri-leaflet/issues/320
        if not contained:
            uncontained_holes.append(hole)
    return uncontained_holes


def gtiff2xarray(
    r_dict: Dict[str, bytes],
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
    geo_crs: str,
    ds_dims: Optional[Tuple[str, str]] = None,
    driver: Optional[str] = None,
    all_touched: bool = False,
) -> Union[xr.DataArray, xr.Dataset]:
    """Convert (Geo)Tiff byte responses to ``xarray.Dataset``.

    Parameters
    ----------
    r_dict : dict
        Dictionary of (Geo)Tiff byte responses where keys are some names that are used
        for naming each responses, and values are bytes.
    geometry : Polygon, MultiPolygon, or tuple
        The geometry to mask the data that should be in the same CRS as the r_dict.
    geo_crs : str
        The spatial reference of the input geometry.
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
        of the shapes, or if it is selected by Bresenham’s line algorithm.

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

    attrs = get_gtiff_attrs(r_dict[key1], ds_dims, driver)

    tmp_dir = tempfile.gettempdir()

    def to_dataset(lyr: str, resp: bytes) -> xr.DataArray:
        with rio.MemoryFile() as memfile:
            memfile.write(resp)
            with memfile.open(driver=driver) as vrt:
                ds = xr.open_rasterio(vrt)
                with contextlib.suppress(ValueError):
                    ds = ds.squeeze("band", drop=True)
                ds = ds.sortby(attrs.dims[0], ascending=False)
                ds.attrs["crs"] = attrs.crs.to_string()
                ds.attrs["transform"] = attrs.transform
                ds.name = var_name[lyr]
                fpath = Path(tmp_dir, f"{uuid.uuid4().hex}.nc")
                ds.to_netcdf(fpath)
                return fpath

    ds = xr.open_mfdataset(
        (to_dataset(lyr, resp) for lyr, resp in r_dict.items()),
        parallel=True,
    )
    if len(ds.variables) - len(ds.dims) == 1:
        ds = ds[list(ds.keys())[0]]
    ds = ds.sortby(attrs.dims[0], ascending=False)
    ds.attrs["crs"] = attrs.crs.to_string()
    ds.attrs["nodatavals"] = (attrs.nodata,)
    transform, _, _ = get_transform(ds, attrs.dims)
    ds.attrs["transform"] = transform2tuple(transform)
    ds.attrs["res"] = (transform.a, transform.e)

    ds = attrs2tuple(ds, ["scales", "offsets"])
    if isinstance(ds, xr.Dataset):
        for v in ds.keys():
            ds[v] = attrs2tuple(ds[v], ["scales", "offsets"])

    return xarray_geomask(ds, geometry, geo_crs, attrs.dims, all_touched)


def attrs2tuple(
    ds: Union[xr.Dataset, xr.DataArray], attrs: List[str]
) -> Union[xr.Dataset, xr.DataArray]:
    """Convert the floats attributes of a dataset or dataarray to a tuple."""
    for attr in attrs:
        if attr in ds.attrs and not isinstance(ds.attrs[attr], tuple):
            ds.attrs[attr] = (ds.attrs[attr],)
    return ds


def xarray_geomask(
    ds: Union[xr.Dataset, xr.DataArray],
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
    geo_crs: str,
    ds_dims: Optional[Tuple[str, str]] = None,
    all_touched: bool = False,
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
        of the shapes, or if it is selected by Bresenham’s line algorithm.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The input dataset with a mask applied (np.nan)
    """
    attrs = ds.attrs
    if "crs" not in attrs:
        raise MissingAttribute("crs", list(attrs.keys()))

    if ds_dims is None:
        ds_dims = _get_dim_names(ds)

    valid_dims = list(ds.sizes)
    if ds_dims is None or any(d not in valid_dims for d in ds_dims):
        raise MissingAttribute("ds_dims", valid_dims)

    transform, width, height = get_transform(ds, ds_dims)
    _geometry = geo2polygon(geometry, geo_crs, ds.crs)

    _mask = rio_features.geometry_mask(
        [_geometry], (height, width), transform, invert=True, all_touched=all_touched
    )

    coords = {ds_dims[0]: ds.coords[ds_dims[0]], ds_dims[1]: ds.coords[ds_dims[1]]}
    mask = xr.DataArray(_mask, coords, dims=ds_dims)

    ds_masked = ds.where(mask, drop=True)
    ds_masked.attrs = attrs
    ds_masked.attrs["transform"] = transform2tuple(transform)
    ds_masked.attrs["bounds"] = _geometry.bounds
    ds_masked.attrs["res"] = (transform.a, transform.e)

    return ds_masked


class Attrs(NamedTuple):
    """Attributes of a GTiff byte response."""

    nodata: np.float64
    crs: pyproj.crs.crs.CRS
    dims: Tuple[str, str]
    transform: Tuple[float, float, float, float, float, float]


def get_gtiff_attrs(
    resp: bytes, ds_dims: Optional[Tuple[str, str]] = None, driver: Optional[str] = None
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

    Returns
    -------
    dict
        No data, CRS, and dimension names for vertical and horizontal directions or
        a list of the existing dimensions if they are not in a list of common names.
    """
    with rio.MemoryFile() as memfile:
        memfile.write(resp)
        with memfile.open(driver=driver) as src:
            r_crs = pyproj.CRS(src.crs)
            if src.nodata is None:
                try:
                    nodata = np.iinfo(src.dtypes[0]).max
                except ValueError:
                    nodata = np.nan
            else:
                nodata = np.dtype(src.dtypes[0]).type(src.nodata)

            ds = xr.open_rasterio(src)
            if ds_dims is None:
                ds_dims = _get_dim_names(ds)

            valid_dims = list(ds.sizes)
            if ds_dims is None or any(d not in valid_dims for d in ds_dims):
                raise MissingAttribute("ds_dims", valid_dims)
            if not isinstance(src.transform, tuple):
                transform = transform2tuple(src.transform)
            else:
                transform = src.transform  # type: ignore
    return Attrs(nodata, r_crs, ds_dims, transform)


def _get_dim_names(ds: Union[xr.DataArray, xr.Dataset]) -> Optional[Tuple[str, str]]:
    """Get vertical and horizontal dimension names."""
    y_dims = {"y", "Y", "lat", "Lat", "latitude", "Latitude"}
    x_dims = {"x", "X", "lon", "Lon", "longitude", "Longitude"}
    try:
        y_dim = list(set(ds.coords).intersection(y_dims))[0]
        x_dim = list(set(ds.coords).intersection(x_dims))[0]
    except IndexError:
        return None
    else:
        return (y_dim, x_dim)


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

    Returns
    -------
    rasterio.Affine, int, int
        The affine transform, width, and height
    """
    ydim, xdim = ds_dims
    height, width = ds.sizes[ydim], ds.sizes[xdim]

    left, right = ds[xdim].min().item(), ds[xdim].max().item()
    bottom, top = ds[ydim].min().item(), ds[ydim].max().item()

    x_res = abs(left - right) / (width - 1)
    y_res = abs(top - bottom) / (height - 1)

    left -= x_res * 0.5
    right += x_res * 0.5
    top += y_res * 0.5
    bottom -= y_res * 0.5

    transform = rio_transform.from_bounds(left, bottom, right, top, width, height)
    return transform, width, height


def transform2tuple(transform: rio.Affine) -> Tuple[float, float, float, float, float, float]:
    """Convert an affine transform to a tuple.

    Parameters
    ----------
    transform : rio.Affine
        The affine transform to be converted

    Returns
    -------
    tuple
        The affine transform as a tuple (a, b, c, d, e, f)
    """
    return tuple(getattr(transform, c) for c in ["a", "b", "c", "d", "e", "f"])  # type: ignore


def geo2polygon(
    geometry: Union[Polygon, MultiPolygon, Tuple[float, float, float, float]],
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
    da: xr.DataArray, dtype: str, mask_da: Optional[xr.DataArray] = None
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
        source=da.to_numpy().astype(_dtype), transform=da.transform, mask=mask
    )
    geometry, values = zip(*shapes)
    return gpd.GeoDataFrame(
        data={da.name: _dtype(values)},
        geometry=[sgeom.shape(g) for g in geometry],
        crs=da.crs,
    )
