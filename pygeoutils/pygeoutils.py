"""Some utilities for manipulating GeoSpatial data."""
import contextlib
import logging
import numbers
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import affine
import geopandas as gpd
import numpy as np
import orjson as json
import pyproj
import rasterio as rio
import rasterio.mask as rio_mask
import rasterio.transform as rio_transform
import shapely.geometry as sgeom
import xarray as xr
from shapely import ops
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon

from .exceptions import InvalidInputType, InvalidInputValue

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(""))
logger.handlers = [handler]
logger.propagate = False

DEF_CRS = "epsg:4326"
BOX_ORD = "(west, south, east, north)"


def json2geodf(
    content: Union[List[Dict[str, Any]], Dict[str, Any]],
    in_crs: str = DEF_CRS,
    crs: str = DEF_CRS,
) -> gpd.GeoDataFrame:
    """Create GeoDataFrame from (Geo)JSON.

    Parameters
    ----------
    content : dict or list of dict
        A (Geo)JSON dictionary e.g., r.json() or a list of them.
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
    except StopIteration:
        raise StopIteration("Resnpose is empty.")

    if len(content) > 1:
        geodf = geodf.append([gpd.GeoDataFrame.from_features(c) for c in content[1:]])

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
    ds_dims: Tuple[str, str] = ("y", "x"),
    driver: str = "GTiff",
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
        of the target dataset, default to ("y", "x").
    driver : str, optional
        A GDAL driver for reading the content, defaults to GTiff. A list of the drivers
        can be found here: https://gdal.org/drivers/raster/index.html

    Returns
    -------
    xarray.Dataset or xarray.DataAraay
        Parallel (with dask) dataset or dataarray.
    """
    if not isinstance(r_dict, dict):
        raise InvalidInputType("r_dict", "dict", '{"name": Response.content}')  # noqa: FS003

    try:
        key1 = next(iter(r_dict.keys()))
    except StopIteration:
        raise StopIteration("Resnpose dict is empty.")

    if "_dd_" in key1:
        var_name = {lyr: "_".join(lyr.split("_")[:-3]) for lyr in r_dict.keys()}
    else:
        var_name = dict(zip(r_dict, r_dict))

    nodata, r_crs = _get_nodata_crs(r_dict[key1], driver)

    _geometry = _geo2polygon(geometry, geo_crs, r_crs)

    tmp_dir = tempfile.gettempdir()

    def to_dataset(lyr: str, resp: bytes) -> xr.DataArray:
        with rio.MemoryFile() as memfile:
            memfile.write(resp)
            with memfile.open(driver=driver) as src:
                geom = [_geometry.intersection(sgeom.box(*src.bounds))]
                if geom[0].is_empty:
                    msk, transform, _ = rio_mask.raster_geometry_mask(src, [_geometry], invert=True)
                else:
                    msk, transform, _ = rio_mask.raster_geometry_mask(src, geom, invert=True)
                meta = src.meta
                meta.update(
                    {
                        "driver": driver,
                        "height": msk.shape[0],
                        "width": msk.shape[1],
                        "transform": transform,
                        "nodata": nodata,
                    }
                )

                with rio.vrt.WarpedVRT(src, **meta) as vrt:
                    ds = xr.open_rasterio(vrt)
                    valid_dims = list(ds.sizes)
                    if any(d not in valid_dims for d in ds_dims):
                        raise InvalidInputValue("ds_dims", valid_dims)
                    with contextlib.suppress(ValueError):
                        ds = ds.squeeze("band", drop=True)

                    coords = {
                        ds_dims[0]: ds.coords[ds_dims[0]],
                        ds_dims[1]: ds.coords[ds_dims[1]],
                    }
                    msk_da = xr.DataArray(msk, coords, dims=ds_dims)
                    ds = ds.where(msk_da, drop=True)
                    ds.attrs["crs"] = r_crs.to_string()
                    ds.name = var_name[lyr]
                    fpath = Path(tmp_dir, f"{uuid.uuid4().hex}.nc")
                    ds.to_netcdf(fpath)
                    return fpath

    ds = xr.open_mfdataset((to_dataset(lyr, resp) for lyr, resp in r_dict.items()), parallel=True)

    if len(ds.variables) - len(ds.dims) == 1:
        ds = ds[list(ds.keys())[0]]

    ds.attrs["nodatavals"] = (nodata,)

    valid_ycoords = {"y", "Y", "lat", "Lat", "latitude", "Latitude"}
    valid_xcoords = {"x", "X", "lon", "Lon", "longitude", "Longitude"}
    ycoord = list(set(ds.coords).intersection(valid_ycoords))
    xcoord = list(set(ds.coords).intersection(valid_xcoords))
    if len(xcoord) == 1 and len(ycoord) == 1:
        transform, _, _ = _get_transform(ds, ds_dims)
        ds = ds.sortby(ycoord[0], ascending=False)
        ds.attrs["transform"] = transform
        ds.attrs["res"] = (transform.a, transform.e)

    return ds


def _get_nodata_crs(resp: bytes, driver: str) -> Tuple[np.float64, pyproj.crs.crs.CRS]:
    """Get nodata and crs value of a raster in bytes.

    Parameters
    ----------
    resp : bytes
        Raster response returned from a wed service request such as WMS
    driver : str
        A GDAL driver for reading the content, defaults to GTiff. A list of the drivers
        can be found here: https://gdal.org/drivers/raster/index.html

    Returns
    -------
    tuple
        (NoData, CRS)
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
    return nodata, r_crs


def _get_transform(
    ds: Union[xr.Dataset, xr.DataArray],
    ds_dims: Tuple[str, str] = ("y", "x"),
) -> Tuple[affine.Affine, int, int]:
    """Get transform of a ``xarray.Dataset`` or ``xarray.DataArray``.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset(array) to be masked
    ds_dims : tuple, optional
        Names of the coordinames in the dataset, defaults to ``("y", "x")``.

    Returns
    -------
    affine.Affine, int, int
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


def _geo2polygon(
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
    match_crs = MatchCRS(geo_crs, crs)
    geom = sgeom.box(*geometry) if isinstance(geometry, tuple) else geometry
    geom = match_crs.geometry(geom)

    if not geom.is_valid:
        geom = geom.buffer(0.0)
    return geom


class MatchCRS:
    """Reproject a geometry to another CRS.

    Parameters
    ----------
    in_crs : str
        Spatial reference of the input geometry
    out_crs : str
        Target spatial reference
    """

    def __init__(self, in_crs: str, out_crs: str):
        self.project = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True).transform

    def geometry(
        self, geom: Union[Polygon, LineString, MultiLineString, MultiPolygon, Point, MultiPoint]
    ) -> Union[Polygon, LineString, MultiLineString, MultiPolygon, Point, MultiPoint]:
        """Reproject a geometry to the specified output CRS.

        Parameters
        ----------
        geom : LineString, MultiLineString, Polygon, MultiPolygon, Point, or MultiPoint
            Input geometry.

        Returns
        -------
        LineString, MultiLineString, Polygon, MultiPolygon, Point, or MultiPoint
            Input geometry in the specified CRS.

        Examples
        --------
        >>> from pygeoogc import MatchCRS
        >>> from shapely.geometry import Point
        >>> point = Point(-7766049.665, 5691929.739)
        >>> MatchCRS("epsg:3857", "epsg:4326").geometry(point).xy
        (array('d', [-69.7636111130079]), array('d', [45.44549114818127]))
        """
        if not isinstance(
            geom, (Polygon, LineString, MultiLineString, MultiPolygon, Point, MultiPoint)
        ):
            types = "LineString, MultiLineString, Polygon, MultiPolygon, Point, or MultiPoint"
            raise InvalidInputType("geom", types)

        return ops.transform(self.project, geom)

    def bounds(self, geom: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Reproject a bounding box to the specified output CRS.

        Parameters
        ----------
        geom : tuple
            Input bounding box (xmin, ymin, xmax, ymax).

        Returns
        -------
        tuple
            Input bounding box in the specified CRS.

        Examples
        --------
        >>> from pygeoogc import MatchCRS
        >>> bbox = (-7766049.665, 5691929.739, -7763049.665, 5696929.739)
        >>> MatchCRS("epsg:3857", "epsg:4326").bounds(bbox)
        (-69.7636111130079, 45.44549114818127, -69.73666165448431, 45.47699468552394)
        """
        if not (isinstance(geom, tuple) and len(geom) == 4):
            raise InvalidInputType("geom", "tuple", BOX_ORD)

        return ops.transform(self.project, sgeom.box(*geom)).bounds

    def coords(self, geom: List[Tuple[float, float]]) -> List[Tuple[Any, ...]]:
        """Reproject a list of coordinates to the specified output CRS.

        Parameters
        ----------
        geom : list of tuple
            Input coords [(x1, y1), ...].

        Returns
        -------
        tuple
            Input list of coords in the specified CRS.

        Examples
        --------
        >>> from pygeoogc import MatchCRS
        >>> coords = [(-7766049.665, 5691929.739)]
        >>> MatchCRS("epsg:3857", "epsg:4326").coords(coords)
        [(-69.7636111130079, 45.44549114818127)]
        """
        if not (isinstance(geom, list) and all(len(c) == 2 for c in geom)):
            raise InvalidInputType("geom", "list of tuples", "[(x1, y1), ...]")

        return list(zip(*self.project(*zip(*geom))))
