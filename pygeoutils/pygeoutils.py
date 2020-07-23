"""Some utilities for Hydrodata."""
import numbers
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

import geopandas as gpd
import numpy as np
import pyproj
import rasterio as rio
import rasterio.features as rio_features
import rasterio.warp as rio_warp
import simplejson as json
import xarray as xr
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import transform

from .exceptions import InvalidInputType

DEF_CRS = "epsg:4326"


def json2geodf(
    content: Union[List[Dict[str, Any]], Dict[str, Any]], in_crs: str = DEF_CRS, crs: str = DEF_CRS,
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
        geodf = gpd.GeoDataFrame.from_features(content[0], crs=in_crs)
    except TypeError:
        content = [arcgis2geojson(c) for c in content]
        geodf = gpd.GeoDataFrame.from_features(content[0], crs=in_crs)

    if len(content) > 1:
        geodf = geodf.append([gpd.GeoDataFrame.from_features(c, crs=in_crs) for c in content[1:]])

    if in_crs != crs:
        geodf = geodf.to_crs(crs)

    return geodf.drop_duplicates()


def arcgis2geojson(arcgis: Dict[str, Any], idAttribute: Optional[str] = None) -> Dict[str, Any]:
    """Convert ESRIGeoJSON format to GeoJSON.

    Notes
    -----
    Based on https://github.com/chris48s/arcgis2geojson

    Parameters
    ----------
    arcgis : str or binary
        The ESRIGeoJSON format str (or binary)
    idAttribute : str
        ID of the attribute of interest

    Returns
    -------
    dict
        A GeoJSON file readable by GeoPandas
    """

    def convert(arcgis, idAttribute=None):
        """Convert an ArcGIS JSON object to a GeoJSON object."""
        geojson = {}

        if "features" in arcgis and arcgis["features"]:
            geojson["type"] = "FeatureCollection"
            geojson["features"] = []
            geojson["features"] = [convert(feature, idAttribute) for feature in arcgis["features"]]

        if (
            "x" in arcgis
            and isinstance(arcgis["x"], numbers.Number)
            and "y" in arcgis
            and isinstance(arcgis["y"], numbers.Number)
        ):
            geojson["type"] = "Point"
            geojson["coordinates"] = [arcgis["x"], arcgis["y"]]
            if "z" in arcgis and isinstance(arcgis["z"], numbers.Number):
                geojson["coordinates"].append(arcgis["z"])

        if "points" in arcgis:
            geojson["type"] = "MultiPoint"
            geojson["coordinates"] = arcgis["points"]

        if "paths" in arcgis:
            if len(arcgis["paths"]) == 1:
                geojson["type"] = "LineString"
                geojson["coordinates"] = arcgis["paths"][0]
            else:
                geojson["type"] = "MultiLineString"
                geojson["coordinates"] = arcgis["paths"]

        if "rings" in arcgis:
            geojson = rings2geojson(arcgis["rings"])

        if (
            "xmin" in arcgis
            and isinstance(arcgis["xmin"], numbers.Number)
            and "ymin" in arcgis
            and isinstance(arcgis["ymin"], numbers.Number)
            and "xmax" in arcgis
            and isinstance(arcgis["xmax"], numbers.Number)
            and "ymax" in arcgis
            and isinstance(arcgis["ymax"], numbers.Number)
        ):
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

        if "geometry" in arcgis or "attributes" in arcgis:
            geojson["type"] = "Feature"
            if "geometry" in arcgis:
                geojson["geometry"] = convert(arcgis["geometry"])
            else:
                geojson["geometry"] = None

            if "attributes" in arcgis:
                geojson["properties"] = arcgis["attributes"]
                try:
                    attributes = arcgis["attributes"]
                    keys = [idAttribute, "OBJECTID", "FID"] if idAttribute else ["OBJECTID", "FID"]
                    for key in keys:
                        if key in attributes and (
                            isinstance(attributes[key], (numbers.Number, str))
                        ):
                            geojson["id"] = attributes[key]
                            break
                except KeyError:
                    warn("No valid id attribute found.")
            else:
                geojson["properties"] = None

        if "geometry" in geojson and not geojson["geometry"]:
            geojson["geometry"] = None

        return geojson

    def rings2geojson(rings):
        """Check for holes in the ring and fill them."""
        outerRings = []
        holes = []
        x = None  # iterable
        outerRing = None  # current outer ring being evaluated
        hole = None  # current hole being evaluated

        for ring in rings:
            if not all(np.isclose(ring[0], ring[-1])):
                ring.append(ring[0])

            if len(ring) < 4:
                continue

            total = sum(
                (pt2[0] - pt1[0]) * (pt2[1] + pt1[1]) for pt1, pt2 in zip(ring[:-1], ring[1:])
            )
            # Clock-wise check
            if total >= 0:
                outerRings.append(
                    [ring[::-1]]
                )  # wind outer rings counterclockwise for RFC 7946 compliance
            else:
                holes.append(ring[::-1])  # wind inner rings clockwise for RFC 7946 compliance

        uncontainedHoles = []

        # while there are holes left...
        while holes:
            # pop a hole off out stack
            hole = holes.pop()

            # loop over all outer rings and see if they contain our hole.
            contained = False
            x = len(outerRings) - 1
            while x >= 0:
                outerRing = outerRings[x][0]
                l1, l2 = LineString(outerRing), LineString(hole)
                p2 = Point(hole[0])
                intersects = l1.intersects(l2)
                contains = l1.contains(p2)
                if not intersects and contains:
                    # the hole is contained push it into our polygon
                    outerRings[x].append(hole)
                    contained = True
                    break
                x = x - 1

            # ring is not contained in any outer ring
            # sometimes this happens https://github.com/Esri/esri-leaflet/issues/320
            if not contained:
                uncontainedHoles.append(hole)

        # if we couldn't match any holes using contains we can try intersects...
        while uncontainedHoles:
            # pop a hole off out stack
            hole = uncontainedHoles.pop()

            # loop over all outer rings and see if any intersect our hole.
            intersects = False
            x = len(outerRings) - 1
            while x >= 0:
                outerRing = outerRings[x][0]
                l1, l2 = LineString(outerRing), LineString(hole)
                intersects = l1.intersects(l2)
                if intersects:
                    # the hole is contained push it into our polygon
                    outerRings[x].append(hole)
                    intersects = True
                    break
                x = x - 1

            if not intersects:
                outerRings.append([hole[::-1]])

        if len(outerRings) == 1:
            return {"type": "Polygon", "coordinates": outerRings[0]}

        return {"type": "MultiPolygon", "coordinates": outerRings}

    if isinstance(arcgis, str):
        return json.dumps(convert(json.loads(arcgis), idAttribute))

    return convert(arcgis, idAttribute)


def gtiff2xarray(
    r_dict: Dict[str, bytes],
    geometry: Union[Polygon, Tuple[float, float, float, float]],
    geo_crs: str,
) -> Union[xr.DataArray, xr.Dataset]:
    """Convert responses from ``pygeoogc.wms_bybox`` to ``xarray.Dataset``.

    Parameters
    ----------
    r_dict : dict
        The output of ``wms_bybox`` function.
    geometry : Polygon or tuple
        The geometry to mask the data that should be in the same CRS as the r_dict.
    geo_crs : str
        The spatial reference of the input geometry.

    Returns
    -------
    xarray.Dataset or xarray.DataAraay
        The dataset or data array based on the number of variables.
    """
    var_name = {lyr: f"{''.join(n for n in lyr.split('_')[:-1])}" for lyr in r_dict.keys()}

    ds = xr.merge([_create_dataset(r, var_name[lyr]) for lyr, r in r_dict.items()])
    ds = _geometry_mask(ds, geometry, geo_crs)

    if len(ds.variables) - len(ds.dims) == 1:
        ds = ds[list(ds.keys())[0]]
    return ds


def _create_dataset(content: bytes, name: str,) -> Union[xr.Dataset, xr.DataArray]:
    """Create dataset from a response clipped by a geometry.

    Parameters
    ----------
    content : requests.Response
        The response to be processed
    name : str
        Variable name in the dataset

    Returns
    -------
    xarray.Dataset
        Generated xarray DataSet or DataArray
    """
    with rio.MemoryFile() as memfile:
        memfile.write(content)
        with memfile.open() as src:
            ds = xr.open_rasterio(src)
            try:
                ds = ds.squeeze("band", drop=True)
            except ValueError:
                pass
            ds.name = name

            ds.attrs["transform"] = src.transform
            ds.attrs["res"] = (src.transform[0], src.transform[4])
            ds.attrs["bounds"] = tuple(src.bounds)
            ds.attrs["nodatavals"] = src.nodatavals
            ds.attrs["crs"] = src.crs
    return ds


def _geometry_mask(ds: Union[xr.Dataset, xr.DataArray], geometry: Polygon, geo_crs: str):
    """Mask a ``xarray.Dataset`` based on a geometry.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset(array) to be masked
    geometry : Polygon
        The geometry to mask the data
    geo_crs : str
        The spatial reference of the input geometry

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The input dataset with a mask applied (np.nan)
    """
    if isinstance(geometry, tuple):
        check_bbox(geometry)  # type: ignore
        _geometry = box(*geometry)
    elif isinstance(geometry, Polygon):
        _geometry = geometry
    else:
        raise InvalidInputType("geometry", "Polygon or tuple of length 4")

    left, bottom, right, top = _geometry.bounds
    height, width = list(ds.sizes.values())[-2:]
    transform = rio_warp.calculate_default_transform(
        geo_crs, geo_crs, width, height, left, bottom, right, top
    )[0]
    _mask = rio_features.geometry_mask([_geometry], (height, width), transform, invert=True)
    mask = xr.DataArray(_mask, dims=list(ds.sizes.keys())[-2:])

    def mask_da(da):
        da = da.where(mask, drop=True)
        da.attrs["nodatavals"] = (np.nan,)
        return da

    return ds.map(mask_da)


class MatchCRS:
    """Match CRS of a input geometry (Polygon, bbox, coord) with the output CRS.

    Parameters
    ----------
    geometry : tuple or Polygon
        The input geometry (Polygon, bbox, coord)
    in_crs : str
        The spatial reference of the input geometry
    out_crs : str
        The target spatial reference
    """

    @staticmethod
    def geometry(geom: Polygon, in_crs: str, out_crs: str) -> Polygon:
        if not isinstance(geom, Polygon):
            raise InvalidInputType("geom", "Polygon")

        project = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True).transform
        return transform(project, geom)

    @staticmethod
    def bounds(
        geom: Tuple[float, float, float, float], in_crs: str, out_crs: str
    ) -> Tuple[float, float, float, float]:
        if not isinstance(geom, tuple) and len(geom) != 4:
            raise InvalidInputType("geom", "tuple of length 4", "(west, south, east, north)")

        project = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True).transform
        return transform(project, box(*geom)).bounds

    @staticmethod
    def coords(
        geom: Tuple[Tuple[float, ...], Tuple[float, ...]], in_crs: str, out_crs: str
    ) -> Tuple[Any, ...]:
        if not isinstance(geom, tuple) and len(geom) != 2:
            raise InvalidInputType("geom", "tuple of length 2", "((xs), (ys))")

        project = pyproj.Transformer.from_crs(in_crs, out_crs, always_xy=True).transform
        return tuple(zip(*[project(x, y) for x, y in zip(*geom)]))


def check_bbox(bbox: Tuple[float, float, float, float]) -> None:
    """Check if an input inbox is a tuple of length 4."""
    if not isinstance(bbox, tuple) or len(bbox) != 4:
        raise InvalidInputType("bbox", "tuple", "(west, south, east, north)")
