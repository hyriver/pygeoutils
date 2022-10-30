"""Some utilities for manipulating GeoSpatial data."""
from __future__ import annotations

import itertools
import os
import sys
from dataclasses import dataclass
from numbers import Number
from typing import Any, NamedTuple, Union

import numpy as np
import pyproj
import rasterio as rio
import ujson as json
import xarray as xr
from loguru import logger
from shapely.geometry import LineString, Point

NUMBER = Union[int, float, np.number]

logger.configure(
    handlers=[
        {
            "sink": sys.stdout,
            "colorize": True,
            "format": " | ".join(
                [
                    "{time:YYYY-MM-DD at HH:mm:ss}",  # noqa: FS003
                    "{name: ^15}.{function: ^15}:{line: >3}",  # noqa: FS003
                    "{message}",  # noqa: FS003
                ]
            ),
        }
    ]
)
if os.environ.get("HYRIVER_VERBOSE", "false").lower() == "true":
    logger.enable("pygeoutils")
else:
    logger.disable("pygeoutils")


@dataclass
class Convert:
    """Functions to Convert an ArcGIS JSON object to a GeoJSON object."""

    id_attr: str | None = None

    def features(self, arcgis: dict[str, Any], geojson: dict[str, Any]) -> dict[str, Any]:
        """Get the features from an ArcGIS JSON object."""
        geojson["type"] = "FeatureCollection"
        geojson["features"] = [convert(f, self.id_attr) for f in arcgis["features"]]
        return geojson

    @staticmethod
    def points(arcgis: dict[str, Any], geojson: dict[str, Any]) -> dict[str, Any]:
        """Get the points from an ArcGIS JSON object."""
        geojson["type"] = "MultiPoint"
        geojson["coordinates"] = arcgis["points"]
        return geojson

    @staticmethod
    def paths(arcgis: dict[str, Any], geojson: dict[str, Any]) -> dict[str, Any]:
        """Get the paths from an ArcGIS JSON object."""
        if len(arcgis["paths"]) == 1:
            geojson["type"] = "LineString"
            geojson["coordinates"] = arcgis["paths"][0]
        else:
            geojson["type"] = "MultiLineString"
            geojson["coordinates"] = arcgis["paths"]
        return geojson

    def xy(self, arcgis: dict[str, Any], geojson: dict[str, Any]) -> dict[str, Any]:
        """Get the xy coordinates from an ArcGIS JSON object."""
        if self.isnumber([arcgis.get("x"), arcgis.get("y")]):
            geojson["type"] = "Point"
            geojson["coordinates"] = [arcgis["x"], arcgis["y"]]
            if self.isnumber([arcgis.get("z")]):
                geojson["coordinates"].append(arcgis["z"])
        return geojson

    def rings(self, arcgis: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        """Get the rings from an ArcGIS JSON object."""
        outer_rings, holes = self.get_outer_rings(arcgis["rings"])
        uncontained_holes = self.get_uncontained_holes(outer_rings, holes)

        while uncontained_holes:
            # pop a hole off out stack
            hole = uncontained_holes.pop()

            intersects = False
            x = len(outer_rings) - 1
            while x >= 0:
                outer_ring = outer_rings[x][0]
                l1, l2 = LineString(outer_ring), LineString(hole)
                intersects = l1.intersects(l2)
                if intersects:
                    outer_rings[x].append(hole)
                    intersects = True
                    break
                x = x - 1

            if not intersects:
                outer_rings.append([hole[::-1]])

        if len(outer_rings) == 1:
            return {"type": "Polygon", "coordinates": outer_rings[0]}

        return {"type": "MultiPolygon", "coordinates": outer_rings}

    def coords(self, arcgis: dict[str, Any], geojson: dict[str, Any]) -> dict[str, Any]:
        """Get the bounds from an ArcGIS JSON object."""
        if self.isnumber([arcgis.get(c) for c in ("xmin", "xmax", "ymin", "ymax")]):
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
        return geojson

    @staticmethod
    def geometry(arcgis: dict[str, Any], geojson: dict[str, Any]) -> dict[str, Any]:
        """Get the geometry from an ArcGIS JSON object."""
        if arcgis.get("geometry") is not None:
            curves = {
                "curveRings": "Curved Polygon",
                "curvePaths": "Curved Polyline",
                "a": "Elliptic Arc",
                "b": "Bézier Curve",
                "c": "Circular Arc",
            }
            not_supported = [v for k, v in curves.items() if k in arcgis["geometry"]]
            if not_supported:
                logger.warning(
                    "Elements of type {} can not be converted. Converting to null geometry",
                    ", ".join(not_supported),
                )
                geojson["geometry"] = None
            else:
                geojson["geometry"] = convert(arcgis["geometry"])
        else:
            geojson["geometry"] = None

        return geojson

    def attributes(self, arcgis: dict[str, Any], geojson: dict[str, Any]) -> dict[str, Any]:
        """Get the attributes from an ArcGIS JSON object."""
        geojson["properties"] = arcgis["attributes"]
        keys = {self.id_attr, "OBJECTID", "FID"} if self.id_attr else {"OBJECTID", "FID"}
        key = list(itertools.dropwhile(lambda k: arcgis["attributes"].get(k) is None, keys))
        if key:
            geojson["id"] = arcgis["attributes"][key[0]]
        return geojson

    @staticmethod
    def get_outer_rings(
        rings: list[Any],
    ) -> tuple[list[Any], list[Any]]:
        """Get outer rings and holes in a list of rings."""
        outer_rings = []
        holes = []
        for ring in rings:
            if not np.all(np.isclose(ring[0], ring[-1])):
                ring.append(ring[0])

            if len(ring) < 4:
                continue

            total = sum(
                (pt2[0] - pt1[0]) * (pt2[1] + pt1[1]) for pt1, pt2 in zip(ring[:-1], ring[1:])
            )
            # Clock-wise check
            if total >= 0:
                # wind outer rings counterclockwise for RFC 7946 compliance
                outer_rings.append([ring[::-1]])
            else:
                # wind inner rings clockwise for RFC 7946 compliance
                holes.append(ring[::-1])
        return outer_rings, holes

    @staticmethod
    def get_uncontained_holes(outer_rings: list[Any], holes: list[Any]) -> list[Any]:
        """Get all the uncontstrained holes."""
        uncontained_holes = []

        while holes:
            hole = holes.pop()

            contained = False
            x = len(outer_rings) - 1
            while x >= 0:
                outer_ring = outer_rings[x][0]
                l1, l2 = LineString(outer_ring), LineString(hole)
                p2 = Point(hole[0])
                intersects = l1.intersects(l2)
                contains = l1.contains(p2)
                if not intersects and contains:
                    outer_rings[x].append(hole)
                    contained = True
                    break
                x = x - 1

            # ring is not contained in any outer ring
            # sometimes this happens https://github.com/Esri/esri-leaflet/issues/320
            if not contained:
                uncontained_holes.append(hole)
        return uncontained_holes

    @staticmethod
    def isnumber(nums: list[Any]) -> bool:
        """Check if all items in a list are numbers."""
        return all(isinstance(n, Number) for n in nums)


def convert(arcgis: dict[str, Any], id_attr: str | None = None) -> dict[str, Any]:
    """Convert an ArcGIS JSON object to a GeoJSON object."""
    togeojson = Convert(id_attr)
    geojson: dict[str, Any] = {}

    keys = ["features", "xy", "points", "paths", "rings", "coords"]
    for k in keys:
        if arcgis.get(k) is not None or k in ("xy", "coords"):
            geojson = togeojson.__getattribute__(k)(arcgis, geojson)

    if "geometry" in arcgis or "attributes" in arcgis:
        geojson["type"] = "Feature"
        geojson = togeojson.geometry(arcgis, geojson)

        if arcgis.get("attributes") is not None:
            geojson = togeojson.attributes(arcgis, geojson)
        else:
            geojson["properties"] = None

    if json.dumps(geojson.get("geometry")) == json.dumps({}):
        geojson["geometry"] = None

    return geojson


class Attrs(NamedTuple):
    """Attributes of a GTiff byte response."""

    nodata: NUMBER
    crs: pyproj.CRS
    dims: tuple[str, str]
    transform: tuple[float, float, float, float, float, float]


def get_nodata(src: Any) -> NUMBER:
    """Get the nodata value of a GTiff byte response."""
    if src.nodata is None:
        try:
            nodata: NUMBER = np.iinfo(src.dtypes[0]).max
        except ValueError:
            nodata = np.nan
    else:
        nodata = np.dtype(src.dtypes[0]).type(src.nodata)
    return nodata


def get_dim_names(ds: xr.DataArray | xr.Dataset) -> tuple[str, str] | None:
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


def get_bounds(
    ds: xr.Dataset | xr.DataArray,
    ds_dims: tuple[str, str] = ("y", "x"),
) -> tuple[float, float, float, float]:
    """Get bounds of a ``xarray.Dataset`` or ``xarray.DataArray``.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The dataset(array) to be masked
    ds_dims : tuple, optional
        Names of the coordinames in the dataset, defaults to ``("y", "x")``.
        The order of the dimension names must be (vertical, horizontal).

    Returns
    -------
    tuple
        The bounds in the order of (left, bottom, right, top)
    """
    ydim, xdim = ds_dims
    left, right = ds[xdim].min().item(), ds[xdim].max().item()
    bottom, top = ds[ydim].min().item(), ds[ydim].max().item()
    return left, bottom, right, top


def transform2tuple(transform: rio.Affine) -> tuple[float, float, float, float, float, float]:
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
