"""Top-level package for PyGeoUtils."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pygeoutils import exceptions
from pygeoutils._utils import get_gtiff_attrs, get_transform, transform2tuple, xd_write_crs
from pygeoutils.geotools import (
    Coordinates,
    break_lines,
    coords_list,
    geo2polygon,
    geometry_list,
    geometry_reproject,
    multi2poly,
    nested_polygons,
    query_indices,
    snap2nearest,
)
from pygeoutils.print_versions import show_versions
from pygeoutils.pygeoutils import (
    arcgis2geojson,
    geodf2xarray,
    gtiff2vrt,
    gtiff2xarray,
    json2geodf,
    sample_window,
    xarray2geodf,
    xarray_geomask,
)
from pygeoutils.smoothing import (
    GeoSpline,
    anchored_smoothing,
    line_curvature,
    make_spline,
    smooth_linestring,
    smooth_multilinestring,
    spline_curvature,
    spline_linestring,
)

cert_path = os.getenv("HYRIVER_SSL_CERT")
if cert_path is not None:
    from pyproj.network import set_ca_bundle_path

    if not Path(cert_path).exists():
        raise FileNotFoundError(cert_path)
    set_ca_bundle_path(cert_path)

try:
    __version__ = version("pygeoutils")
except PackageNotFoundError:
    __version__ = "999"

__all__ = [
    "arcgis2geojson",
    "break_lines",
    "geo2polygon",
    "geometry_list",
    "get_gtiff_attrs",
    "get_transform",
    "gtiff2xarray",
    "gtiff2vrt",
    "snap2nearest",
    "xarray2geodf",
    "multi2poly",
    "geodf2xarray",
    "json2geodf",
    "transform2tuple",
    "xd_write_crs",
    "xarray_geomask",
    "sample_window",
    "coords_list",
    "Coordinates",
    "query_indices",
    "nested_polygons",
    "geometry_reproject",
    "GeoSpline",
    "make_spline",
    "spline_linestring",
    "smooth_linestring",
    "line_curvature",
    "anchored_smoothing",
    "smooth_multilinestring",
    "spline_curvature",
    "show_versions",
    "exceptions",
    "__version__",
]
