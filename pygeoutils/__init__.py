"""Top-level package for PyGeoUtils."""
from importlib.metadata import PackageNotFoundError, version

from pygeoutils._utils import get_gtiff_attrs, get_transform, transform2tuple, xd_write_crs
from pygeoutils.exceptions import (
    EmptyResponseError,
    InputRangeError,
    InputTypeError,
    InputValueError,
    MatchingCRSError,
    MissingAttributeError,
    MissingColumnError,
    MissingCRSError,
)
from pygeoutils.geotools import (
    Coordinates,
    GeoBSpline,
    break_lines,
    bspline_curvature,
    coords_list,
    geo2polygon,
    geometry_list,
    geometry_reproject,
    make_bspline,
    multi2poly,
    nested_polygons,
    query_indices,
    smooth_linestring,
    snap2nearest,
)
from pygeoutils.print_versions import show_versions
from pygeoutils.pygeoutils import (
    arcgis2geojson,
    geodf2xarray,
    gtiff2xarray,
    json2geodf,
    xarray2geodf,
    xarray_geomask,
)

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
    "snap2nearest",
    "xarray2geodf",
    "multi2poly",
    "geodf2xarray",
    "json2geodf",
    "transform2tuple",
    "xd_write_crs",
    "xarray_geomask",
    "coords_list",
    "Coordinates",
    "GeoBSpline",
    "bspline_curvature",
    "make_bspline",
    "smooth_linestring",
    "query_indices",
    "nested_polygons",
    "geometry_reproject",
    "InputTypeError",
    "InputValueError",
    "InputRangeError",
    "MissingAttributeError",
    "MissingColumnError",
    "MissingCRSError",
    "MatchingCRSError",
    "UnprojectedCRSError",
    "EmptyResponseError",
    "show_versions",
]
