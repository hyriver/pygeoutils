"""Top-level package for PyGeoUtils."""
from importlib.metadata import PackageNotFoundError, version

from ._utils import transform2tuple
from .exceptions import (
    EmptyResponseError,
    InputRangeError,
    InputTypeError,
    InputValueError,
    MissingAttributeError,
    MissingColumnError,
    MissingCRSError,
    UnprojectedCRSError,
)
from .print_versions import show_versions
from .pygeoutils import (
    Coordinates,
    GeoBSpline,
    arcgis2geojson,
    break_lines,
    geo2polygon,
    geometry_list,
    get_gtiff_attrs,
    get_transform,
    gtiff2xarray,
    json2geodf,
    snap2nearest,
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
    "json2geodf",
    "transform2tuple",
    "xarray_geomask",
    "Coordinates",
    "GeoBSpline",
    "InputTypeError",
    "InputValueError",
    "InputRangeError",
    "MissingAttributeError",
    "MissingColumnError",
    "MissingCRSError",
    "UnprojectedCRSError",
    "EmptyResponseError",
    "show_versions",
]
