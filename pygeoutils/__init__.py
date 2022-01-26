"""Top-level package for PyGeoUtils."""
from ._utils import transform2tuple
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
from .print_versions import show_versions
from .pygeoutils import (
    Coordinates,
    GeoBSpline,
    arcgis2geojson,
    break_lines,
    geo2polygon,
    get_gtiff_attrs,
    get_transform,
    gtiff2xarray,
    json2geodf,
    snap2nearest,
    xarray2geodf,
    xarray_geomask,
)

try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata  # type: ignore[no-redef]

try:
    __version__ = metadata.version("pygeoutils")
except Exception:
    __version__ = "999"

__all__ = [
    "arcgis2geojson",
    "break_lines",
    "geo2polygon",
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
    "InvalidInputType",
    "InvalidInputValue",
    "InvalidInputRange",
    "MissingAttribute",
    "MissingColumns",
    "MissingCRS",
    "UnprojectedCRS",
    "EmptyResponse",
    "show_versions",
]
