"""Top-level package for PyGeoUtils."""
from .exceptions import (
    EmptyResponse,
    InvalidInputRange,
    InvalidInputType,
    InvalidInputValue,
    MissingAttribute,
    MissingCRS,
)
from .print_versions import show_versions
from .pygeoutils import (
    Coordinates,
    GeoBSpline,
    arcgis2geojson,
    geo2polygon,
    get_gtiff_attrs,
    get_transform,
    gtiff2xarray,
    json2geodf,
    xarray2geodf,
    xarray_geomask,
)
from .utils import transform2tuple

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
    "geo2polygon",
    "get_gtiff_attrs",
    "get_transform",
    "gtiff2xarray",
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
    "MissingCRS",
    "EmptyResponse",
    "show_versions",
]
