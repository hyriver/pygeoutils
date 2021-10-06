"""Top-level package for PyGeoUtils."""
from pkg_resources import DistributionNotFound, get_distribution

from .exceptions import EmptyResponse, InvalidInputType, InvalidInputValue, MissingAttribute
from .print_versions import show_versions
from .pygeoutils import (
    arcgis2geojson,
    geo2polygon,
    get_gtiff_attrs,
    get_transform,
    gtiff2xarray,
    json2geodf,
    xarray2geodf,
    xarray_geomask,
)

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "999"

__all__ = [
    "arcgis2geojson",
    "geo2polygon",
    "get_gtiff_attrs",
    "get_transform",
    "gtiff2xarray",
    "xarray2geodf",
    "json2geodf",
    "xarray_geomask",
    "InvalidInputType",
    "InvalidInputValue",
    "MissingAttribute",
    "EmptyResponse",
    "show_versions",
]
