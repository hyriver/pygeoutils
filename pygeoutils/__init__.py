"""Top-level package for PyGeoUtils."""
from importlib.metadata import PackageNotFoundError, version

if int(version("shapely").split(".")[0]) > 1:
    import os

    os.environ["USE_PYGEOS"] = "0"

from pygeoutils._utils import get_gtiff_attrs, transform2tuple
from pygeoutils.exceptions import (
    EmptyResponseError,
    InputRangeError,
    InputTypeError,
    InputValueError,
    MatchingCRSError,
    MissingAttributeError,
    MissingColumnError,
    MissingCRSError,
    UnprojectedCRSError,
)
from pygeoutils.print_versions import show_versions
from pygeoutils.pygeoutils import (
    Coordinates,
    GeoBSpline,
    arcgis2geojson,
    break_lines,
    coords_list,
    geo2polygon,
    geodf2xarray,
    geometry_list,
    get_transform,
    gtiff2xarray,
    json2geodf,
    nested_polygons,
    query_indices,
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
    "geodf2xarray",
    "json2geodf",
    "transform2tuple",
    "xarray_geomask",
    "coords_list",
    "Coordinates",
    "GeoBSpline",
    "query_indices",
    "nested_polygons",
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
