"""Top-level package for PyGeoUtils."""
from pkg_resources import DistributionNotFound, get_distribution

from .exceptions import InvalidInputType
from .pygeoutils import (
    MatchCRS,
    arcgis2geojson,
    check_bbox,
    geo2polygon,
    gtiff2xarray,
    json2geodf,
    xarray_geomask,
)

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
