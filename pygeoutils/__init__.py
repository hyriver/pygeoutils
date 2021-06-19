"""Top-level package for PyGeoUtils."""
from pkg_resources import DistributionNotFound, get_distribution

from .exceptions import InvalidInputType, InvalidInputValue
from .print_versions import show_versions
from .pygeoutils import MatchCRS, arcgis2geojson, gtiff2xarray, json2geodf

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
