"""Top-level package for PyNHD."""
from pkg_resources import DistributionNotFound, get_distribution

from .exceptions import InvalidInputType
from .pygeoutils import arcgis2geojson, gtiff2xarray, json2geodf

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
