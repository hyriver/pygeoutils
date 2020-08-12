.. image:: https://raw.githubusercontent.com/cheginit/hydrodata/master/docs/_static/pygeoutils_logo.png
    :target: https://github.com/cheginit/pygeoutils
    :align: center

|

=========== ===========================================================================
Package     Description
=========== ===========================================================================
Hydrodata_  Access NWIS, HCDN 2009, NLCD, and SSEBop databases
PyGeoOGC_   Query data from any ArcGIS RESTful-, WMS-, and WFS-based services
PyGeoUtils_ Convert responses from PyGeoOGC's supported web services to datasets
PyNHD_      Access NLDI and WaterData web services for navigating the NHDPlus database
Py3DEP_     Access topographic data through the 3D Elevation Program (3DEP) web service
PyDaymet_   Access the Daymet database for daily climate data
=========== ===========================================================================

.. _Hydrodata: https://github.com/cheginit/hydrodata
.. _PyGeoOGC: https://github.com/cheginit/pygeoogc
.. _PyGeoUtils: https://github.com/cheginit/pygeoutils
.. _PyNHD: https://github.com/cheginit/pynhd
.. _Py3DEP: https://github.com/cheginit/py3dep
.. _PyDaymet: https://github.com/cheginit/pydaymet

PyGeoUtils: Manipulate (Geo)JSON and (Geo)TIFF data
---------------------------------------------------

.. image:: https://img.shields.io/pypi/v/pygeoutils.svg
    :target: https://pypi.python.org/pypi/pygeoutils
    :alt: PyPi

.. image:: https://img.shields.io/conda/vn/conda-forge/pygeoutils.svg
    :target: https://anaconda.org/conda-forge/pygeoutils
    :alt: Conda Version

.. image:: https://codecov.io/gh/cheginit/pygeoutils/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/cheginit/pygeoutils
    :alt: CodeCov

.. image:: https://github.com/cheginit/pygeoutils/workflows/build/badge.svg
    :target: https://github.com/cheginit/pygeoutils/workflows/build
    :alt: Github Actions

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/cheginit/hydrodata/master?filepath=docs%2Fexamples
    :alt: Binder

|

.. image:: https://www.codefactor.io/repository/github/cheginit/pygeoutils/badge
   :target: https://www.codefactor.io/repository/github/cheginit/pygeoutils
   :alt: CodeFactor

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
    :target: https://github.com/pre-commit/pre-commit
    :alt: pre-commit

|

🚨 **This package is under heavy development and breaking changes are likely to happen.** 🚨

Features
--------

PyGeoUtils is a part of Hydrodata software stack and provides utilities for manipulating
(Geo)JSON and (Geo)TIFF data. These utilities are:

- ``json2geodf``: For converting (Geo)JSON objects to GroPandas dataframe.
- ``arcgis2geojson``: For converting ESRIGeoJSON objects to standard GeoJSON format.
- ``gtiff2xarray``: For converting (Geo)TIFF objects to `xarray <https://xarray.pydata.org/>`__
  datasets.
- ``xarray_geomask``: For masking a ``xarray.Dataset`` or ``xarray.DataArray`` using a polygon.

All these function handle all necessary CRS transformations. Moreover, requests for additional
functionalities can be submitted via
`issue tracker <https://github.com/cheginit/pygeoutils/issues>`__.

Installation
------------

You can install PyGeoUtils using ``pip`` after installing ``libgdal`` on your system
(for example, in Ubuntu run ``sudo apt install libgdal-dev``):

.. code-block:: console

    $ pip install pygeoutils

Alternatively, PyGeoUtils can be installed from the ``conda-forge`` repository
using `Conda <https://docs.conda.io/en/latest/>`__:

.. code-block:: console

    $ conda install -c conda-forge pygeoutils

Quick start
-----------

To demonstrate capabilities of PyGeoUtils lets use
`PyGeoOGC <https://github.com/cheginit/pygeoogc>`__ to access
`National Wetlands Inventory <https://www.fws.gov/wetlands/>`__ from WMS, and
`FEMA National Flood Hazard <https://www.fema.gov/national-flood-hazard-layer-nfhl>`__
via WFS, then convert the output to ``GeoDataFrame`` and ``xarray.Dataset`` using PyGeoUtils.

.. code-block:: python

    import pygeoutils as geoutils
    from pygeoogc import WFS, WMS
    from shapely.geometry import Polygon


    geometry =  Polygon(
        [
            [-118.72, 34.118],
            [-118.31, 34.118],
            [-118.31, 34.518],
            [-118.72, 34.518],
            [-118.72, 34.118],
        ]
    )

    url_wms = "https://www.fws.gov/wetlands/arcgis/services/Wetlands_Raster/ImageServer/WMSServer"
    wms = WMS(
        url_wms,
        layers="0",
        outformat="image/tiff",
        crs="epsg:3857",
    )
    r_dict = wms.getmap_bybox(
        geometry.bounds,
        1e3,
        box_crs="epsg:4326",
    )
    wetlands = geoutils.gtiff2xarray(r_dict, geometry, "epsg:4326")

    url_wfs = "https://hazards.fema.gov/gis/nfhl/services/public/NFHL/MapServer/WFSServer"
    wfs = WFS(
        url_wfs,
        layer="public_NFHL:Base_Flood_Elevations",
        outformat="esrigeojson",
        crs="epsg:4269",
    )
    r = wfs.getfeature_bybox(geometry.bounds, box_crs="epsg:4326")
    flood = geoutils.json2geodf(r.json(), "epsg:4269", "epsg:4326")

Contributing
------------

Contributions are very welcomed. Please read
`CONTRIBUTING.rst <https://github.com/cheginit/pygeoogc/blob/master/CONTRIBUTING.rst>`__
file for instructions.
