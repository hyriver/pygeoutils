.. image:: https://raw.githubusercontent.com/cheginit/HyRiver-examples/main/notebooks/_static/pygeoutils_logo.png
    :target: https://github.com/cheginit/HyRiver

|

.. image:: https://joss.theoj.org/papers/b0df2f6192f0a18b9e622a3edff52e77/status.svg
    :target: https://joss.theoj.org/papers/b0df2f6192f0a18b9e622a3edff52e77
    :alt: JOSS

|

.. |pygeohydro| image:: https://github.com/cheginit/pygeohydro/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pygeohydro/actions/workflows/test.yml
    :alt: Github Actions

.. |pygeoogc| image:: https://github.com/cheginit/pygeoogc/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pygeoogc/actions/workflows/test.yml
    :alt: Github Actions

.. |pygeoutils| image:: https://github.com/cheginit/pygeoutils/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pygeoutils/actions/workflows/test.yml
    :alt: Github Actions

.. |pynhd| image:: https://github.com/cheginit/pynhd/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pynhd/actions/workflows/test.yml
    :alt: Github Actions

.. |py3dep| image:: https://github.com/cheginit/py3dep/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/py3dep/actions/workflows/test.yml
    :alt: Github Actions

.. |pydaymet| image:: https://github.com/cheginit/pydaymet/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/pydaymet/actions/workflows/test.yml
    :alt: Github Actions

.. |async| image:: https://github.com/cheginit/async_retriever/actions/workflows/test.yml/badge.svg
    :target: https://github.com/cheginit/async_retriever/actions/workflows/test.yml
    :alt: Github Actions

=============== ==================================================================== ============
Package         Description                                                          Status
=============== ==================================================================== ============
PyNHD_          Navigate and subset NHDPlus (MR and HR) using web services           |pynhd|
Py3DEP_         Access topographic data through National Map's 3DEP web service      |py3dep|
PyGeoHydro_     Access NWIS, NID, WQP, HCDN 2009, NLCD, CAMELS, and SSEBop databases |pygeohydro|
PyDaymet_       Access Daymet for daily climate data both single pixel and gridded   |pydaymet|
AsyncRetriever_ High-level API for asynchronous requests with persistent caching     |async|
PyGeoOGC_       Send queries to any ArcGIS RESTful-, WMS-, and WFS-based services    |pygeoogc|
PyGeoUtils_     Convert responses from PyGeoOGC's supported web services to datasets |pygeoutils|
=============== ==================================================================== ============

.. _PyGeoHydro: https://github.com/cheginit/pygeohydro
.. _AsyncRetriever: https://github.com/cheginit/async_retriever
.. _PyGeoOGC: https://github.com/cheginit/pygeoogc
.. _PyGeoUtils: https://github.com/cheginit/pygeoutils
.. _PyNHD: https://github.com/cheginit/pynhd
.. _Py3DEP: https://github.com/cheginit/py3dep
.. _PyDaymet: https://github.com/cheginit/pydaymet

PyGeoUtils: Utilities for (Geo)JSON and (Geo)TIFF Conversion
------------------------------------------------------------

.. image:: https://img.shields.io/pypi/v/pygeoutils.svg
    :target: https://pypi.python.org/pypi/pygeoutils
    :alt: PyPi

.. image:: https://img.shields.io/conda/vn/conda-forge/pygeoutils.svg
    :target: https://anaconda.org/conda-forge/pygeoutils
    :alt: Conda Version

.. image:: https://codecov.io/gh/cheginit/pygeoutils/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/cheginit/pygeoutils
    :alt: CodeCov

.. image:: https://img.shields.io/pypi/pyversions/pygeoutils.svg
    :target: https://pypi.python.org/pypi/pygeoutils
    :alt: Python Versions

.. image:: https://pepy.tech/badge/pygeoutils
    :target: https://pepy.tech/project/pygeoutils
    :alt: Downloads

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

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/cheginit/HyRiver-examples/main?urlpath=lab/tree/notebooks
    :alt: Binder

|

Features
--------

PyGeoUtils is a part of `HyRiver <https://github.com/cheginit/HyRiver>`__ software stack that
is designed to aid in hydroclimate analysis through web services. This package provides
utilities for manipulating (Geo)JSON and (Geo)TIFF responses from web services.
These utilities are:

- ``json2geodf``: For converting (Geo)JSON objects to GeoPandas dataframe.
- ``arcgis2geojson``: For converting ESRIGeoJSON to the standard GeoJSON format.
- ``gtiff2xarray``: For converting (Geo)TIFF objects to `xarray <https://xarray.pydata.org/>`__
  datasets.
- ``xarray2geodf``: For converting ``xarray.DataArray`` to a ``geopandas.GeoDataFrame``, i.e.,
  vectorization.
- ``xarray_geomask``: For masking a ``xarray.Dataset`` or ``xarray.DataArray`` using a polygon.

All these functions handle all necessary CRS transformations.

You can find some example notebooks `here <https://github.com/cheginit/HyRiver-examples>`__.

You can also try using PyGeoUtils without installing
it on your system by clicking on the binder badge. A Jupyter Lab
instance with the HyRiver stack pre-installed will be launched in your web browser, and you
can start coding!

Please note that since this project is in the early development stages, while the provided
functionalities should be stable, changes in APIs are possible in new releases. But we
appreciate it if you give this project a try and provide feedback. Contributions are most welcome.

Moreover, requests for additional functionalities can be submitted via
`issue tracker <https://github.com/cheginit/pygeoutils/issues>`__.

Installation
------------

You can install PyGeoUtils using ``pip`` after installing ``libgdal`` on your system
(for example, in Ubuntu run ``sudo apt install libgdal-dev``).

.. code-block:: console

    $ pip install pygeoutils

Alternatively, PyGeoUtils can be installed from the ``conda-forge`` repository
using `Conda <https://docs.conda.io/en/latest/>`__:

.. code-block:: console

    $ conda install -c conda-forge pygeoutils

Quick start
-----------

To demonstrate the capabilities of PyGeoUtils let's use
`PyGeoOGC <https://github.com/cheginit/pygeoogc>`__ to access
`National Wetlands Inventory <https://www.fws.gov/wetlands/>`__ from WMS, and
`FEMA National Flood Hazard <https://www.fema.gov/national-flood-hazard-layer-nfhl>`__
via WFS, then convert the output to ``xarray.Dataset`` and ``GeoDataFrame``, respectively.

.. code-block:: python

    import pygeoutils as geoutils
    from pygeoogc import WFS, WMS, ServiceURL
    from shapely.geometry import Polygon


    geometry = Polygon(
        [
            [-118.72, 34.118],
            [-118.31, 34.118],
            [-118.31, 34.518],
            [-118.72, 34.518],
            [-118.72, 34.118],
        ]
    )
    crs = "epsg:4326"

    wms = WMS(
        ServiceURL().wms.mrlc,
        layers="NLCD_2011_Tree_Canopy_L48",
        outformat="image/geotiff",
        crs=crs,
    )
    r_dict = wms.getmap_bybox(
        geometry.bounds,
        1e3,
        box_crs=crs,
    )
    canopy = geoutils.gtiff2xarray(r_dict, geometry, crs)

    mask = canopy > 60
    canopy_gdf = geoutils.xarray2geodf(canopy, "float32", mask)

    url_wfs = "https://hazards.fema.gov/gis/nfhl/services/public/NFHL/MapServer/WFSServer"
    wfs = WFS(
        url_wfs,
        layer="public_NFHL:Base_Flood_Elevations",
        outformat="esrigeojson",
        crs="epsg:4269",
    )
    r = wfs.getfeature_bybox(geometry.bounds, box_crs=crs)
    flood = geoutils.json2geodf(r.json(), "epsg:4269", crs)

Contributing
------------

Contributions are very welcomed. Please read
`CONTRIBUTING.rst <https://github.com/cheginit/pygeoogc/blob/main/CONTRIBUTING.rst>`__
file for instructions.
