=======
History
=======

0.13.0 (2022-03-03)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Write ``nodata`` attribute using ``rioxarray`` in ``geotiff2xarray`` since the
  clipping operation of ``rioxarray`` uses this value as fill value.

Bug Fixes
~~~~~~~~~
- In the ``break_lines`` function, convert ``MultiLineString`` into
  ``LineString`` since ``shapely.ops.substring`` only accepts ``LineString``.

0.12.3 (2022-02-04)
-------------------

New Features
~~~~~~~~~~~~
- Add a function called ``break_lines`` for breaking lines at given points.
- Add a function called ``snap2nearest`` for snapping points to the nearest
  point on a line with a given tolerance. It accepts a ``geopandas.GeoSeries`` of
  points and a ``geopandas.GeoSeries`` or ``geopandas.GeoDataFrame`` of lines. It
  automatically snaps to the closest lines in the input data.

0.12.2 (2022-01-15)
-------------------

New Features
~~~~~~~~~~~~
- Add a new class called ``GeoBSpline`` that generates B-splines from a set of
  coordinates. The ``spline`` attribute of this class has five attributes:
  ``x`` and ``y`` coordinates, ``phi`` and ``radius`` which are curvature and
  radius of curvature, respectively, and ``distance`` which is the total distance
  of each point along the B-spline from the starting points.
- Add a new class called ``Coordinates`` that validates a set of lon/lat coordinates.
  It normalizes longitudes to the range [-180, 180) and has a ``points`` property
  that is ``geopandas.GeoSeries`` with validated coordinates. It uses spatial indexing
  to speed up the validation and should be able to handle large datasets efficiently.
- Make ``transform2tuple`` a public function.

Internal Changes
~~~~~~~~~~~~~~~~
- The ``geometry`` and ``geo_crs`` arguments of ``gtiff2xarray`` are now optional.
  This is useful for cases when the input ``GeoTiff`` response is the results of
  a bounding box query and there is no need for a geometry mask.
- Replace the missing values after adding geometry mask via ``xarray_geomask`` by the
  ``nodatavals`` attribute of the input ``xarray.DataArray`` or ``xarray.Dataset``.
  Therefore, the data type of the input ``xarray.DataArray`` or ``xarray.Dataset``
  is conserved.
- Expose ``connectivity`` argument of ``rasterio.features.shapes`` function in
  ``xarray2geodf`` function.
- Move all private functions to a new module to make the main module less cluttered.

0.12.1 (2021-12-31)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Refactor ``arcgis2geojson`` for better readability and maintainability.
- In ``arcgis2geojson`` set the geometry to null if its type is not supported,
  such as curved polylines.

0.12.0 (2021-12-27)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Add all the missing types so ``mypy --strict`` passes.
- Bump version to 0.12.0 to match the release of ``pygeoogc``.

0.11.7 (2021-11-09)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Use ``rioxarray`` for dealing with ``GeoTIFF`` binaries since ``xarray``
  deprecated the ``xarray.open_rasterio`` function, as it's discussed
  in this `PR <https://github.com/pydata/xarray/pull/5808>`__.
- Use ``importlib-metadata`` for getting the version instead of ``pkg_resources``
  to decrease import time as discussed in this
  `issue <https://github.com/pydata/xarray/issues/5676>`__.

0.11.6 (2021-10-06)
-------------------

New Features
~~~~~~~~~~~~
- Add a new function, ``xarray2geodf``, to convert a ``xarray.DataArray`` to a
  ``geopandas.GeoDataFrame``.

0.11.5 (2021-06-16)
-------------------

Bug Fixes
~~~~~~~~~
- Fix an issue with ``gtiff2xarray`` where the ``scales`` and ``offsets``
  attributes of the output ``DataArray`` were floats rather than tuples (:issue_3dep:`30`).

Internal Changes
~~~~~~~~~~~~~~~~
- Add a new function, ``transform2tuple``, for converting ``Affine`` transforms to a tuple.
  Previously, the ``Affine`` transform was converted to a tuple using ``to_gdal()`` method
  of ``rasterio.Affine`` which was not compatible with ``rioxarray``.

0.11.4 (2021-08-26)
-------------------

Internal Changes
~~~~~~~~~~~~~~~~
- Use ``ujson`` for JSON parsing instead of ``orjson`` since ``orjson`` only serializes to
  ``bytes`` which is not compatible with ``aiohttp``.
- Convert the transform attribute data type from ``Affine`` to ``tuple`` since saving a data
  array to ``netcdf`` cannot handle the ``Affine`` type.

0.11.3 (2021-08-19)
-------------------

- Fix an issue in ``geotiff2xarray`` related to saving a ``xarray`` object to NetCDF when its
  transform attribute has ``Affine`` type rather than a tuple.

0.11.2 (2021-07-31)
-------------------

The highlight of this release is performance improvement in ``gtiff2xarray`` for
handling large responses.

New Features
~~~~~~~~~~~~
- Automatic detection of the driver by default in ``gtiff2xarray`` as opposed to it being
  ``GTiff``.

Internal Changes
~~~~~~~~~~~~~~~~
- Make ``geo2polygon``, ``get_transform``, and ``get_nodata_crs`` public functions
  since other packages use it.
- Make ``xarray_mask`` a public function and simplify ``gtiff2xarray``.
- Remove ``MatchCRS`` since it's already available in ``pygeoogc``.
- Validate input geometry in ``geo2polygon``.
- Refactor ``gtiff2xarray`` to check for the ``ds_dims`` outside the main loops to
  improve the performance. Also, the function tries to detect the dimension names
  automatically if ``ds_dims`` is not provided by the user, explicitly.
- Improve performance of ``json2geodf`` by using list comprehension and performing
  checks outside the main loop.

Bug Fixes
~~~~~~~~~
- Add the missing arguments for masking the data in ``gtiff2xarray``.

0.11.1 (2021-06-19)
-------------------

Bug Fixes
~~~~~~~~~
- In some edge cases the y-coordinates of a response might not be monotonically sorted so
  ``dask`` fails. This release sorts them to address this issue.

0.11.0 (2021-06-19)
-------------------

New Features
~~~~~~~~~~~~
- Function ``gtiff2xarray`` returns a parallelized ``xarray.Dataset`` or ``xarray.DataAraay``
  that can handle large responses much more efficiently. This is achieved using ``dask``.

Breaking Changes
~~~~~~~~~~~~~~~~
- Drop support for Python 3.6 since many of the dependencies such as ``xarray`` and ``pandas``
  have done so.
- Refactor ``MatchCRS``. Now, it should be instantiated by providing the in and out CRSs like so:
  ``MatchCRS(in_crs, out_crs)``. Then its methods, namely, ``geometry``, ``bounds`` and ``coords``,
  can be called. These methods now have only one input, geometry.
- Change input and output types of ``MatchCRS.coords`` from tuple of lists of coordinates
  to list of ``(x, y)`` coordinates.
- Remove ``xarray_mask`` and ``gtiff2file`` since ``rioxarray`` is more general and suitable.

Internal Changes
~~~~~~~~~~~~~~~~
- Remove unnecessary type checks for private functions.
- Refactor ``json2geodf`` to improve robustness. Use ``get`` method of ``dict`` for checking
  key availability.

0.10.1 (2021-03-27)
-------------------

- Setting transform of the merged dataset explicitly (:issue_3dep:`3`).
- Add announcement regarding the new name for the software stack, HyRiver.
- Improve ``pip`` installation and release workflow.

0.10.0 (2021-03-06)
-------------------

- The first release after renaming ``hydrodata`` to ``PyGeoHydro``.
- Address :issue_3dep:`1` by sorting y coordinate after merge.
- Make ``mypy`` checks more strict and fix all the errors and prevent possible
  bugs.
- Speed up CI testing by using ``mamba`` and caching.

0.9.0 (2021-02-14)
------------------

- Bump version to the same version as PyGeoHydro.
- Add ``gtiff2file`` for saving raster responses as ``geotiff`` file(s).
- Fix an error in ``_get_nodata_crs`` for handling no data value when its value in the source
  is None.
- Fix the warning during the ``GeoDataFrame`` generation in ``json2geodf`` when there is
  no geometry column in the input JSON.

0.2.0 (2020-12-06)
-------------------

- Added checking the validity of input arguments in ``gtiff2xarray`` function and provide
  useful messages for debugging.
- Add support for multipolygon.
- Remove the ``fill_hole`` argument.
- Fixed a bug in ``xarray_geomask`` for getting the transform.

0.1.10 (2020-08-18)
-------------------

- Fixed the ``gtiff2xarray`` issue with high resolution requests and improved robustness
  of the function.
- Replaced ``simplejson`` with ``orjson`` to speed up JSON operations.


0.1.9 (2020-08-11)
------------------

- Modified ``griff2xarray`` to reflect the latest changes in ``pygeoogc`` 0.1.7.

0.1.8 (2020-08-03)
------------------

- Retained the compatibility with ``xarray`` 0.15 by removing the ``attrs`` flag.
- Added ``xarray_geomask`` function and made it a public function.
- More efficient handling of large GeoTiff responses by cropping the response before
  converting it into a dataset.
- Added a new function called ``geo2polygon`` for converting and transforming
  a polygon or bounding box into a Shapely's Polygon in the target CRS.

0.1.6 (2020-07-23)
------------------

- Fixed the issue with flipped mask in ``WMS``.
- Removed ``drop_duplicates`` since it may cause issues in some instances.


0.1.4 (2020-07-22)
------------------

- Refactor ``griff2xarray`` and added support for WMS 1.3.0 and WFS 2.0.0.
- Add ``MatchCRS`` class.
- Remove dependency on PyGeoOGC.
- Increase test coverage.

0.1.3 (2020-07-21)
------------------

- Remove duplicate rows before returning the dataframe in the ``json2geodf`` function.
- Add the missing dependency

0.1.0 (2020-07-21)
------------------

- First release on PyPI.
