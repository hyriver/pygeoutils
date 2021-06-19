=======
History
=======

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
- Add announcement regarding the new name for the softwate stack, HyRiver.
- Improve ``pip`` installation and release workflow.

0.10.0 (2021-03-06)
-------------------

- The first release after renaming ``hydrodata`` to ``pygeohydro``.
- Address cheginit/py3dep#1 by sorting y coordinate after merge.
- Make ``mypy`` checks more strict and fix all the errors and prevent possible
  bugs.
- Speed up CI testing by using ``mamba`` and caching.

0.9.0 (2021-02-14)
------------------

- Bump version to the same version as pygeohydro.
- Add ``gtiff2file`` for saving raster responses as ``geotiff`` file(s).
- Fix an error in ``_get_nodata_crs`` for handling nodata value when its value in the source
  is None.
- Fix the warning during the ``GeoDataFrame`` generation in ``json2geodf`` when there is
  no geometry column in the input json.

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
- Replaced ``simplejson`` with ``orjson`` to speed up json operations.


0.1.9 (2020-08-11)
------------------

- Modified ``griff2xarray`` to reflect the latest changes in ``pygeoogc`` 0.1.7.

0.1.8 (2020-08-03)
------------------

- Retained the compatibility with ``xarray`` 0.15 by removing the ``attrs`` flag.
- Added ``xarray_geomask`` function and made it a public function.
- More efficient handling of large geotiff responses by croping the response before
  converting it into a dataset.
- Added a new function called ``geo2polygon`` for converting and transforming
  a polygon or bounding box into a Shapley's Polygon in the target CRS.

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
