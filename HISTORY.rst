=======
History
=======

0.1.10 (2020-08-18)
-------------------

- Fixed the ``gtiff2xarray`` issue with high resolution requests and improved robustness
  of the function.
- Replaces ``simplejson`` with ``orjson`` to speed up json operations.


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
- Add the missing dependecy

0.1.0 (2020-07-21)
------------------

- First release on PyPI.
