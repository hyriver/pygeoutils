from __future__ import annotations

import pytest
from shapely import Polygon

import pygeoutils as geoutils
from pygeoutils import EmptyResponseError, InputTypeError, InputValueError

DEF_CRS = 4326
GEO_URB = Polygon(
    [
        [-118.72, 34.118],
        [-118.31, 34.118],
        [-118.31, 34.518],
        [-118.72, 34.518],
        [-118.72, 34.118],
    ]
)

GEO_NAT = Polygon(
    [[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]]
)
DEF_CRS = 4326


def test_invalid_json2geodf_type():
    with pytest.raises(InputTypeError) as ex:
        _ = geoutils.json2geodf("")
    assert "content" in str(ex.value)


def test_json2geodf_empty():
    with pytest.raises(EmptyResponseError) as ex:
        _ = geoutils.json2geodf([])
    assert "The input response is empty." in str(ex.value)


def test_geom_list_wrong_geom():
    with pytest.raises(InputTypeError) as ex:
        _ = geoutils.geometry_list([-69.77, 45.07])
    assert "length 4" in str(ex.value)


def test_gtiff2xarray_type():
    with pytest.raises(InputTypeError) as ex:
        _ = geoutils.gtiff2xarray([], (0, 0, 0, 0), DEF_CRS)
    assert "bytes" in str(ex.value)


def test_gtiff2xarray_empty():
    with pytest.raises(EmptyResponseError) as ex:
        _ = geoutils.gtiff2xarray({}, (0, 0, 0, 0), DEF_CRS)
    assert "The input response is empty." in str(ex.value)


class TestX2GFail:
    def test_wrong_array_type_da(self, wms_resp):
        canopy = geoutils.gtiff2xarray(wms_resp, GEO_NAT, DEF_CRS)
        mask = canopy > 60
        with pytest.raises(InputTypeError) as ex:
            _ = geoutils.xarray2geodf(canopy.to_dataset(), "float32", mask)
        assert "DataArray" in str(ex.value)

    def test_unsupported_dtype(self, wms_resp):
        canopy = geoutils.gtiff2xarray(wms_resp, GEO_NAT, DEF_CRS)
        mask = canopy > 60
        with pytest.raises(InputValueError) as ex:
            _ = geoutils.xarray2geodf(canopy, "float64", mask)
        assert "float32" in str(ex.value)

    def test_wrong_array_type_mask(self, wms_resp):
        canopy = geoutils.gtiff2xarray(wms_resp, GEO_NAT, DEF_CRS)
        mask = canopy > 60
        with pytest.raises(InputTypeError) as ex:
            _ = geoutils.xarray2geodf(canopy, "float32", mask.to_dataset())
        assert "DataArray" in str(ex.value)
