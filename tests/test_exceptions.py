import pytest
from shapely.geometry import Polygon

import pygeoutils as geoutils
from pygeoutils import EmptyResponse, InvalidInputType, InvalidInputValue

try:
    import typeguard  # noqa: F401
except ImportError:
    has_typeguard = False
else:
    has_typeguard = True

DEF_CRS = "epsg:4326"
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
DEF_CRS = "epsg:4326"


@pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
def test_invalid_json2geodf_type():
    with pytest.raises(InvalidInputType) as ex:
        _ = geoutils.json2geodf("")
    assert "content" in str(ex.value)


def test_json2geodf_empty():
    with pytest.raises(EmptyResponse) as ex:
        _ = geoutils.json2geodf([])
    assert "The input response is empty." in str(ex.value)


@pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
def test_gtiff2xarray_type():
    with pytest.raises(InvalidInputType) as ex:
        _ = geoutils.gtiff2xarray([], (0, 0, 0, 0), DEF_CRS)
    assert "bytes" in str(ex.value)


def test_gtiff2xarray_empty():
    with pytest.raises(EmptyResponse) as ex:
        _ = geoutils.gtiff2xarray({}, (0, 0, 0, 0), DEF_CRS)
    assert "The input response is empty." in str(ex.value)


class TestX2GFail:
    @pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
    def test_wrong_array_type_da(self, wms_resp):
        canopy = geoutils.gtiff2xarray(wms_resp, GEO_NAT, DEF_CRS)
        mask = canopy > 60
        with pytest.raises(InvalidInputType) as ex:
            _ = geoutils.xarray2geodf(canopy.to_dataset(), "float32", mask)
        assert "DataArray" in str(ex.value)

    @pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
    def test_unsupported_dtype(self, wms_resp):
        canopy = geoutils.gtiff2xarray(wms_resp, GEO_NAT, DEF_CRS)
        mask = canopy > 60
        with pytest.raises(InvalidInputValue) as ex:
            _ = geoutils.xarray2geodf(canopy, "float64", mask)
        assert "float32" in str(ex.value)

    @pytest.mark.skipif(has_typeguard, reason="Broken if Typeguard is enabled")
    def test_wrong_array_type_mask(self, wms_resp):
        canopy = geoutils.gtiff2xarray(wms_resp, GEO_NAT, DEF_CRS)
        mask = canopy > 60
        with pytest.raises(InvalidInputType) as ex:
            _ = geoutils.xarray2geodf(canopy, "float32", mask.to_dataset())
        assert "DataArray" in str(ex.value)
