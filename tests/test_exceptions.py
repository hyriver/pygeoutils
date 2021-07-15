import pytest

import pygeoutils as utils
from pygeoutils import EmptyResponse, InvalidInputType

DEF_CRS = "epsg:4326"


def test_invalid_json2geodf_type():
    with pytest.raises(InvalidInputType) as ex:
        _ = utils.json2geodf("")
    assert "content" in str(ex.value)


def test_json2geodf_empty():
    with pytest.raises(EmptyResponse) as ex:
        _ = utils.json2geodf([])
    assert "The input response is empty." in str(ex.value)


def test_gtiff2xarray_type():
    with pytest.raises(InvalidInputType) as ex:
        _ = utils.gtiff2xarray([], (0, 0, 0, 0), DEF_CRS)
    assert "bytes" in str(ex.value)


def test_gtiff2xarray_empty():
    with pytest.raises(EmptyResponse) as ex:
        _ = utils.gtiff2xarray({}, (0, 0, 0, 0), DEF_CRS)
    assert "The input response is empty." in str(ex.value)
