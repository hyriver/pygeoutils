import pytest

import pygeoutils as utils
from pygeoutils import InvalidInputType, InvalidInputValue

DEF_CRS = "epsg:4326"


def test_invalid_json2geodf_type():
    with pytest.raises(InvalidInputType) as ex:
        _ = utils.json2geodf("")
    assert "content" in str(ex.value)


def test_json2geodf_empty():
    with pytest.raises(StopIteration) as ex:
        _ = utils.json2geodf([])
    assert "Resnpose is empty." in str(ex.value)


def test_gtiff2xarray_type():
    with pytest.raises(InvalidInputType) as ex:
        _ = utils.gtiff2xarray([], (0, 0, 0, 0), DEF_CRS)
    assert "Response.content" in str(ex.value)


def test_gtiff2xarray_empty():
    with pytest.raises(StopIteration) as ex:
        _ = utils.gtiff2xarray({}, (0, 0, 0, 0), DEF_CRS)
    assert "Resnpose dict is empty." in str(ex.value)
