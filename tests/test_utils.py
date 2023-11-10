
import sys,os
import pytest
import unittest.mock as mock
import rasterio

from src.validation import raster_comparison

from numba import jit
from utils import *

sys.path.append(os.path.join(os.path.dirname(__file__),os.pardir,"validation"))

INPUT_RASTER = "../satelite_images_sigpac/tests/resources/test_raster.tif"
INPUT_SHP = "../satelite_images_sigpac/tests/resources/example_shp.shp"
OUTPUT_DATA = "../satelite_images_sigpac/tests/resources/"


def test_reproject_raster_crs():
    # create a mock reproject function that simply copies the input data to the output data
    def mock_reproject(*args):
        dst = args[1]
        with rasterio.open(args[0]) as src:
            data = src.read(1)
            dst.write(data, 1)
        return None

    # replace the reproject function with the mock function
    with mock.patch('utils.reproject_raster', side_effect=mock_reproject) as mock_function:
        # call the reproject_raster function
        reproject_raster(INPUT_RASTER, OUTPUT_DATA, 'output_data.tif', 'EPSG:4258')

        # check that the CRS is set correctly in the output raster
        assert os.path.exists(OUTPUT_DATA + 'output_data.tif')
        with rasterio.open(OUTPUT_DATA + 'output_data.tif') as src:
            assert src.crs.to_string() == 'EPSG:4258'


def test_get_id_codigo_uso():

    code = 'FO'
    expected_id = 11
    assert expected_id == get_id_codigo_uso(code)
    assert 300 != get_id_codigo_uso(code)    


def test_is_point_in_polygon():

    polygon = [[0,0], [20,0], [0,20], [20,20]]
    point = (3,4)
    point1 = (0,0)
    point2 = (20,20)
    point3 = (101,202)

    assert is_point_in_polygon(point1[0], point1[1], polygon) is True
    assert is_point_in_polygon(point2[0], point2[1], polygon) is True
    assert is_point_in_polygon(point3[0], point3[1], polygon) is False


def test_raster_comparison():

    # create a mock reproject function that simply copies the input data to the output data
    def mock_reproject(*args):
        dst = args[1]
        with rasterio.open(args[0]) as src:
            data = src.read(1)
            dst.write(data, 1)
        return None

    # replace the reproject function with the mock function
    with mock.patch('utils.reproject_raster', side_effect=mock_reproject) as mock_function:
        # call the reproject_raster function
        raster_comparison(INPUT_RASTER, OUTPUT_DATA, 'output_data.tif', 'EPSG:4258')

        # check that the CRS is set correctly in the output raster
        assert os.path.exists(OUTPUT_DATA + 'output_data.tif')
        with rasterio.open(OUTPUT_DATA + 'output_data.tif') as src:
            assert src.crs.to_string() == 'EPSG:4258'

    return
