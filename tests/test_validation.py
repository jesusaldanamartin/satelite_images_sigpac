import sys,os
from validation import raster_comparison
import rasterio
import unittest.mock as mock

sys.path.append(os.path.join(os.path.dirname(__file__),os.pardir,"validation"))

INPUT_RASTER = "../satelite_images_sigpac/tests/resources/test_raster.tif"
INPUT_SHP = "../satelite_images_sigpac/tests/resources/example_shp.shp"
OUTPUT_DATA = "../satelite_images_sigpac/tests/resources/"

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