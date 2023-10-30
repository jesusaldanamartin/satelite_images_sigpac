import sys
import os
import shutil
import logging
import rasterio
import fiona
import warnings

from pathlib import Path
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

from utils import mask_shp, save_output_file, reproject_raster
from validation import read_needed_files, raster_comparison, raster_comparison_confmatrix, create_dataframe_and_graphs

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TIFF_PATH = sys.argv[1]
SHP_PATH = sys.argv[2]
OUT_FILENAME = sys.argv[3]
TEMPORARY_FILES = sys.argv[4]

TMP_PATH = "/app/data/tmp/"
OUTPUT = "/app/data/results/"

logging.warning("TIME TO FINISH")

try:
    with rasterio.open(TIFF_PATH) as raster:
        tif_crs = raster.crs

    with fiona.open(SHP_PATH) as shapefile:
        shp_crs = shapefile.crs

    if tif_crs != shp_crs:
        logging.info("Files have different CRS")
        reproject_raster(TIFF_PATH, TMP_PATH,
                         "reference_reprojected.tif", shp_crs)
        TIFF_PATH = TMP_PATH+"reference_reprojected.tif"

except (SystemExit, KeyboardInterrupt):
    raise
except Exception as exception:
    logger.error('Failed to open file', exc_info=True)

logging.info("Creating mask of the shapefile")
mask_shp(SHP_PATH, TIFF_PATH, OUTPUT+OUT_FILENAME+"_mask.tif")
print("")

logging.info("Creating new raster with the new band values")
save_output_file(SHP_PATH, OUTPUT+OUT_FILENAME +
                    "_mask.tif", OUTPUT+OUT_FILENAME+"_sigpac.tif")
print("")
logging.info("The validation raster has been saved in", OUTPUT)

rows, cols, metadata, style, msk_band, sgc_band = read_needed_files(
    "/app/json/crop_style_sheet.json", OUTPUT+OUT_FILENAME+"_mask.tif", OUTPUT+OUT_FILENAME+"_sigpac.tif")

logging.info("Generating True/False raster: ")
raster_comparison(rows, cols, metadata, OUTPUT +
                  "true_false.tif", style, msk_band, sgc_band)

logging.info("Generating confusion matrix raster: ")
raster_comparison_confmatrix(
    rows, cols, metadata, OUTPUT+"_conf_matrix.tif", style, msk_band, sgc_band)

logging.info("Generating metrics and graphs: ")
create_dataframe_and_graphs(msk_band, sgc_band, OUTPUT +
                               OUT_FILENAME+"_metrics.csv")

if TEMPORARY_FILES == "yes":
    shutil.rmtree(TMP_PATH)
