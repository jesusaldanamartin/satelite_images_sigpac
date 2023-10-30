import sys
import os
import shutil
import logging
import rasterio
import fiona

from pathlib import Path

#from src.utils import *
#from src.validation import *

from utils import mask_shp, save_output_file, masked_all_shapefiles_in_directory, merge_tiff_images_in_directory, read_masked_files, reproject_raster
from validation import read_needed_files, raster_comparison, raster_comparison_confmatrix, create_dataframe_and_graphs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TIFF_PATH = sys.argv[1]
SHP_PATH = sys.argv[2]
OUT_FILENAME = sys.argv[3]
TEMPORARY_FILES = sys.argv[4]

TMP_PATH = "../satelite_images_sigpac/data/tmp/"
OUTPUT = "../satelite_images_sigpac/data/results/"

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

if os.path.isfile(SHP_PATH):

    logging.info("Creating mask of the shapefile")
    mask_shp(SHP_PATH, TIFF_PATH, OUTPUT+OUT_FILENAME+"_mask.tif")
    print("")

    logging.info("Creating new raster with the new band values")
    save_output_file(SHP_PATH, OUTPUT+OUT_FILENAME +
                     "_mask.tif", OUTPUT+OUT_FILENAME+"_sigpac.tif")
    print("")
    logging.info("The validation raster has been saved in", OUTPUT)

else:

    logging.info("Creating mask of all the shapefiles in the folder...")
    masked_all_shapefiles_in_directory(SHP_PATH, TMP_PATH+"masked/", TIFF_PATH)
    print("")

    logging.info("Merging all masked files...")
    merge_tiff_images_in_directory(
        TMP_PATH+"masked/", OUTPUT, OUT_FILENAME+"_mask.tif")
    print("")

    logging.info("Creating a new raster for each new mask with the new band values...")
    read_masked_files(TMP_PATH+"masked/", SHP_PATH, TMP_PATH+"sigpac/")
    print("")

    merge_tiff_images_in_directory(
        TMP_PATH+"sigpac/", OUTPUT, OUT_FILENAME+"_sigpac.tif")

    logging.info("The outcome rasters has been saved in:", OUTPUT)
    print("")

rows, cols, metadata, style, msk_band, sgc_band = read_needed_files(
    "../satelite_images_sigpac/json/crop_style_sheet.json", OUTPUT+OUT_FILENAME+"_mask.tif", OUTPUT+OUT_FILENAME+"_sigpac.tif")

logging.info("Generating True/False raster: ")
raster_comparison(rows, cols, metadata, OUTPUT +
                  "red_green.tif", style, msk_band, sgc_band)

logging.info("Generating confusion matrix raster: ")
raster_comparison_confmatrix(
    rows, cols, metadata, OUTPUT+"_conf_matrix.tif", style, msk_band, sgc_band)

logging.info("Generating metrics and graphs: ")
create_dataframe_and_graphs(msk_band, sgc_band, OUTPUT +
                               OUT_FILENAME+"_metrics.csv")

if TEMPORARY_FILES == "yes":
    shutil.rmtree(TMP_PATH)
