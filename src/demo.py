import sys
import os
import shutil
import logging
import rasterio
import fiona
import warnings

from pathlib import Path
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning

from utils import mask_shp, save_output_file, reproject_raster
from validation import read_needed_files, raster_comparison, raster_comparison_confmatrix, create_dataframe_and_graphs

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaWarning)

logging.basicConfig(level=logging.INFO)
logging.getLogger('numba.core.transforms').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

TIFF_PATH = sys.argv[1]
SHP_PATH = sys.argv[2]
OUT_FILENAME = sys.argv[3]
TEMPORARY_FILES = sys.argv[4]

#TMP_PATH = "/app/demo/data/tmp/"
TMP_PATH = "/app/demo/tif/mask_demo.tif"
OUTPUT = "/app/demo/data/results/"

logging.warning("TIME TO FINISH")

logging.info("Creating mask of the shapefile")

logging.info("Creating new raster with the new band values")
save_output_file(SHP_PATH, TMP_PATH, OUTPUT+OUT_FILENAME+"_sigpac.tif")
print("")
logging.info("The validation raster has been saved in", OUTPUT)

rows, cols, metadata, style, msk_band, sgc_band = read_needed_files(
    "/app/json/crop_style_sheet.json", TMP_PATH, OUTPUT+OUT_FILENAME+"_sigpac.tif")

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
