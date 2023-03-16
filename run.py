import sys
import os
import shutil
import logging
from pathlib import Path

from utils import *
from validation import *

# Use a logging library instead of print statements:
# It is better to use a logging library such as logging instead of print statements. It provides a more flexible and robust way to log and handle errors and warnings.

# Use argparse instead of sys.argv:
# It is better to use argparse to parse command-line arguments instead of directly using sys.argv. Argparse provides more functionality such as help messages, default values, type checking, etc.

# Use pathlib instead of os.path:
# It is better to use the pathlib module instead of os.path for working with file paths. Pathlib provides an object-oriented interface for file system paths, which is easier to read and maintain.

# Use context managers for file handling:
# When working with files, it is better to use context managers such as with statements to ensure that the files are properly closed when they are no longer needed.

# Refactor repeated code into functions:
# You can refactor the code into smaller functions to make it more modular and reusable.

# Use docstrings to document functions:
# You can use docstrings to document each function and explain what it does and what arguments it takes.

# Use constants instead of hardcoding values:
# You can define constants for values that are used repeatedly in the code instead of hardcoding them.


# TODO Terminal de procesar CyL.
# TODO Script de ML
# TODO Crear gráficas para mostrar los datos. (para poner en la memoria)
# TODO OPCIONAL: Mejorar la forma de buscar los shp restantes para el sgc. ¿Contains?
# TODO OPCIONAL: showcase.ipynb y ejemplo de uso simple (rápido) en repositorio.
# TODO OPCIONAL: README.md
# TODO Memoria del TFG.

# * (tfg_venv) jesus@jesus-XPS-15-9560:~/Documents/TFG/satelite_images_sigpac$ bash launch.sh -r /home/jesus/Documents/TFG/satelite_images_sigpac/data/sat_images/spain/spain30T.tif -s /home/jesus/Documents/TFG/satelite_images_sigpac/data/SIGPAC_CyL_Municipios/47_Valladolid -o  valladolid -t no

# * /home/jesus/Documents/TFG/satelite_images_sigpac/data/sat_images/spain/spain30T.tif
# * /home/jesus/Documents/TFG/satelite_images_sigpac/data/ftp.itacyl.es/cartografia/05_SIGPAC/2021_ETRS89/Parcelario_SIGPAC_CyL_Municipios/47_Valladolid
# * valladolid
# * no

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
create_dataframe_metrics_crops(msk_band, sgc_band, OUTPUT +
                               OUT_FILENAME+"_metrics.csv")

# graphs() #TODO

if TEMPORARY_FILES == "yes":
    shutil.rmtree(TMP_PATH)
