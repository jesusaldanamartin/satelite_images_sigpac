import sys
import os
import shutil

from utils import *
from validation import *

TIFF_PATH = sys.argv[1]
SHP_PATH = sys.argv[2]
OUT_FILENAME = sys.argv[3]
TEMPORARY_FILES = sys.argv[4]

# * /home/jesus/Documents/TFG/satelite_images_sigpac/data/sat_images/spain30T_4258.tif
# * /home/jesus/Documents/TFG/satelite_images_sigpac/data/CastillaLeon/09_Burgos/09001_RECFE.shp
# * prueba_de_script
# * yes

TMP_PATH = "../satelite_images_sigpac/data/tmp/"
OUTPUT = "../satelite_images_sigpac/data/results/"


# DONE #TODO Update script now check if arguments are file or folders. If folder call folder functions.
# DONE #TODO Update launch.sh It must get arguments and options via terminal. With help, version, usage.
# PARTIAL DONE # TODO Fix the way the files are obtained, better functions instead of split. ¿Contains?
# PARTIAL DONE  # TODO Finish script with the comparison rasters, csv.

# TODO Graphs
# TODO ONCE bash script is finished , continue with ml
# No hace falta, ese modelo va a parte # TODO Add the new machine learning workflow to the bash script.
# TODO Finish the showcase.ipynb and upload custom example.
# TODO Refactor all the code.
# TODO Start the TFG memory essay

with rasterio.open(TIFF_PATH) as raster:
    tif_crs = raster.crs

with fiona.open(SHP_PATH) as shapefile:
    shp_crs = shapefile.crs

if tif_crs != shp_crs:
    print("Files have different CRS")
    reproject_raster(TIFF_PATH, TMP_PATH, TMP_PATH +
                     "reference_reprojected.tif", shp_crs)
    TIFF_PATH = TMP_PATH+"reference_reprojected.tif"

if os.path.isfile(SHP_PATH):

    print("Creating mask of the shapefile")
    mask_shp(SHP_PATH, TIFF_PATH, OUTPUT+OUT_FILENAME+"_mask.tif")
    print("")

    print("Creating new raster with the new band values")
    save_output_file(SHP_PATH, OUTPUT+OUT_FILENAME +
                     "_mask.tif", OUTPUT+OUT_FILENAME+"_sigpac.tif")
    print("")
    print(f"The validation raster has been saved in: {OUTPUT}")

else:

    print("Creating mask of all the shapefiles in the folder...")
    masked_all_shapefiles_in_directory(SHP_PATH, TMP_PATH+"masked/", TIFF_PATH)
    print("")

    print("Merging all masked files...")
    merge_tiff_images_in_directory(
        TMP_PATH+"masked/", OUTPUT, OUT_FILENAME+"_mask.tif")
    print("")

    print("Creating a new raster for each new mask with the new band values...")
    read_masked_files(TMP_PATH+"masked/", SHP_PATH, TMP_PATH+"sigpac/")
    print("")

    merge_tiff_images_in_directory(
        TMP_PATH+"sigpac/", OUTPUT, OUT_FILENAME+"_sigpac.tif")

    print(f"The outcome rasters has been saved in: {OUTPUT}")
    print("")

rows, cols, metadata, style, msk_band, sgc_band = read_needed_files(
    "../satelite_images_sigpac/json/crop_style_sheet.json", OUTPUT+OUT_FILENAME+"_mask.tif", OUTPUT+OUT_FILENAME+"_sigpac.tif")

print("Generating True/False raster: ")
raster_comparison(rows, cols, metadata, OUTPUT +
                  "red_green.tif", style, msk_band, sgc_band)

print("Generating confusion matrix raster: ")
raster_comparison_confmatrix(
    rows, cols, metadata, OUTPUT+"_conf_matrix.tif", style, msk_band, sgc_band)

print("Generating metrics and graphs: ")
create_dataframe_metrics_crops(msk_band, sgc_band, OUTPUT +
                         OUT_FILENAME+"_metrics.csv")

if TEMPORARY_FILES == "yes":
    shutil.rmtree(TMP_PATH)
