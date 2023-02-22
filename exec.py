from utils import *
import sys
import os

TIFF_PATH = sys.argv[1]
SHP_PATH = sys.argv[2]
OUT_FILENAME = sys.argv[3]
TEMPORARY_FILES = sys.argv[4]

TMP_PATH = "./satelite_images_sigpac/data/tmp/"
OUTPUT = "./satelite_images_sigpac/data/results/"

with fiona.open(TIFF_PATH) as raster:
    tif_crs = raster.crs

with fiona.open(SHP_PATH) as shapefile:
    shp_crs = shapefile.crs

if tif_crs != shp_crs:
    print("Files have different CRS")
    reproject_raster(TIFF_PATH, TMP_PATH, "reference_reprojected.tif", shp_crs)


print("Creating mask of the shapefile")
mask_shp(SHP_PATH, TIFF_PATH, TMP_PATH+OUT_FILENAME+"masked.tif")
print("")

print("Creating new raster with the new band values")
save_output_file(SHP_PATH, TMP_PATH+OUT_FILENAME+"masked.tif", OUTPUT+OUT_FILENAME+"sigpac.tif")
print("")
print(f"The validation raster has been saved in: {OUTPUT}")


#TODO Finish script with the validation rasters, csv and graphs.
#TODO Update script now check if arguments are file or folders. If folder call folder functions.
#TODO Update launch.sh It must get arguments and options via terminal. With help, version, usage.
#TODO Fix the way the files are obtained, better functions instead of split. ¿Contains?
#TODO Add new machine learning feature and link it to the actual workflow.
#TODO Refactor all the code and the github.
#TODO Finish the ipynb and upload custom example.

# reproject_raster("./satelite_images_sigpac/data/sat_images/spain30T.tif", 
#     "./satelite_images_sigpac/data/sat_images/", "spain30T_4258.tif")

# merge_tiff_images_in_directory("./satelite_images_sigpac/data/sat_images/spain_30T", 
#     "./satelite_images_sigpac/data/sat_images/", "spain30T.tif")


#* MASK SINGLE FILE
# mask_shp("./satelite_images_sigpac/data/CastillaLeon/09_Burgos/09001_RECFE.shp",
#          "./satelite_images_sigpac/data/sat_images/spain30T_4258.tif", 
#         "burgos_09001_croppped.tif")

#* MASK FOLDER OF SHP
# masked_all_shapefiles_in_directory("./satelite_images_sigpac/data/CastillaLeon/09_Burgos/",
#      "./satelite_images_sigpac/data/CastillaLeon/masked_Burgos/",
#      "./satelite_images_sigpac/data/sat_images/spain30T_4258.tif")

#* MERGE ALL TIFFS FROM FOLDER
# merge_tiff_images_in_directory("./masked_shp/HUELVA/","./results/huelva/huelvaMasked.tif")

#* SINGLE RASTER OUTPUT
# save_output_file("./satelite_images_sigpac/data/CastillaLeon/09_Burgos/09001_RECFE.shp",
#                 "burgos_09001_croppped.tif",
#                 "nucleos_process3_burgos_09001_sigpac.tif")

#* FOLDER RASTER OUTPUT
read_masked_files("./satelite_images_sigpac/data/CastillaLeon/masked_Burgos/",
    "./satelite_images_sigpac/data/CastillaLeon/09_Burgos",
    "./satelite_images_sigpac/data/CastillaLeon/sigpac_Burgos/")

#* FUNCTIONS USED TO VALIDATE THE RASTER
# def results_validation():

#     # style_sheet, sigpac_band, classification_band = apply_style_sheet_to_raster("./json/olive_style_sheet.json",
#     #     "./results/huelva/huelvaMask_sigpac.tif",
#     #     "./results/huelva/huelvaMasked.tif")

#     # crops_hit_rate_to_json(style_sheet, sigpac_band, classification_band, "hit_rate.json")
#     # x,y = validation("./results/malaga/raster_comparison_malaga.tif")
#     # crop_metrics(sigpac_band, classification_band, 'csv/huelva.csv')

#     results_validation()

#* ANDALUCIA CSV
# process_dataframe("./csv/andalucia_tp_tn.csv")
