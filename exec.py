from utils import *

reproject_raster("./results/spain30T.tif", "./results/spain30T_4258.tif")

#* MASK SINGLE FILE
# mask_shp("./Shapefile_Data/AVILA/05_RECFE.shp",
#          "./results/spain30T_4258.tif", 
#         "avilaMasked.tif")

#* MASK FOLDER OF SHP
# masked_all_shapefiles_in_directory("./Shapefile_Data/HUELVA", "./masked_shp/HUELVA/","./results/spain29S_latlon.tif")

#* MERFE ALL TIFFS FROM FOLDER
# merge_tiff_images_in_directory("./masked_shp/HUELVA/","./results/huelva/huelvaMasked.tif")

#* SINGLE RASTER OUTPUT
# save_output_file("./masked_shp/AVILA/avilaMasked.tif",
#                 "./Shapefile_Data/AVILA/05_RECFE.shp",
#                 "resul_avila.tif")

#* FOLDER RASTER OUTPUT
# read_masked_files("./masked_shp/HUELVA/", "./Shapefile_Data/HUELVA","./masked_sigpac/HUELVA/")

#* FUNCTIONS USED TO VALIDATE THE RASTER
def results_validation():

    # style_sheet, sigpac_band, classification_band = apply_style_sheet_to_raster("./json/olive_style_sheet.json",
    #     "./results/huelva/huelvaMask_sigpac.tif",
    #     "./results/huelva/huelvaMasked.tif")

    # crops_hit_rate_to_json(style_sheet, sigpac_band, classification_band, "hit_rate.json")
    # x,y = validation("./results/malaga/raster_comparison_malaga.tif")
    # crop_metrics(sigpac_band, classification_band, 'csv/huelva.csv')

    results_validation()

#* ANDALUCIA CSV
# process_dataframe("./csv/andalucia_tp_tn.csv")
