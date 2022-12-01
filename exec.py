from utils import *


masked_all_shapefiles_in_directory("./Shapefile_Data/CADIZ", "./masked_shp/CADIZ/","./results/spain29S.tif")

merge_tiff_images_in_directory("./masked_sigpac/CADIZ","./results/cadiz/cadizMask_sigpac.tif")

#* FINAL RASTER OUTPUT
read_masked_files("./masked_shp/CADIZ/", "./Shapefile_Data/CADIZ","./masked_sigpac/CADIZ")

#* FUNCTIONS USED TO VALIDATE THE RASTER
def results_validation():

    style_sheet, sigpac_band, classification_band = apply_style_sheet_to_raster("./json/olive_style_sheet.json",
        "./results/cadiz/cadizMask_sigpac.tif",
        "./results/cadiz/cadizMask.tif")

    # crops_hit_rate_to_json(style_sheet, sigpac_band, classification_band, "hit_rate.json")
    # x,y = validation("./results/malaga/raster_comparison_malaga.tif")
    # crop_metrics(sigpac_band, classification_band)

# results_validation()