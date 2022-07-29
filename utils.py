import rasterio
import rasterio.features
import rasterio.warp
import rasterio._err
from rasterio.merge import merge
from rasterio.plot import show
from pathlib import Path
import os
from os import listdir
from os.path import isfile, join
import tifftools
from tifftools import TifftoolsError
from matplotlib import pyplot

PATH = "C:\TFG_resources\satelite_img"

def get_list_of_tiles_in_spain():
    tiles = ['30SYG', '29TPG', '31SCC', '31TDE', '31SBD', '31SBC', '29SPC', '30STH', '30SYJ',
    '30SYH', '31SCD', '31SED', '31SDD', '29SQC', '29TPF', '30SVH', '30SVJ', '30SWJ',
    '30STG', '30SUH', '29SPD', '29TPH', '30TUM', '30SUJ', '30SUE', '30TVK', '31TCF',
    '29SQD', '31TEE', '29SQA', '29SPA', '30SWF', '30SUF', '30TTM', '29TQG', '29TQE',
    '29SQB', '30TTK', '29TNG', '29SPB', '29SQV', '30SXG', '30SXJ', '30SXH', '30SUG',
    '30STJ', '30TWL', '29TPE', '30STF', '30SVF', '30STE', '30TWK', '30TUK', '30SWG',
    '30SVG', '29TQF', '30SWH', '31TBE', '30SXF', '30TTL', '30TVL', '31TBF', '30TUL',
    '30TYK', '30TXK', '31TDF', '30TYL', '31TBG', '30TYM', '27RYM', '30TXL', '29TNH',
    '27RYL', '29TQH', '31TCG', '27RYN', '30TXM', '31TDG', '30TUN', '30TVM', '31TFE',
    '30TWM', '29TNG', '29THN', '29TNJ', '29TPJ', '29TQJ', '30TPU', '30TVP', '30TWP',
    '30TVN', '30TWN', '30TXN', '30TYN', '31TCH' ]

    return tiles

tiles = get_list_of_tiles_in_spain()
folder_files = os.listdir(PATH)

def get_tiles_merge(name_list):
    file_names = []
    file_paths = []
    for file in name_list:
        if "_" in file:
            sep = file.split("_",1)[1]
        if "." in file:
            cod = sep.split(".",1)[0]
            if cod in tiles:
                file_names.append(file)
                file_paths.append(PATH+f"\{file}")
    return file_names, file_paths


def merge_tiles_tifftools(output_name):
    folder_files = os.listdir(PATH)
    if output_name not in folder_files:
        file_names, file_paths = get_tiles_merge(folder_files)
        tif = tifftools.read_tiff(file_paths[0])
        for file in file_names[1:]:
            next_file = tifftools.read_tiff(PATH+f"\{file}")
            tif['ifds'].extend(next_file['ifds'])
        try:
            tifftools.write_tiff(tif, PATH+f"\{output_name}.tif")
        except TifftoolsError:
            # print("Tifftools merge file already exists")
            pass


def merge_tiles_rasterio(output_name):
    src_files_to_mosaic = []
    folder_files = os.listdir(PATH)
    output_file = PATH+f"\{output_name}.tif"
    if output_name not in folder_files:
        _, file_paths = get_tiles_merge(folder_files)
        for i in file_paths:
            src = rasterio.open(i)
            src_files_to_mosaic.append(src)

        mosaic, out_trans = merge(src_files_to_mosaic)

        out_meta = src.meta.copy()

        out_meta.update({"driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
            }
            )
        try:
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(mosaic)
        except rasterio._err.CPLE_BaseError:
            print("Rasterio merge file already exists")
    return mosaic

south_spain_rasterio = merge_tiles_rasterio("spain_merged")
merge_tiles_tifftools("spain_merged_layers")

show(south_spain_rasterio, cmap='gist_earth', title='satelite view') # More cmap parameters: viridis, Greens, Blues, Reds, jet

# _, file_paths = get_tiles_merge(folder_files)


# Update the metadata



# show(mosaic, cmap='terrain')

# dataset = rasterio.open("C:\TFG_resources\satelite_img\classification_30SUF.tif")
# dataset = rasterio.open("C:\TFG_resources\satelite_img\merged_layers.tiff")

# print(dataset.indexes)
# print(dataset.count)

# print(dataset.width)
# print(dataset.height)

# print(dataset.bounds)# All 4 corners of the image

# # Affine transformation matrix that maps pixel location in (row,col) coordinates to (x,y) spatial positions.
# print(dataset.transform)
# print(dataset.transform*(0,0)) # Top left corner
# print(dataset.transform*(dataset.width,dataset.height)) # Top left corner

# print(dataset.indexes) #(1,) only 1 band in TIF image
# band1 = dataset.read(1)

# print(band1)
# print(band1[123,123])

# x,y = (dataset.bounds.left + 100000, dataset.bounds.top - 50000)
# print(x,y)
# row, col = dataset.index(x,y)
# print(row,col)
# print(band1[row,col])

# # CENTER OF IMAGE
# print(dataset.xy(dataset.height, dataset.width))