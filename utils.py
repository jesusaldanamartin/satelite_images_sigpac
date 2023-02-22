from rasterio import (
    rasterio,
    merge,
    mask
)

import multiprocessing
from multiprocessing import Manager, Array, pool
import ctypes as c

from rasterio.warp import Resampling, reproject, calculate_default_transform

import rasterio._err
import pandas as pd

from tqdm import tqdm
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import fiona
import numpy as np
import threading

from typing import Any, List, Tuple, BinaryIO
import os
import json
from os import listdir
from os.path import isfile, join
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

COD_USO = ['AG' ,'CA', 'CF' ,'CI', 'CS', 'CV', 'ED', 'EP', 'FF', 'FL', 'FO', 
    'FS', 'FV', 'FY', 'IM', 'IV', 'OC', 'OF', 'OV', 'PA', 'PR', 'PS',
    'TA', 'TH', 'VF', 'VI', 'VO', 'ZC', 'ZU', 'ZV' ]

TILES = ['30SYG', '29TPG', '31SCC', '31TDE', '31SBD', '31SBC', '29SPC', '30STH', '30SYJ',
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

def get_id_codigo_uso(key: str):
    '''Raster bands cannot have string value so each cod_uso it is been replaced with an id.

    Args:
        key (str): Key stored in shp metadata.

    Returns:
        None
    '''
    if key == 'AG' : return 1      #* Corrientes y superficies de agua
    if key == 'CA' : return 2      #* Viales
    if key == 'CF' : return 3      #* Citricos-Frutal
    if key == 'CI' : return 4      #* Citricos
    if key == 'CS' : return 5      #* Citricos-Frutal de cascara
    if key == 'CV' : return 6      #* Citricos-Viñedo
    if key == 'ED' : return 7      #* Edificaciones
    if key == 'EP' : return 8      #* Elemento del Paisaje
    if key == 'FF' : return 9      #* Frutal de Cascara-Frutal
    if key == 'FL' : return 10      #* Frutal de Cascara-Olivar
    if key == 'FO' : return 11      #* Forestal
    if key == 'FS' : return 12      #* Frutal de Cascara
    if key == 'FV' : return 13      #* Frutal de Cascara-Viñedo
    if key == 'FY' : return 14      #* Frutal
    if key == 'IM' : return 15      #* Improductivo
    if key == 'IV' : return 16      #* Imvernadero y cultivos bajo plastico
    if key == 'OC' : return 17      #* Olivar-Citricos
    if key == 'OF' : return 18      #* Olivar-Frutal
    if key == 'OV' : return 19      #* Olivar
    if key == 'PA' : return 20      #* Pasto Arbolado
    if key == 'PR' : return 21      #* Pasto Arbustivo
    if key == 'PS' : return 22      #* Pastizal
    if key == 'TA' : return 23      #* Tierra Arable
    if key == 'TH' : return 24      #* Huerta
    if key == 'VF' : return 25      #* Frutal-Viñedo
    if key == 'VI' : return 26      #* Viñedo
    if key == 'VO' : return 27      #* Olivar-Viñedo
    if key == 'ZC' : return 28      #* Zona Concentrada
    if key == 'ZU' : return 29      #* Zona Urbana
    if key == 'ZV' : return 30      #* Zona Censurada


def reproject_raster(in_path: str, out_path: str, file_name: str, new_crs):
    '''Given an in and out path this function reproject a raster into any coordinate reference system set

    Args:
        in_path (str): Path to the raster we want to reproject.
        out_path (str):  Output path where the raster will be saved.

    Return:
        None
    '''
    #  - 
    crs = "EPSG:4258 ETRS89"
    crs = new_crs
    # reproject raster to project crs
    with rasterio.open(in_path) as src:
        src_crs = src.crs
        transform, width, height = calculate_default_transform(src_crs, crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()

        kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height})

        with rasterio.open(out_path + file_name, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=Resampling.nearest)

# reproject_raster("C:\TFG_resources\satelite_img\W020N60_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif", 
#     "/home/jesus/Documents/satelite_images_sigpac/results/UK_CORINE/british_grid.tif")

def mask_shp(shp_path: str, tif_path: str, output_name: str):
    '''Crop a tif image with the shapefile geoemetries.

    Args:
        shp_path (str): Path to the shapefile file.
        tif_path (str): Path to the tif image.
        output_name (str): File name to be saved.

    Returns:
        Save in working directory the image croppped.
    '''

    with fiona.open(shp_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(tif_path, "r") as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
        src_crs = src.crs

    out_meta.update({
                 "driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform,
                 "crs": src_crs
                 #"crs": "+proj=utm +zone=30 +ellps=WGS84 +units=m +no_defs"   
            })
            
    with rasterio.open(output_name, "w", **out_meta) as dest:
        dest.write(out_image)

# mask_shp("./Shapefile_Data/ALMERIA/",
#          "./results/spain30S.tif", 
#         "almeriaMasked.tif")

def masked_all_shapefiles_in_directory(folder_path: str, output_path: str, mask: str):
    '''Read all shapefiles stored in directory and create a mask for each file.

    Args:
        folder_path (str): Path to the directory where all shapefiles are stored.
        output_path (str): Path to the output directory.
        mask (str) : Path to the .tif image that is going to be used to mask the shapefiles.
    
    Return:
        A file is created for each shapefile in folder.
    '''

    folder_files = os.listdir(folder_path)
    for file in tqdm(folder_files):
        extension = file.split('.')[1]
        file_number = file.split('.')[0]
        if extension == 'shp':
            try:
                mask_shp(folder_path+f"/{file}", mask, output_path+f"{file_number[:5]}_masked.tif")
            except ValueError:
                print(file+" does not overlap figure")


def merge_tiff_images_in_directory(folder_path: str, output_path: str, file_name: str):
    '''Merge all tiff images stored in folder_path.

    All metadata is saved and stored so the output will be be only a merge of all images given as input.

    Args:

        output_name (str): Name as the output file will be stored.
        folder_path (str): Path to the folder where tiff images are.

    Returns: 
        The merged image will be stored in working directory.
    '''

    src_files_to_mosaic = []
    folder_files = os.listdir(folder_path)
    out_meta = {"null:null"}

    for file in folder_files:
        src = rasterio.open(folder_path+f"/{file}")
        src_files_to_mosaic.append(src)
        src_crs = src.crs #* For a new custom crs output

        out_meta = src.meta.copy()

    mosaic, out_trans = merge.merge(src_files_to_mosaic)

    out_meta.update({"driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": "+proj=utm +zone=30 +ellps=WGS84 +units=m +no_defs"
        }
        )
    try:
        with rasterio.open(output_path + file_name, "w", **out_meta) as dest:
            dest.write(mosaic)
    except rasterio._err.CPLE_BaseError:
        print("Merge file already exists")
        pass
    return mosaic


@jit
def is_point_in_polygon(x: int, y: int, polygon: list) -> bool:
    '''Determine if the pixel is in the geometry.

    Args: 
        x (int): The x coordinate of the point
        y (int): The y coordinate of the point
        polygon (List[Any]): List of tuples (x,y)

    Returns:
        Boolean: True if point is inside the geometry, is a corner or is on the boundary
    '''

    size = len(polygon)
    in_geometry = False
    j = size - 1

    for i in range(size):
        if (x == polygon[i][0]) and (y == polygon[i][1]):
            return True #* point is a corner
        if ((polygon[i][1] > y) != (polygon[j][1] > y)):
            slope = (x-polygon[i][0])*(polygon[j][1]-polygon[i][1])-(polygon[j][0]-polygon[i][0])*(y-polygon[i][1])
            if slope == 0:
                return True #* point is on boundary
            if (slope < 0) != (polygon[j][1] < polygon[i][1]):
                in_geometry = not in_geometry #* if an edge is crossed an odd number of times the point is in geometry
        j = i
    return in_geometry

@jit
def index_values(points_list: list, polygon: list) -> List:
    '''Iterate trought the list with all pixels.

    For each pixel is_point_in_polygon() function is called.

    Args: 
        points_list (List[(x,y)]): List of all pixels coordinates of the raster.
        polygon (List[(x,y)]): List of tuples with the coordinates (x,y) of the polygon.

    Returns:
        index_points (List[Any]): Index of the points in points_list that are inside the geometry.
    '''

    index_points=[]
    size = len(points_list)
    for i in range(size): 
        bool = is_point_in_polygon(points_list[i][0], points_list[i][1], polygon)
        if bool: 
            index_points.append(i)
    return index_points

def replace_band_matrix(path: str, points_list: list, arr: np.ndarray, transformer: rasterio.Affine, d):
    '''Replace in the raster band the use code from SIGPAC.

    Args:
        path (str): Path to the shapefile.
        points_list (List[(x,y)]): List of all pixels coordinates of the raster.
        arr (np.ndarray): Numpy array with the raster's band information.
        transformer (rasterio.Affine): 'rasterio.transform.AffineTransformer' to convert from (x,y) to band value.

    Returns:
        arr (np.ndarray): Numpy array with band values changed to SIGPAC use code.
    '''
    result = []
    indexes = []
    codes = []
    ind_values = []
    cd_uso = 0
    with fiona.open(path) as layer:
        for feature in tqdm(layer):
            ord_dict = feature['properties']

            for key in ord_dict.values():
                if key in COD_USO: 
                    cd_uso = get_id_codigo_uso(key) #* save the SIGPAC use code for that exact geometry
            geometry = feature["geometry"]['coordinates']

            for g in geometry:
                ind_values = index_values(points_list, g)

                if len(ind_values) != 0:

                    for ind in ind_values:
                        ind_arr = transformer.rowcol(points_list[ind][0],points_list[ind][1])
                       # indexes.append(ind_arr)
                       # codes.append(cd_uso)
                        result.append((ind_arr, cd_uso))
                        #arr[ind_arr[0],ind_arr[1]] = cd_uso #* replace the new use code from SIGPAC in the old band value.
    # d["ind"] = indexes 
    # d["codes"] = codes
    d[multiprocessing.current_process().pid] = result


def replace_values(d: dict, arr: np.ndarray) -> np.ndarray: 
    '''Swap the band values of the array with the values stored in the dictionary.

    Args:
        d (Dict): Dictionary with the band data stored.
        arr (np.ndarray): Array with the data read from the raster.

    Returns:
        The array with the new values obtained.
    '''

    for item in tqdm(d.keys()):
        for value in d[item]:
            arr[value[0][0], value[0][1]] = value[1]

    return arr             


def multithreading(points_list: list, shp_path: str, arr: np.ndarray, transformer: rasterio.Affine) -> np.ndarray:
    '''Execute the process with multiple threads improving performance.

    Args:
        points_list (List[(x,y)]): List of all pixels coordinates of the raster.
        shp_path (str): Path to the shapefile.
        arr (np.ndarray): Numpy array with the raster's band information.
        transformer (rasterio.Affine): 'rasterio.transform.AffineTransformer' to convert from (x,y) to band value.
    
    Returns:
        None
    '''

    cpu_num = multiprocessing.cpu_count()
    chunked_list = np.array_split(points_list, cpu_num)
    size = len(chunked_list)
    p_list = []
    p_result = []

    # manager = multiprocessing.Manager()
    # return_dict = manager.Array('i', range(1000))
       
    # for i in range(size):
    #     with pool.Pool(processes=cpu_num) as mp_pool:
    #         result = mp_pool.apply(replace_band_matrix, (shp_path, chunked_list[i], arr, transformer))
    #         print(result)
    # queue = multiprocessing.Queue()
    manager = Manager()
    d = manager.dict()  

    for i in range(size):
        # p = threading.Thread(target=replace_band_matrix, args=(shp_path, chunked_list[i], arr, transformer))
        p = multiprocessing.Process(target=replace_band_matrix, args=(shp_path, chunked_list[i], arr, transformer, d))
        p_list.append(p)
        p.start()

    for p in p_list:
        p.join() #* wait until all threads have finished

    new_arr = replace_values(d, arr)

    return new_arr


def save_output_file(shp_path: str, tif_path: str, output_path: str):
    '''Final raster file created.

    Read the given .tif, some metadata is saved and function replace_band_matrix() is called.

    Args:
        tif_path (str): Path to the tif image.
        shp_path (str): Path to the shapefile.
        output_path (str): Path to be saved.

    Returns:
        Save in working directory the final image.
    '''

    with rasterio.open(tif_path) as src:
        profile = src.profile #* raster metadata
        arr = src.read(1)
        transformer = rasterio.transform.AffineTransformer(profile['transform'])
        not_zero_indices = np.nonzero(arr) #* get all indexes of non-zero values in array to reduce its size

        points_list = [transformer.xy(not_zero_indices[0][i],not_zero_indices[1][i]) 
                        for i in tqdm(range(len(not_zero_indices[0])))] #* coordinates list of src values #TQDM

        #? matrix = replace_band_matrix(shp_path, points_list, arr, transformer)
        new_arr = multithreading(points_list, shp_path, arr, transformer)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(new_arr, 1)

# save_output_file("./masked_shp/AVILA/avilaMasked.tif",
#                 "./Shapefile_Data/AVILA/05_RECFE.shp",
#                 "resul_avila.tif")

def read_masked_files(folder_path: str, shp_data_folder: str, sigpac_data_folder: str):
    '''For every masked file in folder save_output_file() function is called 
    to convert input masked files into masked sigpac tif.

    Args:
        folder_path (str): Path to the folder with all masked files.
        shp_data_folder (str): Path to the directory where all shapefiles are stored.
        sigpac_data_folder (str): Path to the sigpac tif file.

    Returns:
        One file for each shapefile.
    '''

    folder_files = os.listdir(folder_path)

    for file in folder_files:
        file_number = file.split('.')[0]

        #? os.path.getsize(folder_path+f"{file}") < 8000000 and
        if file_number[0:5]+f"_sigpac.tif" not in os.listdir(sigpac_data_folder):
            print(file)
            
            save_output_file(shp_data_folder+f"/{file[:5]}_RECFE.shp",
                folder_path+f"/{file}",
                sigpac_data_folder+f"{file[:5]}_sigpac.tif")            
            print(file+" finished")
            print("")


#read_masked_files("/home/jesus/Documents/satelite_images_sigpac/masked_shp/CADIZ/")


#!---------------------------------------------------------------------------------------------------------------------------------------

def raster_comparison(rows: int,cols: int, new_raster_output, 
    style_sheet, sigpac_band, classification_band) -> np.ndarray:
    '''This function compares the band values of two different raster. These values 
    are linked with the crop_style_sheet.json file. Both rasters must have the same size.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        new_raster_output (np.ndarray): 2D numpy array copy of our input raster.
        style_sheet (dict): Path to the json file.
        sigpac_band (ndarray): Sigpac raster band read with rasterio.
        classification_band (ndarray): Lab raster band read with rasterio.

    Returns:
        This function returns a ndarray where band values have been replaced with 
        the new compared values.
    '''

    try:
        for x in tqdm(range(rows)):
            for y in range(cols):
                if sigpac_band[x,y] != 0 and classification_band[x,y] !=0:
                    if len(style_sheet[str(sigpac_band[x,y])]) > 1:
                        for item in style_sheet[str(sigpac_band[x,y])]:
                            # print(item)
                            # print(style_sheet[str(sigpac_band[x,y])])
                            # print(classification_band[x,y])
                            if classification_band[x,y] == item:
                                # print("OK",":",item)
                                new_raster_output[x,y] = 20 #* same band value
                            else:
                                # print("WRONG",":",classification_band[x,y]," distinto ",style_sheet[str(sigpac_band[x,y])])
                                new_raster_output[x,y] = 21 #* diff band value

                    else:
                        if style_sheet[str(sigpac_band[x,y])] == classification_band[x,y]:
                            # print("OK 2",":", classification_band[x,y])
                            new_raster_output[x,y] = 20 #* same band value

                        else:
                            # print("WRONG 2",":",classification_band[x,y]," distinto ",style_sheet[str(sigpac_band[x,y])])
                            new_raster_output[x,y] = 21 #* diff band value
    except IndexError:
        pass
    return new_raster_output

def raster_comparison_cropland(rows: int,cols: int, new_raster_output, 
    style_sheet, sigpac_band, classification_band):
    '''This function compares the crop zones in both land covers given by parameters.
    These values are linked with the id_style_sheet.json file. Both rasters must have the same size.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        new_raster_output (np.ndarray): 2D numpy array copy of our input raster.
        style_sheet (dict): Path to the json file.
        sigpac_band (ndarray): Sigpac raster band read with rasterio.
        classification_band (ndarray): Lab raster band read with rasterio.

    Returns:
        This function returns a ndarray matrix where band values have been replaced with 
        the new compared values.
    '''

    try:
        for x in tqdm(range(rows)):
            for y in range(cols):
                if sigpac_band[x,y] != 0 and classification_band[x,y] !=0:
                    # print(style_sheet[str(sigpac_band[x,y])])
                    # print(classification_band[x,y])
                    if style_sheet[str(sigpac_band[x,y])] == 6 and classification_band[x,y] == 6:
                        if style_sheet[str(sigpac_band[x,y])] == classification_band[x,y]:
                            # print("VERDE")
                            new_raster_output[x,y] = 20 #*True Positives

                    elif classification_band[x,y] == 6 and style_sheet[str(sigpac_band[x,y])] != 6:
                        # print("ROJO")
                        new_raster_output[x,y] = 21 #* False Positive

                    elif classification_band[x,y] != 6 and style_sheet[str(sigpac_band[x,y])] == 6:
                        # print("AZUL")
                        new_raster_output[x,y] = 22 #* False Negatives
                    else:
                        # print("NEGRO")
                        new_raster_output[x,y] = 23 #* True Negatives
    except IndexError:
        pass
    return new_raster_output

def specific_class_raster_comparison(rows: int,cols: int, new_raster_output, 
    style_sheet, sigpac_band, classification_band):
    '''This function compares the crop zones in both land covers given by parameters.
    These values are linked with the id_style_sheet.json file. Both rasters must have the same size.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        new_raster_output (np.ndarray): 2D numpy array copy of our input raster.
        style_sheet (dict): Path to the json file.
        sigpac_band (ndarray): Sigpac raster band read with rasterio.
        classification_band (ndarray): Lab raster band read with rasterio.

    Returns:
        This function returns a ndarray matrix where band values have been replaced with 
        the new compared values.
    '''

    #! TP Acierto seguro (es olivo en sigpac y crop en classification)
    #! FP Cropland no olivo (es cropland pero no mi clase concreta)
    #! TN No cropland no olivo (no es cropland en classification ni olivo en SIGPAC)
    #! FN Fallo seguro (No es ni cropland en sigpac ni forest en classification)
    olive = [10,17,18,19,27]
    try:
        for x in tqdm(range(rows)):
            for y in range(cols):
                if sigpac_band[x,y] != 0 and classification_band[x,y] !=0:
                    # print(style_sheet[str(sigpac_band[x,y])])
                    # # print(classification_band[x,y])
                    # print(sigpac_band[x,y])
                    # print(style_sheet[str(sigpac_band[x,y])])
                    if sigpac_band[x,y] == 19 and classification_band[x,y] == 6:
                        # print("TP = VERDE")
                        new_raster_output[x,y] = 20 #*True Positives

                    elif classification_band[x,y] == 6 and (6 in style_sheet[str(sigpac_band[x,y])] and sigpac_band[x,y] != 19):
                        # print("FP = ROJO")
                        new_raster_output[x,y] = 21 #* False Positive

                    elif classification_band[x,y] != 6 and (6 in style_sheet[str(sigpac_band[x,y])] and sigpac_band[x,y] == 19) :
                        # print("FN = AZUL")
                        new_raster_output[x,y] = 22 #* False Negatives
                    else:
                        # print("TN = NEGRO")
                        new_raster_output[x,y] = 23 #* True Negatives
    except IndexError:
        pass
    return new_raster_output

def apply_style_sheet_to_raster(json_path: str, sigpac_path: str, masked_path: str):
    '''Read all files needed and call function raster_comparison_cropland() to compare both rasters.

    Args:
        json_path (str): Path to the json file.
        sigpac_path (str): Path to the sigpac processed raster.
        masked_path (str): Path to the classification raster (LAB).

    Returns:
        style_sheet (dict): Data obtained from json_path. 
        sigpac_band (np.ndarray): Raster data read with rasterio.
        classification_band (np.ndarray): Raster data read with rasterio.

    '''
    with open(json_path) as jfile:
        dict_json = json.load(jfile)
        style_sheet = dict_json['style_sheet']['SIGPAC_code']
    
    with rasterio.open(sigpac_path) as src:
        sigpac_band = src.read(1) 
        rows = sigpac_band.shape[0] #* 10654
        cols = sigpac_band.shape[1] #* 16555

    with rasterio.open(masked_path) as src:
        classification_band = src.read(1) 
        arr = src.read(1)
        profile = src.profile #* raster metadata
        rows = classification_band.shape[0] #* 10654
        cols = classification_band.shape[1] #* 16555

    # new_raster_output = raster_comparison_cropland(rows, cols, arr, style_sheet, sigpac_band, classification_band)

    # with rasterio.open("FPandFN_raster_comparison_olive_jaen.tif", 'w', **profile) as dst:
    #     dst.write(new_raster_output, 1)

    return style_sheet, sigpac_band, classification_band

# style_sheet, sigpac_band, classification_band = apply_style_sheet_to_raster("json/olive_style_sheet.json",
#     "./results/cadiz/cadizMask_sigpac.tif",
#     "./results/cadiz/cadizMask.tif")

#*  TP  Banda numero 20(verde) coinciden SGP y SAT
#*  TN  Banda numero 23(negro)  SAT cropland SGP no
#*  FP  Banda numero 21(rojo)  No cropland n
#*  FN  Banda numero 22(azul)  SGP cropland SAT ni en SAT ni SG

def crop_metrics(sigpac_band, classification_band, output_path):
    '''Create a Data Frame with all the metrics information obtained.

    Args:
        sigpac_band (np.ndarray): Raster information from sigpac data.
        classification_band (np.ndarray): Raster information Random forest classification from sentinel-2
        output_path (str): Path to the directory where csv will be stored.

    Return:
        None
    '''

    data_output = pd.DataFrame(columns=["Citricos Frutal","Citricos","Citricos-Frutal de cascara","Citricos-Viñedo", "Frutal de Cascara-Frutal",
        "Frutal de Cascara-Olivar", "Frutal de Cascara", "Frutal de Cascara-Viñedo","Frutal", "Imvernadero y cultivos bajo plastico",
        "Olivar-Citricos","Olivar-Frutal","Olivar","Tierra Arable","Huerta","Frutal-Viñedo","Viñedo","Olivar-Viñedo"],
        index=["TP","FN", "Hit rate"])

    crop_codes = [3,4,5,6,9,10,12,13,14,16,17,18,19,23,24,25,26,27]
    truep = []
    falsen = []
    hr = []
    for crop_type in crop_codes:
#        print(crop_type)
        # print(cont)
        index_code_sigpac = np.where(sigpac_band == crop_type)
        values_cl = classification_band[index_code_sigpac]
        tp = len(np.where(values_cl == 6)[0])
        truep.append(tp)
        # data_output.loc['TP',cont] = tp
        print(len(index_code_sigpac))
        print(len(values_cl))
        index_not_crop = np.where(classification_band != 6)
        print(len(index_not_crop))
        values_sg = sigpac_band[index_not_crop]
        fn = len(np.where(values_sg == crop_type)[0])
        falsen.append(fn)
        # data_output.loc['FN',cont] = fn
        try:
            hit_rate = tp/(tp+fn)
            hr.append(hit_rate)

        except ZeroDivisionError:
            pass
        # data_output.loc['Hit rate',cont] = hit_rate

        # data_output.loc["TP"] = truep
        # data_output.loc["FN"] = falsen
        # data_output.loc["Hit rate"] = hr

        print(data_output)
        data_output.to_csv(output_path) 

# crop_metrics(sigpac_band, classification_band)

def process_dataframe(data_path):
    '''Given the dataframe created in crop_metrics() function the df is processed. 
    It now stores the hit percentage and number of pixels

    Args:
        data_path (str): Path to the dataframe.

    Return:
        The new data will be printed in terminal.
    '''
    dataframe = pd.read_csv(data_path)

    tp = (0,3,6,9,12,15,18,21)
    tn = (1,4,7,10,13,16,19,22)
    aciertos=0
    fallos=0
    row = 1
    for i in range(18):
        aciertos = 0
        fallos = 0
        porcentaje = 0
        num_pixeles = 0
        for num in tp: 
            aciertos += dataframe[dataframe.columns[row]][num]
        for num in tn: 
            fallos += dataframe[dataframe.columns[row]][num]
        porcentaje = ((aciertos)/(aciertos+fallos))*100
        num_pixeles = aciertos+fallos
        print(dataframe.columns[row],",",int(aciertos),",",int(fallos),",",round(porcentaje,2),",",int(num_pixeles))
        row+=1

# process_dataframe("./csv/andalucia_tp_tn.csv")

# "./raster_comparison_malaga.tif"
def validation(path: str) -> Tuple[float, float]:
    '''With a given compared raster, create a 2x2 confusion matrix 
    to validate the lab raster's performance crop classification.

    Args:
        path (str): Raster we want to work with.

    Returns:
        This function writes in the terminal the metrics.
    '''

    with rasterio.open(path) as src:
        band_matrix = src.read()
        # rows = band_matrix.shape[0] #* 10654
        # cols = band_matrix.shape[1] #* 16555
        green = np.where(band_matrix==20)
        red = np.where(band_matrix==21)
        blue = np.where(band_matrix==22)
        black = np.where(band_matrix==23)
        white = np.where(band_matrix==0)

        tp = len(band_matrix[green])
        tn = len(band_matrix[black])
        fp = len(band_matrix[red])
        fn = len(band_matrix[blue])
        na = len(band_matrix[white])

        print(tp)
        print(tn)
        print(fp)
        print(fn)
        print(na)
        print("-----------------")
        # print(len(band_matrix[green]))
        # print(band_matrix[green])
        accuracy = (tp+tn)/(tp+tn+fp+tn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2/((1/precision)+(1/recall))
        sensitivity = tp/(tp+fn)
        specificity = fp/(fp+tn)
        tp_rate = sensitivity
        fp_rate = 1-specificity

        print("Accuracy: ",accuracy)
        print("Precision: ",precision)
        print("Recall: ",recall)
        print("F1-Score:",f1_score)
        print("Sensitivity:", sensitivity)
        print("TruePositiveRate: ",tp_rate)
        print("FalsePositiveRate: ",fp_rate)
        print("-------------------------")

    return fp_rate, tp_rate

# x,y = validation("C:\\TFG_resources\\satelite_images_sigpac\\results\\malaga\\raster_comparison_malaga.tif")
# x2,y2 = validation("C:\\TFG_resources\\satelite_images_sigpac\\results\\raster_comparison_cordoba2.tif")
# x3,y3 = validation("C:\\TFG_resources\\satelite_images_sigpac\\results\\raster_comparison_granada.tif")
# x4,y4 = validation("C:\\TFG_resources\\satelite_images_sigpac\\results\\sevilla\\raster_comparison_sevilla.tif")
# x_oliv,y_oliv = validation("/home/jesus/Documents/satelite_images_sigpac/FPandFN_raster_comparison_olive_jaen.tif")

#!---------------------------------------------------------------------------------------------------------------------------------------

def graphs():
    # with open("/home/jesus/Documents/satelite_images_sigpac/csv/andalucia.csv", mode='r') as file:
    #     df = pd.read_csv(file)
    #     print(df.iloc[[1]])
    #     plt.bar(df.iloc[[3]],df.iloc[[0]],'o')
    #     plt.show()
    return

# graphs()

# plt.plot(x2,y2,'o')
# plt.plot(x3,y3,'o')
# plt.plot(x4,y4,'o')
# plt.xlim((0,1))
# plt.ylim((0,1))
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.show()

#TODO SOLUCIONAR PATHS A DIRECTORIOS CON JSON/MINIO (¿necesario?)
#TODO MIRAR SI ES VIABLE HACER UK (IN PROCESS)->(PARECE QUE NO)

# ypoints = np.array([0, y, y2, y3])
# plt.plot(ypoints, linestyle = 'dotted')

#!---------------------------------------------------------------------------------------------------------------------------------------

# ecw_file = gdal.Info("C:\\Users\\Pepe Aldana\\Documents\\PNOA-H_SIGPAC_OF_ETRS89_HU30_h50_1065.ecw")
# tranls = gdal.Translate(ecw_file)
# print(tranls)
# print(ecw_file)

# data = gdal.Open()
# geo_transform = data.GetGeoTransform()
# source_layer = data.GetLayer()
# print(geo_transform)
# print(source_layer)
# print(data)

# ndsm = 'C:\\Users\\Pepe Aldana\\Documents\\PNOA-H_SIGPAC_OF_ETRS89_HU30_h50_1065.ecw'
# shp = 'C:\TFG_resources\shape_files\Malaga_Municipios_Separados\SP20_REC_29012.shp'
# data = gdal.Open(ndsm, gdalconst.GA_ReadOnly)
# geo_transform = data.GetGeoTransform()
# source_layer = data.GetLayer()
# x_min = geo_transform[0]
# y_max = geo_transform[3]
# x_max = x_min + geo_transform[1] * data.RasterXSize
# y_min = y_max + geo_transform[5] * data.RasterYSize
# x_res = data.RasterXSize
# y_res = data.RasterYSize
# mb_v = ogr.Open(shp)
# mb_l = mb_v.GetLayer()
# pixel_width = geo_transform[1]
# output = 'C:\TFG_resources\satelite_images_sigpac\output'
# target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Byte)
# target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, pixel_width))
# band = target_ds.GetRasterBand(1)
# NoData_value = -999999
# band.SetNoDataValue(NoData_value)
# band.FlushCache()
# gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE=hedgerow"])

# target_ds = None

def prueba_de_uk():

    with fiona.open("C:/TFG_resources/RPA_CropMapOfEngland2020CAM_SHP_Full/data/Crop_Map_of_England_2020_Cambridgeshire.shp", "r") as json_cambridge:
        print(json_cambridge)
        # shapes = [feature["geometry"] for feature in json_cambridge]
        for feature in json_cambridge:
            print(feature['properties'])

            # print(feature["geometry"])
        # print(shapes)

# prueba_de_uk()

def prueba_corine_tiff():
  with rasterio.open("C:/TFG_resources/satelite_img/W020N60_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif", "r") as src:
    band = src.read()
    print(band)
    rows = band.shape[0] #* 16555
    cols = band.shape[1] #* 16555
    print(rows)
    print(cols)
    print(len(band[0][900]))
    pink = np.where(band==40)
    print(pink)
    crop = len(pink[0]) + len(pink[1]) + len(pink[2])
    print(crop)

# prueba_corine_tiff()
