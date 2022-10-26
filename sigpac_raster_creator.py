from rasterio import (
    rasterio,
    features,
    warp,
    merge,
    plot,
    mask,
    windows,
    transform,
    crs
)
import rasterio._err
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, MultiPoint, Point

from tqdm import tqdm
from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import fiona
import numpy as np
from multiprocessing import Process, Manager
import threading

from typing import Any, List, Tuple, BinaryIO
from pathlib import Path
import os
from os import listdir
from os.path import isfile, join
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

COD_USO = ['AG' ,'CA', 'CF' ,'CI', 'CS', 'CV', 'ED', 'EP', 'FF', 'FL', 'FO',
            'FS', 'FV', 'FY', 'IM', 'IV', 'OC', 'OF', 'OV', 'PA', 'PR', 'PS','TA', 'TH',
            'VF', 'VI', 'VO', 'ZC', 'ZU', 'ZV' ]

def get_id_codigo_uso(key: str) -> None:
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

    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

    with rasterio.open(output_name, "w", **out_meta) as dest:
        dest.write(out_image)

def masked_all_shapefiles_in_directory(folder_path: str):
    '''Read all shapefiles stored in directory and create a mask for each file.

    Args:
        folder_path (str): Path to the directory where all shapefiles are stored.
    
    Return:
        A file is created for each shp in folder.
    '''

    path_masked_images = "/home/jesus/Documents/satelite_images_sigpac/Satelite_Images/masked_images/MALAGA/"
    path_tif_image = "/home/jesus/Documents/satelite_images_sigpac/Satelite_Images/malagaMask.tif"

    folder_files = os.listdir(folder_path)
    for file in tqdm(folder_files):
        extension = file.split('.')[1]
        file_number = file.split('.')[0]
        if extension == 'shp':
            try:
                mask_shp(folder_path+f"/{file}", path_tif_image, path_masked_images+f"29{file_number[-3:]}_masked.tif")
            except ValueError:
                print(file+" does not overlap figure")

    return "FINISHED"

#? masked_all_shapefiles_in_directory("/home/jesus/Documents/satelite_images_sigpac/Shapefile_Data")

def merge_tiff_images(output_name: str, folder_path: str):
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

    for file in folder_files:
        src = rasterio.open(file)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge.merge(src_files_to_mosaic)

    out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": "+proj=utm +zone=30 +ellps=WGS84 +units=m +no_defs "
        }
        )
    try:
        with rasterio.open(output_name, "w", **out_meta) as dest:
            dest.write(mosaic)
    except rasterio._err.CPLE_BaseError:
        print("Merged file already exists")
        pass
    return mosaic

# merge_tiff_images("malagaMask_sigpac.tif","/home/jesus/Documents/satelite_images_sigpac/Satelite_Images/masked_sigpac/MALAGA/")

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

def replace_band_matrix(path: str, points_list: list, arr: np.ndarray, transformer: rasterio.Affine) -> np.ndarray:
    '''Replace in the raster band the use code from SIGPAC.

    Args:
        path (str): Path to the shapefile.
        points_list (List[(x,y)]): List of all pixels coordinates of the raster.
        arr (np.ndarray): Numpy array with the raster's band information.
        transformer (rasterio.Affine): 'rasterio.transform.AffineTransformer' to convert from (x,y) to band value.

    Returns:
        arr (np.ndarray): Numpy array with band values changed to SIGPAC use code.
    '''

    ind_values=[]
    with fiona.open(path) as layer:
        for feature in tqdm(layer):#TODO TQDM
            ord_dict = feature['properties']

            for key in ord_dict.values():
                if key in COD_USO: 
                    cd_uso = get_id_codigo_uso(key) #* save the SIGPAC use code for that exact geometry
            geometry = feature["geometry"]['coordinates']

            for g in geometry:
                ind_values = index_values(points_list, g)
                if len(ind_values) != 0:
                    # print(ind_values)

                    for ind in ind_values:
                        ind_arr = transformer.rowcol(points_list[ind][0],points_list[ind][1])
                        arr[ind_arr[0],ind_arr[1]] = cd_uso #* replace the new use code from SIGPAC in the old band value.
    # return arr


def multithreading(points_list: list, shp_path: str, arr: np.ndarray, transformer: rasterio.Affine) -> None:
    '''Execute the process with multiple threads improving performance.

    Args:
        points_list (List[(x,y)]): List of all pixels coordinates of the raster.
        shp_path (str): Path to the shapefile.
        arr (np.ndarray): Numpy array with the raster's band information.
        transformer (rasterio.Affine): 'rasterio.transform.AffineTransformer' to convert from (x,y) to band value.
    
    Returns:
        None
    '''

    chunked_list = np.array_split(points_list, 3)
    size = len(chunked_list)
    p_list = []

    for i in range(size):
        p = threading.Thread(target=replace_band_matrix, args=(shp_path, chunked_list[i], arr, transformer))
        p.start()
        p_list.append(p)

    for p in p_list:
        #* wait until all threads have finished
        p.join()

def save_output_file(tif_path: str, shp_path: str,output_name: str):
    '''Final raster file created.

    Read the given .tif, some metadata is saved and function replace_band_matrix() is called.

    Args:
        tif_path (str): Path to the tif image.
        shp_path (str): Path to the shapefile.
        output_name (str): File name to be saved.

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
        multithreading(points_list, shp_path, arr, transformer)

        with rasterio.open(output_name, 'w', **profile) as dst:
            dst.write(arr, 1)

# save_output_file("/home/jesus/Documents/satelite_images_sigpac/Satelite_Images/masked_images/29012_masked.tif",
#                 "/home/jesus/Documents/satelite_images_sigpac/Shapefile_Data/SP20_REC_29012.shp",
#                 "29012_sigpac.tif")

def read_masked_files(folder_path):
    '''For every masked file in folder save_output_file() function is called 
    to convert input masked files into masked sigpac tif.

    Args:
        folder_path (str): Path to the folder with all masked files.

    Returns:
        One file for each shapefile.
    '''

    path_shapefile_data = "/home/jesus/Documents/satelite_images_sigpac/Shapefile_Data"
    path_sigpac = "/home/jesus/Documents/satelite_images_sigpac/masked_sigpac/MALAGA/"
    folder_files = os.listdir(folder_path)

    for file in folder_files:
        file_number = file.split('.')[0]

        if os.path.getsize(folder_path+f"{file}") < 1764228 and file_number[0:5]+f"_sigpac.tif" not in os.listdir(path_sigpac):
            print(file)
            print("")
            save_output_file(folder_path+f"/{file}",
                            path_shapefile_data+f"/SP20_REC_29{file_number[2:5]}.shp",
                            path_sigpac+f"29{file_number[2:5]}_sigpac.tif")
            print("")
            print("File: "+file+" finished")
read_masked_files("/home/jesus/Documents/satelite_images_sigpac/Satelite_Images/masked_images/MALAGA/")


#? Windows Path for save_output_file
# save_output_file("C:\\TFG_resources\\satelite_images_sigpac\\29008_masked.tif",
#                 "C:\\TFG_resources\\satelite_images_sigpac\\Shapefile_Data\\SP20_REC_29008.shp",
#                 "29008_sigpac.tif")

#? Windows Path for mask_shp
# mask_shp("C:\TFG_resources\satelite_images_sigpac\Shapefile_Data\SP20_REC_29017.shp",
#             "C:\TFG_resources\satelite_img\classification_30SUF.tif", 
#             "29017_masked.tif")
