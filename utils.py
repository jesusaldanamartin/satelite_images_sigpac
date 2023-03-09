from rasterio import (
    rasterio,
    mask,
    merge
)

from rasterio.warp import (
    Resampling,
    reproject,
    calculate_default_transform
)

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import fiona
from typing import List
import warnings

import multiprocessing
from multiprocessing import Manager

from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

COD_USO = ['AG', 'CA', 'CF', 'CI', 'CS', 'CV', 'ED', 'EP', 'FF', 'FL', 'FO',
           'FS', 'FV', 'FY', 'IM', 'IV', 'OC', 'OF', 'OV', 'PA', 'PR', 'PS',
           'TA', 'TH', 'VF', 'VI', 'VO', 'ZC', 'ZU', 'ZV']

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
         '30TVN', '30TWN', '30TXN', '30TYN', '31TCH']


def get_id_codigo_uso(key: str):
    '''Raster bands cannot have string value so each cod_uso it is been replaced with an id.

    Args:
        key (str): Key stored in shp metadata.

    Returns:
        None
    '''
    if key == 'AG':
        return 1  # * Corrientes y superficies de agua
    if key == 'CA':
        return 2  # * Viales
    if key == 'CF':
        return 3  # * Citricos-Frutal
    if key == 'CI':
        return 4  # * Citricos
    if key == 'CS':
        return 5  # * Citricos-Frutal de cascara
    if key == 'CV':
        return 6  # * Citricos-Viñedo
    if key == 'ED':
        return 7  # * Edificaciones
    if key == 'EP':
        return 8  # * Elemento del Paisaje
    if key == 'FF':
        return 9  # * Frutal de Cascara-Frutal
    if key == 'FL':
        return 10  # * Frutal de Cascara-Olivar
    if key == 'FO':
        return 11  # * Forestal
    if key == 'FS':
        return 12  # * Frutal de Cascara
    if key == 'FV':
        return 13  # * Frutal de Cascara-Viñedo
    if key == 'FY':
        return 14  # * Frutal
    if key == 'IM':
        return 15  # * Improductivo
    if key == 'IV':
        return 16  # * Imvernadero y cultivos bajo plastico
    if key == 'OC':
        return 17  # * Olivar-Citricos
    if key == 'OF':
        return 18  # * Olivar-Frutal
    if key == 'OV':
        return 19  # * Olivar
    if key == 'PA':
        return 20  # * Pasto Arbolado
    if key == 'PR':
        return 21  # * Pasto Arbustivo
    if key == 'PS':
        return 22  # * Pastizal
    if key == 'TA':
        return 23  # * Tierra Arable
    if key == 'TH':
        return 24  # * Huerta
    if key == 'VF':
        return 25  # * Frutal-Viñedo
    if key == 'VI':
        return 26  # * Viñedo
    if key == 'VO':
        return 27  # * Olivar-Viñedo
    if key == 'ZC':
        return 28  # * Zona Concentrada
    if key == 'ZU':
        return 29  # * Zona Urbana
    if key == 'ZV':
        return 30  # * Zona Censurada


def reproject_raster(in_path: str, out_path: str, file_name: str, new_crs):
    '''Given an in and out path this function reproject a raster into any coordinate reference system set.

    Args:
        in_path (str): Path to the raster we want to reproject.
        out_path (str):  Output path where the raster will be saved.
        file_name (str): Name of the new raster.
        new_crs (str): New coordinate reference system (EPSG:4258 ETRS89).

    Return:
        Save the new raster in out_path.
    '''

    # reproject raster to project crs
    with rasterio.open(in_path) as src:
        src_crs = src.crs
        transform, width, height = calculate_default_transform(
            src_crs, new_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()

        kwargs.update({
            'crs': new_crs,
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
                dst_crs=new_crs,
                resampling=Resampling.nearest)


def mask_shp(shp_path: str, tif_path: str, output_name: str):
    '''Crop a tif image with the shapefile geoemetries.

    Args:
        shp_path (str): Path to the shapefile file.
        tif_path (str): Path to the tif image.
        output_name (str): File name to be saved.

    Returns:
        Save the tif cropped in working directory.
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
        # "crs": "+proj=utm +zone=30 +ellps=WGS84 +units=m +no_defs"
    })

    with rasterio.open(output_name, "w", **out_meta) as dest:
        dest.write(out_image)


def masked_all_shapefiles_in_directory(folder_path: str, output_path: str, mask: str):
    '''Read all shapefiles stored in directory and create a mask for each file.

    Args:
        folder_path (str): Path to the directory where all shapefiles are stored.
        output_path (str): Path to the output directory.
        mask (str) : Path to the .tif image that is going to be used to mask the shapefiles.

    Return:
        A masked file is created for each shapefile in the folder.
    '''

    folder_files = os.listdir(folder_path)
    for file in tqdm(folder_files):
        extension = file.split('.')[1]
        file_name = file.split('.')[0]
        if extension == 'shp':
            try:
                mask_shp(folder_path+f"/{file}", mask,
                         output_path+f"{file_name}_mask.tif")
            except ValueError:
                print(file+" does not overlap figure")


def merge_tiff_images_in_directory(folder_path: str, output_path: str, file_name: str):
    '''Merge all tiff images stored in folder_path.

    All metadata is saved and stored so the output will be be only a merge of all images given as input.

    Args:

        folder_path (str): Path to the folder where tiff images are.
        output_name (str): Name as the output file will be stored.
        file_name (str): Name of the file to be saved.

    Returns: 
        The merged image will be stored in the working directory.
    '''

    src_files_to_mosaic = []
    folder_files = os.listdir(folder_path)
    out_meta = {"null:null"}

    for file in folder_files:
        src = rasterio.open(folder_path+f"/{file}")
        src_files_to_mosaic.append(src)
        src_crs = src.crs  # * For a new custom crs output

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
            return True  # * point is a corner
        if ((polygon[i][1] > y) != (polygon[j][1] > y)):
            slope = (x-polygon[i][0])*(polygon[j][1]-polygon[i][1]) - \
                (polygon[j][0]-polygon[i][0])*(y-polygon[i][1])
            if slope == 0:
                return True  # * point is on boundary
            if (slope < 0) != (polygon[j][1] < polygon[i][1]):
                # * if an edge is crossed an odd number of times the point is in geometry
                in_geometry = not in_geometry
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

    index_points = []
    size = len(points_list)
    for i in range(size):
        bool = is_point_in_polygon(
            points_list[i][0], points_list[i][1], polygon)
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

    dict_tuples = []
    ind_values = []
    cd_uso = 0
    with fiona.open(path) as layer:
        for feature in tqdm(layer):
            ord_dict = feature['properties']

            for key in ord_dict.values():
                if key in COD_USO:
                    # * save the SIGPAC use code for that exact geometry
                    cd_uso = get_id_codigo_uso(key)
            geometry = feature["geometry"]['coordinates']

            for g in geometry:
                ind_values = index_values(points_list, g)

                if len(ind_values) != 0:

                    for ind in ind_values:
                        ind_arr = transformer.rowcol(
                            points_list[ind][0], points_list[ind][1])

                        dict_tuples.append((ind_arr, cd_uso))

    d[multiprocessing.current_process().pid] = dict_tuples


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
            # * replace the new use code from SIGPAC in the old band value.
            arr[value[0][0], value[0][1]] = value[1]
    return arr


def start_parallel_execution(points_list: list, shp_path: str, arr: np.ndarray, transformer: rasterio.Affine) -> np.ndarray:
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

    manager = Manager()
    d = manager.dict()

    for i in range(size):
        p = multiprocessing.Process(target=replace_band_matrix, args=(
            shp_path, chunked_list[i], arr, transformer, d))
        p_list.append(p)
        p.start()

    for p in p_list:
        p.join()  # * wait until all threads have finished

    new_arr = replace_values(d, arr)

    return new_arr


def save_output_file(shp_path: str, tif_path: str, output_path: str):
    '''Final raster file created.

    Read the given .tif, some metadata is saved and function replace_band_matrix() is called.

    Args:
        shp_path (str): Path to the shapefile.
        tif_path (str): Path to the tif image.
        output_path (str): Path to be saved.

    Returns:
        Save in working directory the final image.
    '''

    with rasterio.open(tif_path) as src:
        profile = src.profile  # * raster metadata
        arr = src.read(1)
        transformer = rasterio.transform.AffineTransformer(
            profile['transform'])
        # * get all indexes of non-zero values in array to reduce its size
        not_zero_indices = np.nonzero(arr)

        points_list = [transformer.xy(not_zero_indices[0][i], not_zero_indices[1][i])
                       for i in tqdm(range(len(not_zero_indices[0])))]  # * coordinates list of src values #TQDM

        new_arr = start_parallel_execution(
            points_list, shp_path, arr, transformer)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(new_arr, 1)


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
        file_name = file.split('_')[0]

        if file_name+f"_sigpac.tif" not in os.listdir(sigpac_data_folder):
            print(file)

            save_output_file(shp_data_folder+f"/{file[:-11]}_RECFE.shp",
                             folder_path+f"/{file}",
                             sigpac_data_folder+f"{file_name}_sigpac.tif")
            print(file+" have finished")
            print("")


# merge_tiff_images_in_directory("/home/jesus/Documents/TFG/satelite_images_sigpac/data/CastillaLeon/masked_Burgos", "/home/jesus/Documents/TFG/satelite_images_sigpac/data/","burgos_masked.tif")

# read_masked_files("../satelite_images_sigpac/data/CastillaLeon/masked_Burgos/",
#                   "../satelite_images_sigpac/data/CastillaLeon/09_Burgos", "../satelite_images_sigpac/data/CastillaLeon/sigpac_Burgos/")
