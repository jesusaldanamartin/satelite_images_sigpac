from asyncio.windows_events import NULL
from cmath import nan
from operator import contains
from posixpath import split
from queue import Queue
from tkinter.ttk import Separator
from typing import List
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp
import rasterio._err
from rasterio.merge import merge
from rasterio.plot import show
from pathlib import Path
from os import getpid
from os import listdir, sep
from os.path import isfile, join
import tifftools
from tifftools import TifftoolsError
import matplotlib.pyplot as plt
import math
from shapely.geometry import Polygon, MultiPolygon,MultiLineString, MultiPoint
import fiona
import geojson
from rasterio import mask as msk
from rasterio.windows import Window
import geopandas as gpd
import geocoder
import utm
from rastertodataframe import raster_to_dataframe
from PIL import Image, ImageDraw
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
from osgeo import ogr
import json
import time
import os
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from shapely.geometry import mapping, Point
from shapely.ops import transform

import rioxarray as rxr
import xarray as xr
import geopandas as gpd
import pyproj
import warnings
from shapely.errors import ShapelyDeprecationWarning
import shapely.wkt
from rasterio.transform import from_origin
# from rasterio.crs import crs
from tqdm import tqdm 
from multiprocessing import Process, Manager
from numba import jit, njit

cod_uso = ['AG' ,'CA', 'CF' ,'CI', 'CS', 'CV', 'ED', 'EP', 'FF', 'FL', 'FO',
'FS', 'FV', 'FY', 'IM', 'IV', 'OC', 'OF', 'OV', 'PA', 'PR', 'PS','TA', 'TH',
'VF', 'VI', 'VO', 'ZC', 'ZU', 'ZV' ]

def get_id_codigo_uso(key):
    '''Raster bands cant have String value so each cod_uso it is been replaced with an id'''
    if key == 'AG' : return 10 #* Corrientes y superficies de agua
    if key == 'CA' : return 11 #* Viales
    if key == 'CF' : return 12 #* Citricos-Frutal
    if key == 'CI' : return 13 #* Citricos
    if key == 'CS' : return 14 #* Citricos-Frutal de cascara
    if key == 'CV' : return 15 #* Citricos-Viñedo
    if key == 'ED' : return 16 #* Edificaciones
    if key == 'EP' : return 17 #* Elemento del Paisaje
    if key == 'FF' : return 18 #* Frutal de Cascara-Frutal
    if key == 'FL' : return 19 #* Frutal de Cascara-Olivar
    if key == 'FO' : return 20 #* Forestal
    if key == 'FS' : return 21 #* Frutal de Cascara
    if key == 'FV' : return 22 #* Frutal de Cascara-Viñedo
    if key == 'FY' : return 23 #* Frutal
    if key == 'IM' : return 24 #* Improductivo
    if key == 'IV' : return 25 #* Imvernadero y cultivos bajo plastico
    if key == 'OC' : return 26 #* Olivar-Citricos
    if key == 'OF' : return 27 #* Olivar-Frutal
    if key == 'OV' : return 28 #* Olivar
    if key == 'PA' : return 29 #* Pasto Arbolado
    if key == 'PR' : return 30 #* Pasto Arbustivo
    if key == 'PS' : return 31 #* Pastizal
    if key == 'TA' : return 32 #* Tierra Arable
    if key == 'TH' : return 33 #* Huerta
    if key == 'VF' : return 34 #* Frutal-Viñedo
    if key == 'VI' : return 35 #* Viñedo
    if key == 'VO' : return 36 #* Olivar-Viñedo
    if key == 'ZC' : return 37 #* Zona Concentrada
    if key == 'ZU' : return 38 #* Zona Urbana
    if key == 'ZV' : return 39 #* Zona Censurada


def mask_shp():
    '''Given a SIGPAC shape file and a tif 
    crop the tif image by the geometries of the shp
    '''
    with fiona.open("C:\TFG_resources\satelite_images_sigpac\Shapefile_Data\SP20_REC_29054.shp", "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open("C:\TFG_resources\satelite_img\classification_30SUF.tif") as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
    
    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

    with rasterio.open("29054_masked.tif", "w", **out_meta) as dest:
        dest.write(out_image)
        
# mask_shp()

@jit
def is_point_in_polygon(x: int, y: int, poly) -> bool:
    num = len(poly)
    j = num - 1
    c = False
    for i in range(num):
        if (x == poly[i][0]) and (y == poly[i][1]):
            #* point is a corner
            return True
        if ((poly[i][1] > y) != (poly[j][1] > y)):
            slope = (x-poly[i][0])*(poly[j][1]-poly[i][1])-(poly[j][0]-poly[i][0])*(y-poly[i][1])
            if slope == 0:
                #* point is on boundary
                return True
            if (slope < 0) != (poly[j][1] < poly[i][1]):
                c = not c
        j = i
    return c

@jit
def index_values(lista_points,polygon):
    index_points=[]
    for i in range(len(lista_points)): 
        # bool = core_algorithm(lista_points[i],polygon)
        # print(i)
        bool = is_point_in_polygon(lista_points[i][0],lista_points[i][1],polygon)
        if bool: 
            index_points.append(i)
    return index_points

def get_band_matrix(path,points_list,arr,transformer):
    ind_values=[]

    with fiona.open(path) as layer:
        for feature in tqdm(layer):
            ord_dict = feature['properties']
            
            for key in ord_dict.values():
                if key in cod_uso: 
                    cd_uso = get_id_codigo_uso(key) #* codigo sigpac para cada iteraccion del shapefile
            geometry = feature["geometry"]['coordinates']
            for g in geometry:
                ind_values = index_values(points_list, g)

                if len(ind_values) != 0: #* no todas las parcelas tienen al menos un pixel del arr en su interior
                    # print(ind_values)
                    for ind in ind_values:
                        ind_arr = transformer.rowcol(points_list[ind][0],points_list[ind][1])
                        arr[ind_arr[0],ind_arr[1]] = cd_uso #* sustituye el valor de la banda anterior por el del SIGPAC
    return arr

# get_band_matrix("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp", list_pixels2)


with rasterio.open("C:\\TFG_resources\\satelite_images_sigpac\\29054_masked.tif") as src:
    profile = src.profile #* raster meta-data
    arr = src.read(1) #* band
    transformer = rasterio.transform.AffineTransformer(profile['transform'])
    not_zero_indices = np.nonzero(arr) #* Get all indexes of non-zero values in array to reduce its size

    points_list = [transformer.xy(not_zero_indices[0][i],not_zero_indices[1][i]) for i in tqdm(range(len(not_zero_indices[0])))] #* coordinates list of src values
    print(len(points_list)) 

    matrix = get_band_matrix("C:\TFG_resources\satelite_images_sigpac\Shapefile_Data\SP20_REC_29054.shp",
                            points_list, arr, transformer)

    with rasterio.open('29054_sigpac.tif', 'w', **profile) as dst:
        dst.write(matrix, 1)