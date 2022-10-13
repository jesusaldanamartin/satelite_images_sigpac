from asyncio.windows_events import NULL
from cmath import nan
from operator import contains
from posixpath import split
from tkinter.ttk import Separator
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp
import rasterio._err
from rasterio.merge import merge
from rasterio.plot import show
from pathlib import Path
import os
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
import multiprocessing 
PATH = "C:\TFG_resources\satelite_img"
PATH_TO_GJSON = "C:\TFG_resources\satelite_images_sigpac\GeoJson_Data"


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
    '''Get all tiles name and paths'''
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
    '''Tiff tool library merge tiles'''
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
            print("Tifftools merge file already exists")
            pass

# merge_tiles_tifftools("mask.tif")

def merge_tiles_rasterio(output_name):
    '''Rasterio library merge tiles'''
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
            "crs": "+proj=utm +zone=30 +ellps=WGS84 +units=m +no_defs "
            }
            )
        try:
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(mosaic)
        except rasterio._err.CPLE_BaseError:
            print("Rasterio merge file already exists")
            pass
    return mosaic

# south_spain_rasterio = merge_tiles_rasterio("mask.tif")
# merged_file = PATH+f"/mask.tif"

def create_rasterio_mask_by_geojson_polygon(raster,province,output_name):

    with fiona.open(province, "r") as shapefile: # To mask the data
        # shapes = [feature['geometry']['coordinates'] for feature in shapefile]
        shapes = [feature["geometry"] for feature in shapefile]
    with rasterio.open(raster) as src:
        output_tif, transformed = rasterio.mask.mask(src, shapes, crop=True)
        out_profile = src.profile.copy()

    out_profile.update({'width': output_tif.shape[2],'height': output_tif.shape[1], 'transform': transformed})
    with rasterio.open(output_name, 'w', **out_profile) as dst:
        dst.write(output_tif)

    return output_tif

# create_rasterio_mask_by_geojson_polygon("C:\TFG_resources\satelite_images_sigpac\malagaMask.tif",
#                             "C:\TFG_resources\satelite_images_sigpac\recintos.geojson",
#                             "hola.tif")

def swapCoords(x):
    '''Swap coordinates from column'''

    out = []
    for item in x:
        if isinstance(item, list):
            out.append(swapCoords(item))
        else:
            return [x[1], x[0]]
    return out

def geojson_coords_into_lat_lon():
    '''Change all coords in geojson in lat/lon order'''

    coordinates = []
    with open("C:\TFG_resources\satelite_images_sigpac\GeoJson_Data\cartama.geojson") as f:
        response = geojson.load(f)

    for feature in response['features']:
        feature['geometry']['coordinates'] = swapCoords(feature['geometry']['coordinates'])
        coordinates.append(feature['geometry']['coordinates'])

    return response, coordinates

# gj, coordinates = geojson_coords_into_lat_lon()

def swap_xy_coords(coords):
    for x, y in coords:
        return (y, x)

def utm_to_wgs(bool, tuple):
    if bool:
        wgs = utm.from_latlon(tuple[0],tuple[1])
        return (wgs[0],wgs[1])
    else:
        wgs = utm.from_latlon(tuple[1],tuple[0])
        return (wgs[0],wgs[1])

def get_coordinates_in_geojson_format(path):

    with fiona.open(path) as layer:
        multipolygon = [feature['geometry'] for feature in layer]
        counter=0
        while counter < len(multipolygon):
            i=0
            for coord in multipolygon[counter]['coordinates'][0]:
                multipolygon[counter]['coordinates'] = coord
                i+=1
            counter+=1
    return multipolygon

def mask_raster_by_sigpac_shape():

    gjson = get_coordinates_in_geojson_format("C:\\TFG_resources\\satelite_images_sigpac\\GeoJson_Data\\geoJson_alhaurin.geojson")

    with rasterio.open("C:\TFG_resources\satelite_images_sigpac\malagaMask.tif") as src:
            output_tif, transformed = rasterio.mask.mask(src, gjson, crop=True, invert=False, filled=False)
            out_profile = src.profile.copy()
            out_profile.update({
                'width' : output_tif.shape[1],
                'height' : output_tif.shape[2],
                'transform' : transformed})

    with rasterio.open("funciona", 'w', **out_profile) as dst:
        dst.write(output_tif)

    return print("Shape exportada")

# mask_raster_by_sigpac_shape()

def shapefile_to_raster():
    # cod = get_codigo_parcela()
    # numpy_array = np.array(cod)
    source_ds = ogr.Open("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp")
    source_layer = source_ds.GetLayer()
    pixelWidth = pixelHeight = 10 # depending how fine you want your raster
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    cols = int((x_max - x_min) / pixelHeight)
    rows = int((y_max - y_min) / pixelWidth)
    print(cols)
    print(rows)
    target_ds = gdal.GetDriverByName('GTiff').Create('temp.tif', cols, rows, 1, gdal.GDT_Byte) 
    target_ds.SetGeoTransform((x_min, pixelWidth, 0, y_min, 0, pixelHeight))
    band = target_ds.GetRasterBand(1)
    NoData_value = 255
    band.SetNoDataValue(NoData_value)
    band.FlushCache()

    gdal.RasterizeLayer(target_ds, [1], source_layer) 

    target_dsSRS = osr.SpatialReference()
    target_dsSRS.ImportFromEPSG(4326)
    target_ds.SetProjection(target_dsSRS.ExportToWkt())
    target_ds = None  

    gdal.Open('temp.tif').ReadAsArray()
    print(gdal.Open('temp.tif').ReadAsArray())
    print(len(gdal.Open('temp.tif').ReadAsArray()))

# shapefile_to_raster()

def get_polygon_corners():
    '''Get bounds of every polygon in shapefile'''

    shapefile = "C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp"
    with fiona.open(shapefile) as layer:
        conjunto = [feature['geometry'] for feature in layer]
    counter=0
    x = []
    y = []
    while counter < len(conjunto):
        for recinto in conjunto[counter]['coordinates']:
            polygon = Polygon(recinto)
            xmin,xmax,ymin,ymax = polygon.bounds
            x.append(xmin)
            x.append(xmax)
            y.append(ymin)
            y.append(ymax)
            counter+=1
    max_lon = max(x)
    min_lon = min(x)
    max_lat = max(y)
    min_lat = min(y)

    return  {
        "top_left": (max_lat, min_lon),
        "top_right": (max_lat, max_lon),
        "bottom_left": (min_lat, min_lon),
        "bottom_right": (min_lat, max_lon),
    }

#! -----------------------------------------------------------------------
#! -----------------------------------------------------------------------
#! -----------------------------------------------------------------------
#! -----------------------------------------------------------------------
#! -----------------------------------------------------------------------
#! -----------------------------------------------------------------------
#! -----------------------------------------------------------------------
#! -----------------------------------------------------------------------
#! -----------------------------------------------------------------------
#! -----------------------------------------------------------------------

cod_uso = ['AG' ,'CA', 'CF' ,'CI', 'CS', 'CV', 'ED', 'EP', 'FF', 'FL', 'FO',
'FS', 'FV', 'FY', 'IM', 'IV', 'OC', 'OF', 'OV', 'PA', 'PR', 'PS','TA', 'TH',
'VF', 'VI', 'VO', 'ZC', 'ZU', 'ZV' ]

def get_dataframe_codigo_polygon():
    '''Return a dataframe with | NumberRow | cod_uso(string) | Geometry(polygon format) |'''
    df = pd.DataFrame(columns=['cod_uso','Geometry'])
    # array = []
    with fiona.open("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp") as layer:
        cont = 0
        cont2=0
        for feature in layer:
            ord_dict = feature['properties']
            for key in ord_dict.values():
                if key in cod_uso: 
                    df.loc[cont,'cod_uso']=key
                    # array.append(key)
                    cont+=1
            geometry = feature["geometry"]['coordinates']
            for g in geometry:
                multipolygon = Polygon(g)
                df.loc[cont2,'Geometry']=multipolygon
                cont2+=1
    return df.to_csv("df_codigo_polygon")

# get_dataframe_codigo_polygon()

def transform_x_y_into_point():
    raster_df = pd.read_csv("df_raster.csv", sep =",")
    raster_df = raster_df[raster_df['band1'] == 255]
    with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
            raster_df['Point'] = raster_df.apply(lambda row: Point(row['X'], row['Y']), axis = 1)
    raster_df = raster_df.iloc[: , 3:]

    return  raster_df.to_csv("df_raster_points")

# transform_x_y_into_point()

def create_raster_from_shp():

    input_shp = ogr.Open("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp")
    shp_layer = input_shp.GetLayer()
    pixel_size = 10
    x_min, x_max, y_min, y_max = shp_layer.GetExtent()
    print(x_min, x_max, y_min, y_max)

    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    image_type = 'GTiff'
    driver = gdal.GetDriverByName(image_type)

    #passing the filename, x and y direction resolution, no. of bands, new raster.
    new_raster = driver.Create("primera_prueba", x_res, y_res, 1, gdal.GDT_Byte)

    # transforms between pixel raster space to projection coordinate space.
    new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))

    #get required raster band.
    band = new_raster.GetRasterBand(1)

    #assign no data value to empty cells.
    no_data_value = -9999
    band.SetNoDataValue(no_data_value)
    band.FlushCache()

    gdal.RasterizeLayer(new_raster, [1], shp_layer)

    #adding a spatial reference
    new_rasterSRS = osr.SpatialReference()
    new_rasterSRS.ImportFromEPSG(32630)
    new_raster.SetProjection(new_rasterSRS.ExportToWkt())

# create_raster_from_shp()

def get_numpy_array_from_raster(filepath):
    '''Return data frame with | X | Y | band1|
        band1 binary value 0 == outside, 255 == inside mask'''
    src = rasterio.open(filepath)
    crs = src.crs
    array = src.read()
    xmin, ymax = np.around(src.xy(0.00, 0.00), 9) # Get the center of the raster
    xmax, ymin = np.around(src.xy(src.height-1, src.width-1), 9)  

    x = np.linspace(xmin, xmax, src.width)
    y = np.linspace(ymax, ymin, src.height) 

    xs, ys = np.meshgrid(x, y)

    data = {"X": pd.Series(xs.ravel()),
            "Y": pd.Series(ys.ravel())}

    raster_dataframe = pd.DataFrame(data=data)
    raster_dataframe['band1'] = array[0].ravel() 
    np.round(raster_dataframe["X"],9)
    np.round(raster_dataframe["Y"],9)

    return raster_dataframe.to_csv("df_raster")

# get_numpy_array_from_raster("C:\\TFG_resources\\satelite_images_sigpac\\primera_prueba")

def get_codigo_for_pixel():
    '''Nested for loops read two dataframes and get cod_uso for each pixel
        df1 = 8.000.000
        df2 = 50.000
        hundreds of millions of iterations
        '''
    polygon_df = pd.read_csv("df_codigo_polygon.csv", sep =",")
    # polygon_df = polygon_df[polygon_df['cod_uso'].isin(cod_uso)]
    # print(len(polygon_df))
    raster_points_df = pd.read_csv("df_raster_points.csv", sep=",")
    # print(len(raster_points_df))
    raster_df = pd.read_csv("df_raster.csv",sep=",")
    raster_df = raster_df[raster_df['band1'] == 255]

    # print(raster_points_df.head())
    # print(polygon_df.head())
    #3967902
    #3964000
    for i in range(len(raster_points_df)):

        point = raster_points_df.loc[i,'Point']
        wkt_point_geometry = ogr.CreateGeometryFromWkt(point)
        print("-----------",i,"-----------")
        for j in range(len(polygon_df)):

            polygon = polygon_df.loc[j,'Geometry']
            wkt_polygon_geometry = ogr.CreateGeometryFromWkt(polygon)
            if wkt_polygon_geometry.Contains(wkt_point_geometry):

                print("Pertenece a geometry")
                raster_df.loc[i,'Band2'] = polygon_df.loc[i,"cod_uso"]
                # raster_df[i,"Band2"] = polygon_df.loc[j,"cod_uso"]
                # print(polygon_df.loc[i,"cod_uso"])
            print("------------------",j,"-----------------")
            break
    return raster_df.to_csv("raster_with_bands.csv")

# get_codigo_for_pixel()

#! --------------------------------------------------------------------------------------
#! --------------------------------------------------------------------------------------
#! --------------------------------------------------------------------------------------
#! --------------------------------------------------------------------------------------
#! --------------------------------------------------------------------------------------
#! --------------------------------------------------------------------------------------
#! --------------------------------------------------------------------------------------
#! --------------------------------------------------------------------------------------

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

def x_y_to_point(row):
    '''Return for a row a shapely Point object'''
    return Point(row['X'],row['Y'])

def point_contains_geometry(row):
    '''Return for a row the coordinates and cod_uso'''
    if row['Polygon'].contains(row['Point']):
        x = row['Point'].x
        y = row['Point'].y
        id = row['band1']
        # data_set.loc[len(data_set)] = [x,y,id]
        return x,y,id
    else:
        return np.nan

# data_set = pd.DataFrame(columns=["XY","ID"])
def series_split(row):
    l = row[0].split(",")
    val0 = float(l[0][1:])
    val1 = float(l[1])
    try:
        val2 = float(l[2][1:3])
    except ValueError as e:
        val2 = float(l[2][1:2])
    output = [val0,val1,val2]
    # print(val0)
    # print(val1)
    # print(val2)
    # data_set.loc[len(data_set)] = output
    # print(data_set)
    return output


def is_in_geometry(polygon,df):
    '''Remove all pixels thar are not inside the sigpac polygon geometry'''
    with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
                df['Point'] = df.apply(x_y_to_point, axis = 1)
    df = df.assign(Polygon = polygon)
    df_xy = df.apply(point_contains_geometry, axis = 1)
    df_xy = df_xy.dropna()
    # print(df_xy)
    # # print(df_xy)
    # for i, el in enumerate(df_xy.to_numpy()):
    #     # print(i)
    #     print(el)
    #     print(el.split(","))

    return df_xy

def get_df_each_geometry(path):
    '''Return a data frame with X | Y | Band1 values
        now every coordinate is ensured that is contained in the geometry'''
    with fiona.open(path) as layer:
        dfs=[]
        cont=0
        for feature in layer:
            ord_dict = feature['properties']
            for key in ord_dict.values():
                if key in cod_uso: 
                    cd_uso = get_id_codigo_uso(key)
            geometry = feature["geometry"]['coordinates']
            for g in geometry:

                polygon = Polygon(g)
                print(polygon)
                xmin, ymin, xmax, ymax = polygon.bounds
                xmin = np.round(xmin,9)
                xmax = np.round(xmax,9)
                ymin = np.round(ymin,9)
                ymax = np.round(ymax,9)
                x_res = int((xmax - xmin) / 10)
                y_res = int((ymax - ymin) / 10)

                #Obtener un numpy array cada posicion un x,y teniendo todos los pixeles 
                # print(x_res)
                # print(y_res)
                x = np.linspace(xmin, xmax, x_res)
                array_g = np.zeros((x_res,y_res))
                # print(array_g)
                # print(array_g.shape)
                y = np.linspace(ymax, ymin, y_res) 
                # print(x)
                # print(y)

                xs, ys = np.meshgrid(x, y)
                # print(cd_uso)
                arr = np.array(cd_uso)
                # print(arr)
                # print(xs)
                # print(ys)
                data = {"X": pd.Series(xs.ravel()),
                        "Y": pd.Series(ys.ravel())}
                raster_dataframe = pd.DataFrame(data=data)
                raster_dataframe = raster_dataframe.assign(band1 = cd_uso)

                if not raster_dataframe.empty:

                    dataset = is_in_geometry(polygon ,raster_dataframe)
                    dfs.append(dataset)
                cont+=1
                print(cont)
    return pd.concat(dfs).to_csv("data_inside.csv")

# get_df_each_geometry("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp")

def write_data_into_raster_tiff():
    '''Data needed driver height width dtype crs transform'''
    data_frame = pd.read_csv("data_inside.csv", sep =",")
    del(data_frame[data_frame.columns[0]])
    print(len(data_frame))
    min_values = data_frame.min(axis=0)
    max_values = data_frame.max(axis=0)
    min_values = min_values[0]
    max_values = max_values[0]

    split_min = min_values.split(",")
    split_max = max_values.split(",")
    xmin = split_min[0]
    xmax = split_max[0]

    xmin = float(xmin[1:])
    ymin = float(split_min[1])
    ymax = float(split_max[1])
    xmax = float(xmax[1:])
    print(xmin)
    print(ymin)
    print(xmax)
    print(ymax)
    # xmin=data_frame['X'].min()
    # xmax=data_frame['X'].max()
    # ymin=data_frame['Y'].min()
    # ymax=data_frame['Y'].max()
    # xmin = min_values(0)
    # xmax = max_values(0)
    # ymin = min_values(1)
    # ymax = max_values(1)

    # print(xmin)
    # print(xmax)
    x_res = int((xmax - xmin) / 10)
    y_res = int((ymax - ymin) / 10)

    # print(x_res)
    # print(y_res)
    # data_set = pd.DataFrame(columns=["XY", "ID"])
    # cont=0
    df = data_frame.apply(series_split, axis=1)
    df = df.to_frame()
    l = df[0].tolist()
    my_array = np.asarray(l)
    print(my_array.dtype)
    print(type(my_array))
    print(my_array.shape)
    print(my_array)

    # data_frame = data_frame.to_numpy()
    # print(data_frame)

    # for i, el in enumerate(data_frame.to_numpy()):
    #     data_set.loc[len(data_set)] = [el[0],el[1]]
    #     cont+=1
    #     print(cont)
    # data_set.to_csv("xy_id_float.csv")
    # print("fin")
    # df = data_frame.reset_index(drop=True).to_numpy()
    # # my_array = df.to_numpy()
    # print(df)
    # print(df)
    # my_array = df[0].tolist()
    # print(my_array)
    # data_frame = data_frame.values
    # print(data_set)
    # my_array = np.asarray(data_frame)
    # print(my_array)
    # data_f = pd.DataFrame(columns=["X","Y","ID"])
    # cont=0
    # my_array = data_frame.to_numpy()
    # for i, el in enumerate(data_frame.to_numpy()):
    #     data_f.loc[len(data_f)] = el
    #     cont+=1
    #     print(cont)

    # print(data_f)
    # print(data_frame)

    driver = "GTIFF"
    height = 2844
    width = 2930
    count = 1
    dtype = my_array.dtype

    # crs = crs.from_epsg(32630)
    transforma = from_origin(xmin,ymax,10,10)
    print(transforma)
    print(dtype)
    print(my_array[:,[2]])
    band = my_array[:,[2]].astype(np.int64)
    
    print(band.shape)
    with rasterio.open("AAABmalaga29900_27_09.tif", "w",
                    driver=driver,
                    height=height,
                    width=width,
                    count=count,
                    dtype=dtype,
                    transform=transforma) as dst:
        dst.write_band(1, band)
        

        # dst.write(my_array)  
    return band

# band_cod = write_data_into_raster_tiff()

def mask_band():

    data_frame = pd.read_csv("data_inside.csv", sep =",")
    with fiona.open("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp") as shapefile:
        shapes_crop = [feature["geometry"] for feature in shapefile]
    del(data_frame[data_frame.columns[0]])
    df = data_frame.apply(series_split, axis=1)
    df = df.to_frame()
    l = df[0].tolist()
    my_array = np.asarray(l)
    shapes = my_array[:,0:2]

    print(shapes)

    xmin = np.amin(shapes[:,0])
    xmax = np.amax(shapes[:,0])
    ymin = np.amin(shapes[:,1])
    ymax = np.amax(shapes[:,1])
    # geometry_shape = Polygon(xmin,ymin,xmax,ymax)
    # print(geometry_shape)

    print(xmin)
    print(xmax)
    print(ymin)
    print(ymax)
    
    x_res = int((xmax - xmin) / 10)
    y_res = int((ymax - ymin) / 10)
    print(x_res)
    print(y_res)

    band = my_array[:,[2]]

    # print(band)

    with rasterio.open("C:\\TFG_resources\\satelite_images_sigpac\\AAAAAmalaga29900_27_09.tif") as src:
        # src.write(my_array,1)
        src.read(1)
        masked_band, _ = msk.mask(
            src, shapes=shapes_crop, crop=True, nodata=np.nan
        )
        masked_band = masked_band.astype(np.float32)
        band = masked_band

    # new_kwargs = src.copy()
    new_kwargs = ({"driver": "GTiff",
            "dtype": np.float32,
            "height": 2844,
            "width": 2930,
            "transform": from_origin(xmin, ymax, 10, 10),
            "count": 1,
            "crs": "+proj=utm +zone=30 +ellps=EPSG +units=m +no_defs ",
            }
            )

    with rasterio.open("mask_raster1_29900", "w", **new_kwargs) as dst_file:
        dst_file.write(band)
    

# mask_band()


def probando_arr_mask():

    with fiona.open("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp") as shapefile:
        shapes_crop = [feature["geometry"] for feature in shapefile]

    with rasterio.open("C:\TFG_resources\satelite_img\classification_30SUF.tif") as src:
        # src.write(src.read(1))
        # print(src.read(1))
        # print(src.shape)
        print(src)
        masked_band, _ = msk.mask(
            src, shapes=shapes_crop, crop=True, nodata=0
        )
        masked_band = masked_band.astype(np.int32)
        band = masked_band
        print(band)
        print(band.shape)
        print(band.dtype)

    # new_kwargs = src.copy()
    new_kwargs = ({"driver": "GTiff",
            "dtype": np.int32,
            "height": 2844,
            "width": 2930,
            "transform": from_origin(358288.2619795073, 4084001.8209849824, 10, 10),
            "count": 1,
            # "crs": "+proj=utm +zone=30 +ellps=EPSG +units=m +no_defs "
            }
            )

    with rasterio.open("maskSUF_raster_29900.tif", "w", **new_kwargs) as dst_file:
        dst_file.write(band)

# probando_arr_mask()

#!---------------------------------------------
#!---------------------------------------------
#!---------------------------------------------
#!---------------------------------------------
#!---------------------------------------------
#!---------------------------------------------
#!---------------------------------------------
#!---------------------------------------------

def band_codigo():
    data_frame = pd.read_csv("data_inside.csv", sep =",")
    del(data_frame[data_frame.columns[0]])
    df = data_frame.apply(series_split, axis=1)
    df = df.to_frame()
    l = df[0].tolist()
    my_array = np.asarray(l)
    return my_array[:,[2]]

# esto me devuelve un series con mucho NaN y un value bueno

def get_cod_order_shp(path):
    cd_uso = []
    with fiona.open(path) as layer:
        for feature in layer:
            ord_dict = feature['properties']
            for key in ord_dict.values():
                if key in cod_uso: 
                    cd_uso.append(get_id_codigo_uso(key))
    return cd_uso

# cd_uso_list = get_cod_order_shp("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp")
# print(len(cd_uso_list))


# def point_contained(row, point, iterator):
#     print(iterator)
#     if shapely.wkt.loads(row['polygon']).contains(Point(point)):
#         print("premio")
#         print(iterator)
#         return row['id']
#     return 0

def get_band_matrix(path, points_list, arr):
    with fiona.open(path) as layer:
        for feature in layer:
            ord_dict = feature['properties']
            for key in ord_dict.values():
                if key in cod_uso: 
                    cd_uso = get_id_codigo_uso(key)
                    print(cd_uso)
                    #* codigo sigpac para cada iteraccion del shapefile
            geometry = feature["geometry"]['coordinates']
            for g in geometry:
                print(g)
                polygon = Polygon(g)
                print(polygon.bounds)
                print(polygon)
                index_points = [polygon.contains(Point(elem)) for elem in points_list if polygon.contains(Point(elem))]
                print(index_points)
                #* indices de los puntos que estan contenidos en la geometria
                for index in index_points:
                    print(index)
                    arr[points_list[index][0],points_list[index][1]] = cd_uso
                    print(arr[points_list[index][0],points_list[index][1]])
                    #* sustituyo en la posicion de la matriz original el codigo de uso pertinente
    return arr
# crear un vector de tamaño rows*columns , iterar por el vector(mas rapido?) 
# get_band_matrix("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp")



    # valor = arr[not_zero_indices[0][0],not_zero_indices[1][0]]
    # print(arr[1,1992])
    # print(valor)
  
    # banda = band_codigo()
    # my_array = banda.ravel().astype(np.int32)
    # print(my_array)
    # print(len(my_array))
    # arr[not_zero_indices[0][0],not_zero_indices[1][0]] = my_array[0]

    # for i in range(len(my_array)):
    #     arr[not_zero_indices[0][i],not_zero_indices[1][i]] = my_array[i]

    # print(arr)
    # print(arr[1,1992])
    # print(arr.shape)
    # print(arr.dtype)






        
    # points_list = [shp.transform * (not_zero_indices[0][i],not_zero_indices[1][i]) for i in range(len(my_array))] 


    # print(len(points_list))
    # code_list = [id_polygon.apply(point_contained, args=(points_list,i), axis = 1) for i in range(len(points_list))]

#* This will return a series list where all element contains zero values except for one cod_uso
#* Tengo una matriz con la posicion de cada elemento ((((como reemplazo ese elemento por el del cod_uso))))

    # for i in range(len(not_zero_indices[0])):
    #     coord = Point(shp.transform * (not_zero_indices[0][i],not_zero_indices[1][i]))
    #     print(i)
    # for i in range(arr.shape[0]):
    #     for j in range(arr.shape[1]):
    #         print(cord)

    # new_arr_no_0 = arr[np.where(arr!=0)]
    
    # print(new_arr_no_0)
    # print(new_arr_no_0.shape)
    # print(shp.bounds) # (2844, 2930)
    # print(shp.transform * (500, 1000))

#? ------------------


        # array = np.zeros((2930,461))
        # print(array)
        # print(array.dtype)
        # print(array.shape)


        # print(arr)
        # print(type(arr))

        # print(len(arr[1]))
        # print(arr[1])
        # print(len(arr[:,[1]]))
        # print(arr[:,[1]])


    # if mask_geometry:
    #     print(f"Cropping raster {band_name}")
    #     projected_geometry = _project_shape(mask_geometry, dcs=destination_crs)
    #     with rasterio.io.MemoryFile() as memfile:
    #         with memfile.open(**kwargs) as memfile_band:
    #             memfile_band.write(band)
    #             projected_geometry = _convert_3D_2D(projected_geometry)
    #             masked_band, _ = msk.mask(
    #                 memfile_band, shapes=[projected_geometry], crop=True, nodata=np.nan
    #             )
    #             masked_band = masked_band.astype(np.float32)
    #             band = masked_band

    #     new_kwargs = kwargs.copy()
    #     corners = _get_corners_geometry(mask_geometry)
    #     top_left_corner = corners["top_left"]
    #     top_left_corner = (top_left_corner[1], top_left_corner[0])
    #     project = pyproj.Transformer.from_crs(
    #         pyproj.CRS.from_epsg(4326), new_kwargs["crs"], always_xy=True
    #     ).transform
    #     top_left_corner = transform(project, Point(top_left_corner))
    #     new_kwargs["transform"] = rasterio.Affine(
    #         new_kwargs["transform"][0],
    #         0.0,
    #         top_left_corner.x,
    #         0.0,
    #         new_kwargs["transform"][4],
    #         top_left_corner.y,
    #     )
    #     new_kwargs["width"] = band.shape[2]
    #     new_kwargs["height"] = band.shape[1]
    #     kwargs = new_kwargs

    # if path_to_disk is not None:
    #     with rasterio.open(path_to_disk, "w", **kwargs) as dst_file:
    #         dst_file.write(band)
    # return band

list_pixels = [(359128.2619795073, 4063011.8209849824), (359128.2619795073, 4063001.8209849824), (359128.2619795073, 4062991.8209849824), (359128.2619795073, 4062981.8209849824), (359128.2619795073, 4062971.8209849824), (359128.2619795073, 4062961.8209849824), (359128.2619795073, 4062951.8209849824), (359128.2619795073, 4062941.8209849824), (359128.2619795073, 4062931.8209849824), (359128.2619795073, 4062921.8209849824), (359128.2619795073, 4062911.8209849824), (359128.2619795073, 4062901.8209849824), (359128.2619795073, 4062891.8209849824), (359128.2619795073, 4062881.8209849824), (359128.2619795073, 4062871.8209849824), (359128.2619795073, 4062861.8209849824), (359128.2619795073, 4062851.8209849824), (359128.2619795073, 4062841.8209849824), (359128.2619795073, 4062831.8209849824), (359128.2619795073, 4062821.8209849824), (359128.2619795073, 4062811.8209849824), (359128.2619795073, 4062801.8209849824), (359128.2619795073, 4062791.8209849824), (359138.2619795073, 4064811.8209849824), (359138.2619795073, 4064801.8209849824), (359138.2619795073, 4064791.8209849824), (359138.2619795073, 4064781.8209849824), (359138.2619795073, 4064771.8209849824), (359138.2619795073, 4064761.8209849824), (359138.2619795073, 4064751.8209849824), (359138.2619795073, 4064741.8209849824), (359138.2619795073, 4064731.8209849824), (359138.2619795073, 4064721.8209849824), (359138.2619795073, 4064711.8209849824), (359138.2619795073, 4064701.8209849824), (359138.2619795073, 4064691.8209849824), (359138.2619795073, 4064681.8209849824), (359138.2619795073, 4064671.8209849824), (359138.2619795073, 4064661.8209849824), (359138.2619795073, 4064651.8209849824), (359138.2619795073, 4064641.8209849824), (359138.2619795073, 4064631.8209849824), (359138.2619795073, 4064621.8209849824), (359138.2619795073, 4064611.8209849824),(359138.2619795073, 4064601.8209849824), (359138.2619795073, 4064591.8209849824), (359138.2619795073, 4064581.8209849824), (359138.2619795073, 4064571.8209849824), (359138.2619795073,4064561.8209849824),
                (359138.2619795073, 4064551.8209849824), (359138.2619795073, 4064541.8209849824), (359138.2619795073, 4064531.8209849824), (359138.2619795073, 4064521.8209849824), (359138.2619795073, 4064511.8209849824), (359138.2619795073, 4064501.8209849824), (359138.2619795073, 4064491.8209849824), (359138.2619795073, 4064481.8209849824), (359138.2619795073, 4064471.8209849824), (359138.2619795073, 4064461.8209849824), (359138.2619795073, 4064451.8209849824), (359138.2619795073, 4064441.8209849824), (359138.2619795073, 4064431.8209849824), (359138.2619795073, 4064421.8209849824), (359138.2619795073, 4064411.8209849824), (359138.2619795073, 4064401.8209849824), (359138.2619795073, 4064391.8209849824), (359138.2619795073, 4064381.8209849824), (359138.2619795073, 4064371.8209849824), (359138.2619795073, 4064361.8209849824), (359138.2619795073, 4064351.8209849824), (359138.2619795073, 4064341.8209849824), (359138.2619795073, 4064331.8209849824), (359138.2619795073, 4064321.8209849824), (359138.2619795073, 4064311.8209849824), (359138.2619795073, 4064301.8209849824), (359138.2619795073, 4064291.8209849824), (359138.2619795073, 4064281.8209849824), (359138.2619795073, 4064271.8209849824), (359138.2619795073, 4064261.8209849824), (359138.2619795073, 4064251.8209849824), (359138.2619795073, 4064241.8209849824), (359138.2619795073, 4064231.8209849824), (359138.2619795073, 4064221.8209849824), (359138.2619795073, 4064211.8209849824), (359138.2619795073, 4064201.8209849824), (359138.2619795073, 4064191.8209849824), (359138.2619795073, 4064181.8209849824), (359138.2619795073, 4064171.8209849824), (359138.2619795073, 4064161.8209849824), (359138.2619795073, 4064151.8209849824), (359138.2619795073, 4064141.8209849824), (359138.2619795073, 4064131.8209849824), (359138.2619795073, 4064121.8209849824), (359138.2619795073, 4064111.8209849824), (359138.2619795073, 4064101.8209849824), (359138.2619795073, 4064091.8209849824), (359138.2619795073, 4064081.8209849824), (359138.2619795073, 4064071.8209849824), (359138.2619795073, 4064061.8209849824),
                (359138.2619795073, 4064051.8209849824), (359138.2619795073, 4064041.8209849824), (359138.2619795073, 4064031.8209849824), (359138.2619795073, 4064021.8209849824), (359138.2619795073, 4064011.8209849824), (359138.2619795073, 4064001.8209849824), (359138.2619795073, 4063991.8209849824), (359138.2619795073, 4063981.8209849824), (359138.2619795073, 4063971.8209849824), (359138.2619795073, 4063961.8209849824), (359138.2619795073, 4063951.8209849824), (359138.2619795073, 4063941.8209849824), (359138.2619795073, 4063931.8209849824), (359138.2619795073, 4063921.8209849824), (359138.2619795073, 4063911.8209849824), (359138.2619795073, 4063901.8209849824), (359138.2619795073, 4063891.8209849824), (359138.2619795073, 4063881.8209849824), (359138.2619795073, 4063871.8209849824), (359138.2619795073, 4063861.8209849824), (359138.2619795073, 4063851.8209849824), (359138.2619795073, 4063841.8209849824), (359138.2619795073, 4063831.8209849824), (359138.2619795073, 4063821.8209849824), (359138.2619795073, 4063811.8209849824), (359138.2619795073, 4063801.8209849824), (359138.2619795073, 4063791.8209849824), (359138.2619795073, 4063781.8209849824), (359138.2619795073, 4063771.8209849824), (359138.2619795073, 4063761.8209849824), (359138.2619795073, 4063751.8209849824), (359138.2619795073, 4063741.8209849824), (359138.2619795073, 4063731.8209849824), (359138.2619795073, 4063721.8209849824), (359138.2619795073, 4063711.8209849824), (359138.2619795073, 4063701.8209849824), (359138.2619795073, 4063691.8209849824), (359138.2619795073, 4063681.8209849824), (359138.2619795073, 4063671.8209849824), (359138.2619795073, 4063661.8209849824), (359138.2619795073, 4063651.8209849824), (359138.2619795073, 4063641.8209849824), (359138.2619795073, 4063631.8209849824), (359138.2619795073, 4063621.8209849824), (359138.2619795073, 4063611.8209849824), (359138.2619795073, 4063601.8209849824), (359138.2619795073, 4063591.8209849824), (359138.2619795073, 4063581.8209849824), (359138.2619795073, 4063571.8209849824), (359138.2619795073, 4063561.8209849824),
                (359138.2619795073, 4063551.8209849824), (359138.2619795073, 4063541.8209849824), (359138.2619795073, 4063531.8209849824), (359138.2619795073, 4063521.8209849824), (359138.2619795073, 4063511.8209849824), (359138.2619795073, 4063501.8209849824), (359138.2619795073, 4063491.8209849824), (359138.2619795073, 4063481.8209849824), (359138.2619795073, 4063471.8209849824), (359138.2619795073, 4063461.8209849824), (359138.2619795073, 4063451.8209849824), (359138.2619795073, 4063441.8209849824), (359138.2619795073, 4063431.8209849824), (359138.2619795073, 4063421.8209849824), (359138.2619795073, 4063411.8209849824), (359138.2619795073, 4063401.8209849824), (359138.2619795073, 4063391.8209849824), (359138.2619795073, 4063381.8209849824), (359138.2619795073, 4063371.8209849824), (359138.2619795073, 4063361.8209849824), (359138.2619795073, 4063351.8209849824), (359138.2619795073, 4063341.8209849824), (359138.2619795073, 4063331.8209849824), (359138.2619795073, 4063321.8209849824), (359138.2619795073, 4063311.8209849824), (359138.2619795073, 4063301.8209849824), (359138.2619795073, 4063291.8209849824), (359138.2619795073, 4063281.8209849824), (359138.2619795073, 4063271.8209849824), (359138.2619795073, 4063261.8209849824), (359138.2619795073, 4063251.8209849824), (359138.2619795073, 4063241.8209849824), (359138.2619795073, 4063231.8209849824), (359138.2619795073, 4063221.8209849824), (359138.2619795073, 4063211.8209849824), (359138.2619795073, 4063201.8209849824), (359138.2619795073, 4063191.8209849824), (359138.2619795073, 4063181.8209849824), (359138.2619795073, 4063171.8209849824), (359138.2619795073, 4063161.8209849824), (359138.2619795073, 4063151.8209849824), (359138.2619795073, 4063141.8209849824), (359138.2619795073, 4063131.8209849824), (359138.2619795073, 4063121.8209849824), (359138.2619795073, 4063111.8209849824), (359138.2619795073, 4063101.8209849824), (359138.2619795073, 4063091.8209849824), (359138.2619795073, 4063081.8209849824), (359138.2619795073, 4063071.8209849824), (359138.2619795073, 4063061.8209849824),
                (359138.2619795073, 4063051.8209849824), (359138.2619795073, 4063041.8209849824), (359138.2619795073, 4063031.8209849824), (359138.2619795073, 4063021.8209849824), (359138.2619795073, 4063011.8209849824), (359138.2619795073, 4063001.8209849824), (359138.2619795073, 4062991.8209849824), (359138.2619795073, 4062981.8209849824), (359138.2619795073, 4062971.8209849824), (359138.2619795073, 4062961.8209849824), (359138.2619795073, 4062951.8209849824), (359138.2619795073, 4062941.8209849824), (359138.2619795073, 4062931.8209849824), (359138.2619795073, 4062921.8209849824), (359138.2619795073, 4062911.8209849824), (359138.2619795073, 4062901.8209849824), (359138.2619795073, 4062891.8209849824), (359138.2619795073, 4062881.8209849824), (359138.2619795073, 4062871.8209849824), (359138.2619795073, 4062861.8209849824), (359138.2619795073, 4062851.8209849824), (359138.2619795073, 4062841.8209849824), (359138.2619795073, 4062831.8209849824), (359138.2619795073, 4062821.8209849824), (359138.2619795073, 4062811.8209849824), (359138.2619795073, 4062801.8209849824), (359138.2619795073, 4062791.8209849824), (359138.2619795073, 4062781.8209849824), (359148.2619795073, 4064811.8209849824), (359148.2619795073, 4064801.8209849824), (359148.2619795073, 4064791.8209849824), (359148.2619795073, 4064781.8209849824), (359148.2619795073, 4064771.8209849824), (359148.2619795073, 4064761.8209849824), (359148.2619795073, 4064751.8209849824), (359148.2619795073, 4064741.8209849824), (359148.2619795073, 4064731.8209849824), (359148.2619795073, 4064721.8209849824), (359148.2619795073, 4064711.8209849824), (359148.2619795073, 4064701.8209849824), (359148.2619795073, 4064691.8209849824), (359148.2619795073, 4064681.8209849824), (359148.2619795073, 4064671.8209849824), (359148.2619795073, 4064661.8209849824), (359148.2619795073, 4064651.8209849824), (359148.2619795073, 4064641.8209849824), (359148.2619795073, 4064631.8209849824), (359148.2619795073, 4064621.8209849824), (359148.2619795073, 4064611.8209849824), (359148.2619795073, 4064601.8209849824), 
                (359148.2619795073, 4064591.8209849824), (359148.2619795073, 4064581.8209849824), (359148.2619795073, 4064571.8209849824), (359148.2619795073, 4064561.8209849824), (359148.2619795073, 4064551.8209849824), (359148.2619795073, 4064541.8209849824), (359148.2619795073, 4064531.8209849824), (359148.2619795073, 4064521.8209849824), (359148.2619795073, 4064511.8209849824), (359148.2619795073, 4064501.8209849824), (359148.2619795073, 4064491.8209849824), (359148.2619795073, 4064481.8209849824), (359148.2619795073, 4064471.8209849824), (359148.2619795073, 4064461.8209849824), (359148.2619795073, 4064451.8209849824), (359148.2619795073, 4064441.8209849824), (359148.2619795073, 4064431.8209849824), (359148.2619795073, 4064421.8209849824), (359148.2619795073, 4064411.8209849824), (359148.2619795073, 4064401.8209849824), (359148.2619795073, 4064391.8209849824), (359148.2619795073, 4064381.8209849824), (359148.2619795073, 4064371.8209849824), (359148.2619795073, 4064361.8209849824), (359148.2619795073, 4064351.8209849824), (359148.2619795073, 4064341.8209849824), (359148.2619795073, 4064331.8209849824), (359148.2619795073, 4064321.8209849824), (359148.2619795073, 4064311.8209849824), (359148.2619795073, 4064301.8209849824), (359148.2619795073, 4064291.8209849824), (359148.2619795073, 4064281.8209849824), (359148.2619795073, 4064271.8209849824), (359148.2619795073, 4064261.8209849824), (359148.2619795073, 4064251.8209849824), (359148.2619795073, 4064241.8209849824), (359148.2619795073, 4064231.8209849824), (359148.2619795073, 4064221.8209849824), (359148.2619795073, 4064211.8209849824), (359148.2619795073, 4064201.8209849824), (359148.2619795073, 4064191.8209849824), (359148.2619795073, 4064181.8209849824), (359148.2619795073, 4064171.8209849824), (359148.2619795073, 4064161.8209849824), (359148.2619795073, 4064151.8209849824), (359148.2619795073, 4064141.8209849824), (359148.2619795073, 4064131.8209849824), (359148.2619795073, 4064121.8209849824), (359148.2619795073, 4064111.8209849824), (359148.2619795073, 4064101.8209849824), 
                (359148.2619795073, 4064091.8209849824), (359148.2619795073, 4064081.8209849824), (359148.2619795073, 4064071.8209849824), (359148.2619795073, 4064061.8209849824), (359148.2619795073, 4064051.8209849824), (359148.2619795073, 4064041.8209849824), (359148.2619795073, 4064031.8209849824), (359148.2619795073, 4064021.8209849824), (359148.2619795073, 4064011.8209849824), (359148.2619795073, 4064001.8209849824), (359148.2619795073, 4063991.8209849824), (359148.2619795073, 4063981.8209849824), (359148.2619795073, 4063971.8209849824), (359148.2619795073, 4063961.8209849824), (359148.2619795073, 4063951.8209849824), (359148.2619795073, 4063941.8209849824), (359148.2619795073, 4063931.8209849824), (359148.2619795073, 4063921.8209849824), (359148.2619795073, 4063911.8209849824), (359148.2619795073, 4063901.8209849824), (359148.2619795073, 4063891.8209849824), (359148.2619795073, 4063881.8209849824)]

list_pixels2 = [(359808.2619795073, 4064861.8209849824), (359808.2619795073, 4064851.8209849824), (359808.2619795073, 4064841.8209849824), (359808.2619795073, 4064831.8209849824), (359808.2619795073, 4064821.8209849824), (359808.2619795073, 4064811.8209849824), (359808.2619795073, 4064801.8209849824), (359808.2619795073, 4064791.8209849824), (359808.2619795073, 4064781.8209849824), (359808.2619795073, 4064771.8209849824), (359808.2619795073, 4064761.8209849824), (359808.2619795073, 4064751.8209849824), (359808.2619795073, 4064741.8209849824), (359808.2619795073, 4064731.8209849824), (359808.2619795073, 4064721.8209849824), (359808.2619795073, 4064711.8209849824), (359808.2619795073, 4064701.8209849824), (359808.2619795073, 4064691.8209849824), (359808.2619795073, 4064681.8209849824), (359808.2619795073, 4064671.8209849824), (359808.2619795073, 4064661.8209849824), (359808.2619795073, 4064651.8209849824), (359808.2619795073, 4064641.8209849824), (359808.2619795073, 4064631.8209849824), (359808.2619795073, 4064621.8209849824), (359808.2619795073, 4064611.8209849824), (359808.2619795073, 4064601.8209849824), (359808.2619795073, 4064591.8209849824), (359808.2619795073, 4064581.8209849824), (359808.2619795073, 4064571.8209849824), (359808.2619795073, 4064561.8209849824), (359808.2619795073, 4064551.8209849824), (359808.2619795073, 4064541.8209849824), (359808.2619795073, 4064531.8209849824), (359808.2619795073, 4064521.8209849824), (359808.2619795073, 4064511.8209849824), (359808.2619795073, 4064501.8209849824), (359808.2619795073, 4064491.8209849824), (359808.2619795073, 4064481.8209849824), (359808.2619795073, 4064471.8209849824), (359808.2619795073, 4064461.8209849824), (359808.2619795073, 4064451.8209849824), (359808.2619795073, 4064441.8209849824), (359808.2619795073, 4064431.8209849824), (359808.2619795073, 4064421.8209849824), (359808.2619795073, 4064411.8209849824), (359808.2619795073, 4064401.8209849824), 
(359808.2619795073, 4064391.8209849824), (359808.2619795073, 4064381.8209849824), (359808.2619795073, 4064371.8209849824), (359808.2619795073, 4064361.8209849824), (359808.2619795073, 4064351.8209849824), (359808.2619795073, 4064341.8209849824), (359808.2619795073, 4064331.8209849824), (359808.2619795073, 4064321.8209849824), (359808.2619795073, 4064311.8209849824), (359808.2619795073, 4064301.8209849824), (359808.2619795073, 4064291.8209849824), (359808.2619795073, 4064281.8209849824), (359808.2619795073, 4064271.8209849824), (359808.2619795073, 4064261.8209849824), (359808.2619795073, 4064251.8209849824), (359808.2619795073, 4064241.8209849824), (359808.2619795073, 4064231.8209849824), (359808.2619795073, 4064221.8209849824), (359808.2619795073, 4064211.8209849824), (359808.2619795073, 4064201.8209849824), (359808.2619795073, 4064191.8209849824), (359808.2619795073, 4064181.8209849824), (359808.2619795073, 4064171.8209849824), (359808.2619795073, 4064161.8209849824), (359808.2619795073, 4064151.8209849824), (359808.2619795073, 4064141.8209849824), (359808.2619795073, 4064131.8209849824), (359808.2619795073, 4064121.8209849824), (359808.2619795073, 4064111.8209849824), (359808.2619795073, 4064101.8209849824), (359808.2619795073, 4064091.8209849824), (359808.2619795073, 4064081.8209849824), (359808.2619795073, 4064071.8209849824), (359808.2619795073, 4064061.8209849824), (359808.2619795073, 4064051.8209849824), (359808.2619795073, 4064041.8209849824), (359808.2619795073, 4064031.8209849824), (359808.2619795073, 4064021.8209849824), (359808.2619795073, 4064011.8209849824), (359808.2619795073, 4064001.8209849824), (359808.2619795073, 4063991.8209849824), (359808.2619795073, 4063981.8209849824), (359808.2619795073, 4063971.8209849824), (359808.2619795073, 4063961.8209849824), (359808.2619795073, 4063951.8209849824), (359808.2619795073, 4063941.8209849824), (359808.2619795073, 4063931.8209849824), (359808.2619795073, 4063921.8209849824), (359808.2619795073, 4063911.8209849824), (359808.2619795073, 4063901.8209849824), (359808.2619795073, 4063891.8209849824), (359808.2619795073, 4063881.8209849824), (359808.2619795073, 4063871.8209849824), (359808.2619795073, 4063861.8209849824), (359808.2619795073, 4063851.8209849824), (359808.2619795073, 4063841.8209849824), (359808.2619795073, 4063831.8209849824), (359808.2619795073, 4063821.8209849824), (359808.2619795073, 4063811.8209849824), (359808.2619795073, 4063801.8209849824), (359808.2619795073, 4063791.8209849824), (359808.2619795073, 4063781.8209849824), (359808.2619795073, 4063771.8209849824), (359808.2619795073, 4063761.8209849824), (359808.2619795073, 4063751.8209849824), (359808.2619795073, 4063741.8209849824), (359808.2619795073, 4063731.8209849824), (359808.2619795073, 4063721.8209849824), (359808.2619795073, 4063711.8209849824), (359808.2619795073, 4063701.8209849824), (359808.2619795073, 4063691.8209849824), (359808.2619795073, 4063681.8209849824), (359808.2619795073, 4063671.8209849824), (359808.2619795073, 4063661.8209849824), (359808.2619795073, 4063651.8209849824), (359808.2619795073, 4063641.8209849824), (359808.2619795073, 4063631.8209849824), (359808.2619795073, 4063621.8209849824), (359808.2619795073, 4063611.8209849824), (359808.2619795073, 4063601.8209849824), (359808.2619795073, 
4063591.8209849824), (359808.2619795073, 4063581.8209849824), (359808.2619795073, 4063571.8209849824), (359808.2619795073, 4063561.8209849824), (359808.2619795073, 4063551.8209849824), (359808.2619795073, 4063541.8209849824), (359808.2619795073, 4063531.8209849824), (359808.2619795073, 4063521.8209849824), (359808.2619795073, 4063511.8209849824), (359808.2619795073, 4063501.8209849824), (359808.2619795073, 4063491.8209849824), (359808.2619795073, 4063481.8209849824), (359808.2619795073, 4063471.8209849824), (359808.2619795073, 4063461.8209849824), (359808.2619795073, 4063451.8209849824), (359808.2619795073, 4063441.8209849824), (359808.2619795073, 4063431.8209849824), (359808.2619795073, 4063421.8209849824), (359808.2619795073, 4063411.8209849824), (359808.2619795073, 4063401.8209849824), (359808.2619795073, 4063391.8209849824), (359808.2619795073, 4063381.8209849824), (359808.2619795073, 4063371.8209849824), (359808.2619795073, 4063361.8209849824), (359808.2619795073, 4063351.8209849824), (359808.2619795073, 4063341.8209849824), (359808.2619795073, 4063331.8209849824), (359808.2619795073, 4063321.8209849824), (359808.2619795073, 4063311.8209849824), (359808.2619795073, 4063301.8209849824), (359808.2619795073, 4063291.8209849824), (359808.2619795073, 4063281.8209849824), (359808.2619795073, 4063271.8209849824), (359808.2619795073, 4063261.8209849824), (359808.2619795073, 4063251.8209849824), (359808.2619795073, 4063241.8209849824), (359808.2619795073, 4063231.8209849824), (359808.2619795073, 4063221.8209849824), (359808.2619795073, 4063211.8209849824), (359808.2619795073, 4063201.8209849824), (359808.2619795073, 4063191.8209849824), (359808.2619795073, 4063181.8209849824), (359808.2619795073, 4063171.8209849824), (359808.2619795073, 4063161.8209849824), (359808.2619795073, 4063151.8209849824), (359808.2619795073, 4063141.8209849824), (359808.2619795073, 4063131.8209849824), (359808.2619795073, 4063121.8209849824), (359808.2619795073, 4063111.8209849824), (359808.2619795073, 4063101.8209849824), (359808.2619795073, 4063091.8209849824), (359808.2619795073, 4063081.8209849824), (359808.2619795073, 4063071.8209849824), (359808.2619795073, 4063061.8209849824), (359808.2619795073, 4063051.8209849824), (359808.2619795073, 4063041.8209849824), (359808.2619795073, 4063031.8209849824), (359808.2619795073, 4063021.8209849824), (359808.2619795073, 4063011.8209849824), (359808.2619795073, 4063001.8209849824), (359808.2619795073, 4062991.8209849824), (359808.2619795073, 4062981.8209849824), (359808.2619795073, 4062971.8209849824), (359808.2619795073, 4062961.8209849824), (359808.2619795073, 4062951.8209849824), (359808.2619795073, 4062941.8209849824), (359808.2619795073, 4062931.8209849824), (359808.2619795073, 4062921.8209849824), (359808.2619795073, 4062911.8209849824), (359808.2619795073, 4062901.8209849824), 
(359808.2619795073, 4062891.8209849824), (359808.2619795073, 4062881.8209849824), (359808.2619795073, 4062871.8209849824), (359808.2619795073, 4062861.8209849824), (359808.2619795073, 4062851.8209849824), (359808.2619795073, 4062841.8209849824), (359808.2619795073, 4062831.8209849824), (359808.2619795073, 4062821.8209849824), (359808.2619795073, 4062811.8209849824), (359808.2619795073, 4062801.8209849824), (359808.2619795073, 4062791.8209849824), (359808.2619795073, 4062781.8209849824), (359808.2619795073, 4062771.8209849824), (359808.2619795073, 4062761.8209849824), (359808.2619795073, 4062751.8209849824), (359808.2619795073, 4062741.8209849824), (359808.2619795073, 4062731.8209849824), (359808.2619795073, 4062721.8209849824), (359808.2619795073, 4062711.8209849824), (359808.2619795073, 4062701.8209849824), (359808.2619795073, 4062691.8209849824), (359808.2619795073, 4062681.8209849824), (359808.2619795073, 4062671.8209849824), (359808.2619795073, 4062661.8209849824), (359808.2619795073, 4062651.8209849824), (359808.2619795073, 4062641.8209849824), (359808.2619795073, 4062631.8209849824), (359808.2619795073, 4062621.8209849824), (359808.2619795073, 4062611.8209849824), (359808.2619795073, 4062601.8209849824), (359808.2619795073, 4062591.8209849824), (359808.2619795073, 4062581.8209849824), (359808.2619795073, 4062571.8209849824), (359808.2619795073, 4062561.8209849824), (359808.2619795073, 4062551.8209849824), (359808.2619795073, 4062541.8209849824), (359808.2619795073, 4062531.8209849824), (359808.2619795073, 4062521.8209849824), (359808.2619795073, 4062511.8209849824), (359808.2619795073, 4062501.8209849824), (359808.2619795073, 4062491.8209849824), (359808.2619795073, 4062481.8209849824), (359808.2619795073, 4062471.8209849824), (359808.2619795073, 4062461.8209849824), (359808.2619795073, 4062451.8209849824), (359808.2619795073, 4062441.8209849824), (359808.2619795073, 4062431.8209849824), (359808.2619795073, 4062421.8209849824), (359808.2619795073, 4062411.8209849824), (359808.2619795073, 4062401.8209849824), (359808.2619795073, 4062391.8209849824), (359808.2619795073, 4062381.8209849824), (359808.2619795073, 4062371.8209849824), (359808.2619795073, 4062361.8209849824), (359808.2619795073, 4062351.8209849824), (359808.2619795073, 4062341.8209849824), (359808.2619795073, 4062331.8209849824), (359808.2619795073, 4062321.8209849824), (359808.2619795073, 4062311.8209849824), (359808.2619795073, 4062301.8209849824), (359808.2619795073, 4062291.8209849824), (359808.2619795073, 4062281.8209849824), (359808.2619795073, 4062271.8209849824), (359808.2619795073, 4062261.8209849824), (359808.2619795073, 4062251.8209849824), (359808.2619795073, 4062241.8209849824), (359808.2619795073, 4062231.8209849824), (359808.2619795073, 4062221.8209849824), (359808.2619795073, 4062211.8209849824), (359808.2619795073, 4062201.8209849824), (359808.2619795073, 4062191.8209849824), (359808.2619795073, 4062181.8209849824), (359808.2619795073, 4062171.8209849824), (359808.2619795073, 4062161.8209849824), (359808.2619795073, 4062151.8209849824), (359808.2619795073, 4062141.8209849824), (359808.2619795073, 4062131.8209849824), (359808.2619795073, 4062121.8209849824), (359808.2619795073, 4062111.8209849824), (359808.2619795073, 4062101.8209849824), (359808.2619795073, 
4062091.8209849824), (359808.2619795073, 4062081.8209849824), (359808.2619795073, 4062071.8209849824), (359808.2619795073, 4062061.8209849824), (359818.2619795073, 4065291.8209849824), (359818.2619795073, 4065281.8209849824), (359818.2619795073, 4065271.8209849824), (359818.2619795073, 4065261.8209849824), (359818.2619795073, 4065251.8209849824), (359818.2619795073, 4065241.8209849824), (359818.2619795073, 4065231.8209849824), (359818.2619795073, 4065221.8209849824), (359818.2619795073, 4065211.8209849824), (359818.2619795073, 4065201.8209849824), (359818.2619795073, 4065191.8209849824), (359818.2619795073, 4065181.8209849824), (359818.2619795073, 4065171.8209849824), (359818.2619795073, 4065161.8209849824), (359818.2619795073, 4065151.8209849824), (359818.2619795073, 4065141.8209849824), (359818.2619795073, 4065131.8209849824), (359818.2619795073, 4065121.8209849824), (359818.2619795073, 4065111.8209849824), (359818.2619795073, 4065101.8209849824), (359818.2619795073, 4065091.8209849824), (359818.2619795073, 4065081.8209849824), (359818.2619795073, 4065071.8209849824), (359818.2619795073, 4065061.8209849824), (359818.2619795073, 4065051.8209849824), (359818.2619795073, 4065041.8209849824), (359818.2619795073, 4065031.8209849824), (359818.2619795073, 4065021.8209849824), (359818.2619795073, 4065011.8209849824), (359818.2619795073, 4065001.8209849824), (359818.2619795073, 4064991.8209849824), (359818.2619795073, 4064981.8209849824), (359818.2619795073, 4064971.8209849824), (359818.2619795073, 4064961.8209849824), (359818.2619795073, 4064951.8209849824), (359818.2619795073, 4064941.8209849824), (359818.2619795073, 4064931.8209849824), (359818.2619795073, 4064921.8209849824), (359818.2619795073, 4064911.8209849824), (359818.2619795073, 4064901.8209849824), (359818.2619795073, 4064891.8209849824), (359818.2619795073, 4064881.8209849824), (359818.2619795073, 4064871.8209849824), (359818.2619795073, 4064861.8209849824), (359818.2619795073, 4064851.8209849824), (359818.2619795073, 4064841.8209849824), (359818.2619795073, 4064831.8209849824), (359818.2619795073, 4064821.8209849824), (359818.2619795073, 4064811.8209849824), (359818.2619795073, 4064801.8209849824), (359818.2619795073, 4064791.8209849824), (359818.2619795073, 4064781.8209849824), (359818.2619795073, 4064771.8209849824), (359818.2619795073, 4064761.8209849824), (359818.2619795073, 4064751.8209849824), (359818.2619795073, 4064741.8209849824), (359818.2619795073, 4064731.8209849824), (359818.2619795073, 4064721.8209849824), (359818.2619795073, 4064711.8209849824), (359818.2619795073, 4064701.8209849824), (359818.2619795073, 4064691.8209849824), (359818.2619795073, 4064681.8209849824), (359818.2619795073, 4064671.8209849824), (359818.2619795073, 4064661.8209849824), (359818.2619795073, 4064651.8209849824), (359818.2619795073, 4064641.8209849824), 
(359818.2619795073, 4064631.8209849824), (359818.2619795073, 4064621.8209849824), (359818.2619795073, 4064611.8209849824), (359818.2619795073, 4064601.8209849824), (359818.2619795073, 4064591.8209849824), (359818.2619795073, 4064581.8209849824), (359818.2619795073, 4064571.8209849824), (359818.2619795073, 4064561.8209849824), (359818.2619795073, 4064551.8209849824), (359818.2619795073, 4064541.8209849824), (359818.2619795073, 4064531.8209849824), (359818.2619795073, 4064521.8209849824), (359818.2619795073, 4064511.8209849824), (359818.2619795073, 4064501.8209849824), (359818.2619795073, 4064491.8209849824), (359818.2619795073, 4064481.8209849824), (359818.2619795073, 4064471.8209849824), (359818.2619795073, 4064461.8209849824), (359818.2619795073, 4064451.8209849824), (359818.2619795073, 4064441.8209849824), (359818.2619795073, 4064431.8209849824), (359818.2619795073, 4064421.8209849824), (359818.2619795073, 4064411.8209849824), (359818.2619795073, 4064401.8209849824), (359818.2619795073, 4064391.8209849824), (359818.2619795073, 4064381.8209849824), (359818.2619795073, 4064371.8209849824), (359818.2619795073, 4064361.8209849824), (359818.2619795073, 4064351.8209849824), (359818.2619795073, 4064341.8209849824), (359818.2619795073, 4064331.8209849824), (359818.2619795073, 4064321.8209849824), (359818.2619795073, 4064311.8209849824), (359818.2619795073, 4064301.8209849824), (359818.2619795073, 4064291.8209849824), (359818.2619795073, 4064281.8209849824), (359818.2619795073, 4064271.8209849824), (359818.2619795073, 4064261.8209849824), (359818.2619795073, 4064251.8209849824), (359818.2619795073, 4064241.8209849824), (359818.2619795073, 4064231.8209849824), (359818.2619795073, 4064221.8209849824), (359818.2619795073, 4064211.8209849824), (359818.2619795073, 4064201.8209849824), (359818.2619795073, 4064191.8209849824), (359818.2619795073, 4064181.8209849824), (359818.2619795073, 4064171.8209849824), (359818.2619795073, 4064161.8209849824), (359818.2619795073, 4064151.8209849824), (359818.2619795073, 4064141.8209849824), (359818.2619795073, 4064131.8209849824), (359818.2619795073, 4064121.8209849824), (359818.2619795073, 4064111.8209849824), (359818.2619795073, 4064101.8209849824), (359818.2619795073, 4064091.8209849824), (359818.2619795073, 4064081.8209849824), (359818.2619795073, 4064071.8209849824), (359818.2619795073, 4064061.8209849824), (359818.2619795073, 4064051.8209849824), (359818.2619795073, 4064041.8209849824), (359818.2619795073, 4064031.8209849824), (359818.2619795073, 4064021.8209849824), (359818.2619795073, 4064011.8209849824), (359818.2619795073, 4064001.8209849824), (359818.2619795073, 4063991.8209849824), (359818.2619795073, 4063981.8209849824), (359818.2619795073, 4063971.8209849824), (359818.2619795073, 4063961.8209849824), (359818.2619795073, 4063951.8209849824), (359818.2619795073, 4063941.8209849824), (359818.2619795073, 4063931.8209849824), (359818.2619795073, 4063921.8209849824), (359818.2619795073, 4063911.8209849824), (359818.2619795073, 4063901.8209849824), (359818.2619795073, 4063891.8209849824), (359818.2619795073, 4063881.8209849824), (359818.2619795073, 4063871.8209849824), (359818.2619795073, 4063861.8209849824), (359818.2619795073, 4063851.8209849824), (359818.2619795073, 4063841.8209849824), (359818.2619795073, 
4063831.8209849824), (359818.2619795073, 4063821.8209849824), (359818.2619795073, 4063811.8209849824), (359818.2619795073, 4063801.8209849824), (359818.2619795073, 4063791.8209849824), (359818.2619795073, 4063781.8209849824), (359818.2619795073, 4063771.8209849824), (359818.2619795073, 4063761.8209849824), (359818.2619795073, 4063751.8209849824), (359818.2619795073, 4063741.8209849824), (359818.2619795073, 4063731.8209849824), (359818.2619795073, 4063721.8209849824), (359818.2619795073, 4063711.8209849824), (359818.2619795073, 4063701.8209849824), (359818.2619795073, 4063691.8209849824), (359818.2619795073, 4063681.8209849824), (359818.2619795073, 4063671.8209849824), (359818.2619795073, 4063661.8209849824), (359818.2619795073, 4063651.8209849824), (359818.2619795073, 4063641.8209849824), (359818.2619795073, 4063631.8209849824), (359818.2619795073, 4063621.8209849824), (359818.2619795073, 4063611.8209849824), (359818.2619795073, 4063601.8209849824), (359818.2619795073, 4063591.8209849824), (359818.2619795073, 4063581.8209849824), (359818.2619795073, 4063571.8209849824), (359818.2619795073, 4063561.8209849824), (359818.2619795073, 4063551.8209849824), (359818.2619795073, 4063541.8209849824), (359818.2619795073, 4063531.8209849824), (359818.2619795073, 4063521.8209849824), (359818.2619795073, 4063511.8209849824), (359818.2619795073, 4063501.8209849824), (359818.2619795073, 4063491.8209849824), (359818.2619795073, 4063481.8209849824), (359818.2619795073, 4063471.8209849824), (359818.2619795073, 4063461.8209849824), (359818.2619795073, 4063451.8209849824), (359818.2619795073, 4063441.8209849824), (359818.2619795073, 4063431.8209849824), (359818.2619795073, 4063421.8209849824), (359818.2619795073, 4063411.8209849824), (359818.2619795073, 4063401.8209849824), (359818.2619795073, 4063391.8209849824), (359818.2619795073, 4063381.8209849824), (359818.2619795073, 4063371.8209849824), (359818.2619795073, 4063361.8209849824), (359818.2619795073, 4063351.8209849824), (359818.2619795073, 4063341.8209849824), (359818.2619795073, 4063331.8209849824), (359818.2619795073, 4063321.8209849824), (359818.2619795073, 4063311.8209849824), (359818.2619795073, 4063301.8209849824), (359818.2619795073, 4063291.8209849824), (359818.2619795073, 4063281.8209849824), (359818.2619795073, 4063271.8209849824), (359818.2619795073, 4063261.8209849824), (359818.2619795073, 4063251.8209849824), (359818.2619795073, 4063241.8209849824), (359818.2619795073, 4063231.8209849824), (359818.2619795073, 4063221.8209849824), (359818.2619795073, 4063211.8209849824), (359818.2619795073, 4063201.8209849824), (359818.2619795073, 4063191.8209849824), (359818.2619795073, 4063181.8209849824), (359818.2619795073, 4063171.8209849824), (359818.2619795073, 4063161.8209849824), (359818.2619795073, 4063151.8209849824), (359818.2619795073, 4063141.8209849824), 
(359818.2619795073, 4063131.8209849824), (359818.2619795073, 4063121.8209849824), (359818.2619795073, 4063111.8209849824), (359818.2619795073, 4063101.8209849824), (359818.2619795073, 4063091.8209849824), (359818.2619795073, 4063081.8209849824), (359818.2619795073, 4063071.8209849824), (359818.2619795073, 4063061.8209849824), (359818.2619795073, 4063051.8209849824), (359818.2619795073, 4063041.8209849824), (359818.2619795073, 4063031.8209849824), (359818.2619795073, 4063021.8209849824), (359818.2619795073, 4063011.8209849824), (359818.2619795073, 4063001.8209849824), (359818.2619795073, 4062991.8209849824), (359818.2619795073, 4062981.8209849824), (359818.2619795073, 4062971.8209849824), (359818.2619795073, 4062961.8209849824), (359818.2619795073, 4062951.8209849824), (359818.2619795073, 4062941.8209849824), (359818.2619795073, 4062931.8209849824), (359818.2619795073, 4062921.8209849824), (359818.2619795073, 4062911.8209849824), (359818.2619795073, 4062901.8209849824), (359818.2619795073, 4062891.8209849824), (359818.2619795073, 4062881.8209849824), (359818.2619795073, 4062871.8209849824), (359818.2619795073, 4062861.8209849824), (359818.2619795073, 4062851.8209849824), (359818.2619795073, 4062841.8209849824), (359818.2619795073, 4062831.8209849824), (359818.2619795073, 4062821.8209849824), (359818.2619795073, 4062811.8209849824), (359818.2619795073, 4062801.8209849824), (359818.2619795073, 4062791.8209849824), (359818.2619795073, 4062781.8209849824), (359818.2619795073, 4062771.8209849824), (359818.2619795073, 4062761.8209849824), (359818.2619795073, 4062751.8209849824), (359818.2619795073, 4062741.8209849824), (359818.2619795073, 4062731.8209849824), (359818.2619795073, 4062721.8209849824), (359818.2619795073, 4062711.8209849824), (359818.2619795073, 4062701.8209849824), (359818.2619795073, 4062691.8209849824), (359818.2619795073, 4062681.8209849824), (359818.2619795073, 4062671.8209849824), (359818.2619795073, 4062661.8209849824), (359818.2619795073, 4062651.8209849824), (359818.2619795073, 4062641.8209849824), (359818.2619795073, 4062631.8209849824), (359818.2619795073, 4062621.8209849824), (359818.2619795073, 4062611.8209849824), (359818.2619795073, 4062601.8209849824), (359818.2619795073, 4062591.8209849824), (359818.2619795073, 4062581.8209849824), (359818.2619795073, 4062571.8209849824), (359818.2619795073, 4062561.8209849824), (359818.2619795073, 4062551.8209849824), (359818.2619795073, 4062541.8209849824), (359818.2619795073, 4062531.8209849824), (359818.2619795073, 4062521.8209849824), (359818.2619795073, 4062511.8209849824), (359818.2619795073, 4062501.8209849824), (359818.2619795073, 4062491.8209849824), (359818.2619795073, 4062481.8209849824), (359818.2619795073, 4062471.8209849824), (359818.2619795073, 4062461.8209849824), (359818.2619795073, 4062451.8209849824), (359818.2619795073, 4062441.8209849824), (359818.2619795073, 4062431.8209849824), (359818.2619795073, 4062421.8209849824), (359818.2619795073, 4062411.8209849824), (359818.2619795073, 4062401.8209849824), (359818.2619795073, 4062391.8209849824), (359818.2619795073, 4062381.8209849824), (359818.2619795073, 4062371.8209849824), (359818.2619795073, 4062361.8209849824), (359818.2619795073, 4062351.8209849824), (359818.2619795073, 4062341.8209849824), (359818.2619795073, 
4062331.8209849824), (359818.2619795073, 4062321.8209849824), (359818.2619795073, 4062311.8209849824), (359818.2619795073, 4062301.8209849824), (359818.2619795073, 4062291.8209849824), (359818.2619795073, 4062281.8209849824), (359818.2619795073, 4062271.8209849824), (359818.2619795073, 4062261.8209849824), (359818.2619795073, 4062251.8209849824), (359818.2619795073, 4062241.8209849824), (359818.2619795073, 4062231.8209849824), (359818.2619795073, 4062221.8209849824), (359818.2619795073, 4062211.8209849824), (359818.2619795073, 4062201.8209849824), (359818.2619795073, 4062191.8209849824), (359818.2619795073, 4062181.8209849824), (359818.2619795073, 4062171.8209849824), (359818.2619795073, 4062161.8209849824), (359818.2619795073, 4062151.8209849824), (359818.2619795073, 4062141.8209849824), (359818.2619795073, 4062131.8209849824), (359818.2619795073, 4062121.8209849824), (359818.2619795073, 4062111.8209849824), (359818.2619795073, 4062101.8209849824), (359818.2619795073, 4062091.8209849824), (359818.2619795073, 4062081.8209849824), (359818.2619795073, 4062071.8209849824), (359818.2619795073, 4062061.8209849824), (359828.2619795073, 4065311.8209849824), (359828.2619795073, 4065301.8209849824), (359828.2619795073, 4065291.8209849824), (359828.2619795073, 4065281.8209849824), (359828.2619795073, 4065271.8209849824), (359828.2619795073, 4065261.8209849824), (359828.2619795073, 4065251.8209849824), (359828.2619795073, 4065241.8209849824), (359828.2619795073, 4065231.8209849824), (359828.2619795073, 4065221.8209849824), (359828.2619795073, 4065211.8209849824), (359828.2619795073, 4065201.8209849824), (359828.2619795073, 4065191.8209849824), (359828.2619795073, 4065181.8209849824), (359828.2619795073, 4065171.8209849824), (359828.2619795073, 4065161.8209849824), (359828.2619795073, 4065151.8209849824), (359828.2619795073, 4065141.8209849824), (359828.2619795073, 4065131.8209849824), (359828.2619795073, 4065121.8209849824), (359828.2619795073, 4065111.8209849824), (359828.2619795073, 4065101.8209849824), (359828.2619795073, 4065091.8209849824), (359828.2619795073, 4065081.8209849824), (359828.2619795073, 4065071.8209849824), (359828.2619795073, 4065061.8209849824), (359828.2619795073, 4065051.8209849824), (359828.2619795073, 4065041.8209849824), (359828.2619795073, 4065031.8209849824), (359828.2619795073, 4065021.8209849824), (359828.2619795073, 4065011.8209849824), (359828.2619795073, 4065001.8209849824), (359828.2619795073, 4064991.8209849824), (359828.2619795073, 4064981.8209849824), (359828.2619795073, 4064971.8209849824), (359828.2619795073, 4064961.8209849824), (359828.2619795073, 4064951.8209849824), (359828.2619795073, 4064941.8209849824), (359828.2619795073, 4064931.8209849824), (359828.2619795073, 4064921.8209849824), (359828.2619795073, 4064911.8209849824), (359828.2619795073, 4064901.8209849824), 
(359828.2619795073, 4064891.8209849824), (359828.2619795073, 4064881.8209849824), (359828.2619795073, 4064871.8209849824), (359828.2619795073, 4064861.8209849824), (359828.2619795073, 4064851.8209849824), (359828.2619795073, 4064841.8209849824), (359828.2619795073, 4064831.8209849824), (359828.2619795073, 4064821.8209849824), (359828.2619795073, 4064811.8209849824), (359828.2619795073, 4064801.8209849824), (359828.2619795073, 4064791.8209849824), (359828.2619795073, 4064781.8209849824), (359828.2619795073, 4064771.8209849824), (359828.2619795073, 4064761.8209849824), (359828.2619795073, 4064751.8209849824), (359828.2619795073, 4064741.8209849824), (359828.2619795073, 4064731.8209849824), (359828.2619795073, 4064721.8209849824), (359828.2619795073, 4064711.8209849824), (359828.2619795073, 4064701.8209849824), (359828.2619795073, 4064691.8209849824), (359828.2619795073, 4064681.8209849824), (359828.2619795073, 4064671.8209849824), (359828.2619795073, 4064661.8209849824), (359828.2619795073, 4064651.8209849824), (359828.2619795073, 4064641.8209849824), (359828.2619795073, 4064631.8209849824), (359828.2619795073, 4064621.8209849824), (359828.2619795073, 4064611.8209849824), (359828.2619795073, 4064601.8209849824), (359828.2619795073, 4064591.8209849824), (359828.2619795073, 4064581.8209849824), (359828.2619795073, 4064571.8209849824), (359828.2619795073, 4064561.8209849824), (359828.2619795073, 4064551.8209849824), (359828.2619795073, 4064541.8209849824), (359828.2619795073, 4064531.8209849824), (359828.2619795073, 4064521.8209849824), (359828.2619795073, 4064511.8209849824), (359828.2619795073, 4064501.8209849824), (359828.2619795073, 4064491.8209849824), (359828.2619795073, 4064481.8209849824), (359828.2619795073, 4064471.8209849824), (359828.2619795073, 4064461.8209849824), (359828.2619795073, 4064451.8209849824), (359828.2619795073, 4064441.8209849824), (359828.2619795073, 4064431.8209849824), (359828.2619795073, 4064421.8209849824), (359828.2619795073, 4064411.8209849824), (359828.2619795073, 4064401.8209849824), (359828.2619795073, 4064391.8209849824), (359828.2619795073, 4064381.8209849824), (359828.2619795073, 4064371.8209849824), (359828.2619795073, 4064361.8209849824), (359828.2619795073, 4064351.8209849824), (359828.2619795073, 4064341.8209849824), (359828.2619795073, 4064331.8209849824), (359828.2619795073, 4064321.8209849824), (359828.2619795073, 4064311.8209849824), (359828.2619795073, 4064301.8209849824), (359828.2619795073, 4064291.8209849824), (359828.2619795073, 4064281.8209849824), (359828.2619795073, 4064271.8209849824), (359828.2619795073, 4064261.8209849824), (359828.2619795073, 4064251.8209849824), (359828.2619795073, 4064241.8209849824), (359828.2619795073, 4064231.8209849824), (359828.2619795073, 4064221.8209849824), (359828.2619795073, 4064211.8209849824), (359828.2619795073, 4064201.8209849824), (359828.2619795073, 4064191.8209849824), (359828.2619795073, 4064181.8209849824), (359828.2619795073, 4064171.8209849824), (359828.2619795073, 4064161.8209849824), (359828.2619795073, 4064151.8209849824), (359828.2619795073, 4064141.8209849824), (359828.2619795073, 4064131.8209849824), (359828.2619795073, 4064121.8209849824), (359828.2619795073, 4064111.8209849824), (359828.2619795073, 4064101.8209849824), (359828.2619795073, 
4064091.8209849824), (359828.2619795073, 4064081.8209849824), (359828.2619795073, 4064071.8209849824), (359828.2619795073, 4064061.8209849824), (359828.2619795073, 4064051.8209849824), (359828.2619795073, 4064041.8209849824), (359828.2619795073, 4064031.8209849824), (359828.2619795073, 4064021.8209849824), (359828.2619795073, 4064011.8209849824), (359828.2619795073, 4064001.8209849824), (359828.2619795073, 4063991.8209849824), (359828.2619795073, 4063981.8209849824), (359828.2619795073, 4063971.8209849824), (359828.2619795073, 4063961.8209849824), (359828.2619795073, 4063951.8209849824), (359828.2619795073, 4063941.8209849824), (359828.2619795073, 4063931.8209849824), (359828.2619795073, 4063921.8209849824), (359828.2619795073, 4063911.8209849824), (359828.2619795073, 4063901.8209849824), (359828.2619795073, 4063891.8209849824), (359828.2619795073, 4063881.8209849824), (359828.2619795073, 4063871.8209849824), (359828.2619795073, 4063861.8209849824), (359828.2619795073, 4063851.8209849824), (359828.2619795073, 4063841.8209849824), (359828.2619795073, 4063831.8209849824), (359828.2619795073, 4063821.8209849824), (359828.2619795073, 4063811.8209849824), (359828.2619795073, 4063801.8209849824), (359828.2619795073, 4063791.8209849824), (359828.2619795073, 4063781.8209849824), (359828.2619795073, 4063771.8209849824), (359828.2619795073, 4063761.8209849824), (359828.2619795073, 4063751.8209849824), (359828.2619795073, 4063741.8209849824), (359828.2619795073, 4063731.8209849824), (359828.2619795073, 4063721.8209849824), (359828.2619795073, 4063711.8209849824), (359828.2619795073, 4063701.8209849824), (359828.2619795073, 4063691.8209849824), (359828.2619795073, 4063681.8209849824), (359828.2619795073, 4063671.8209849824), (359828.2619795073, 4063661.8209849824), (359828.2619795073, 4063651.8209849824), (359828.2619795073, 4063641.8209849824), (359828.2619795073, 4063631.8209849824), (359828.2619795073, 4063621.8209849824), (359828.2619795073, 4063611.8209849824), (359828.2619795073, 4063601.8209849824), (359828.2619795073, 4063591.8209849824), (359828.2619795073, 4063581.8209849824), (359828.2619795073, 4063571.8209849824), (359828.2619795073, 4063561.8209849824), (359828.2619795073, 4063551.8209849824), (359828.2619795073, 4063541.8209849824), (359828.2619795073, 4063531.8209849824), (359828.2619795073, 4063521.8209849824), (359828.2619795073, 4063511.8209849824), (359828.2619795073, 4063501.8209849824), (359828.2619795073, 4063491.8209849824), (359828.2619795073, 4063481.8209849824), (359828.2619795073, 4063471.8209849824), (359828.2619795073, 4063461.8209849824), (359828.2619795073, 4063451.8209849824), (359828.2619795073, 4063441.8209849824), (359828.2619795073, 4063431.8209849824), (359828.2619795073, 4063421.8209849824), (359828.2619795073, 4063411.8209849824), (359828.2619795073, 4063401.8209849824), 
(359828.2619795073, 4063391.8209849824), (359828.2619795073, 4063381.8209849824), (359828.2619795073, 4063371.8209849824), (359828.2619795073, 4063361.8209849824), (359828.2619795073, 4063351.8209849824), (359828.2619795073, 4063341.8209849824), (359828.2619795073, 4063331.8209849824), (359828.2619795073, 4063321.8209849824), (359828.2619795073, 4063311.8209849824), (359828.2619795073, 4063301.8209849824), (359828.2619795073, 4063291.8209849824), (359828.2619795073, 4063281.8209849824), (359828.2619795073, 4063271.8209849824), (359828.2619795073, 4063261.8209849824), (359828.2619795073, 4063251.8209849824), (359828.2619795073, 4063241.8209849824), (359828.2619795073, 4063231.8209849824), (359828.2619795073, 4063221.8209849824), (359828.2619795073, 4063211.8209849824), (359828.2619795073, 4063201.8209849824), (359828.2619795073, 4063191.8209849824), (359828.2619795073, 4063181.8209849824), (359828.2619795073, 4063171.8209849824), (359828.2619795073, 4063161.8209849824), (359828.2619795073, 4063151.8209849824), (359828.2619795073, 4063141.8209849824), (359828.2619795073, 4063131.8209849824), (359828.2619795073, 4063121.8209849824), (359828.2619795073, 4063111.8209849824), (359828.2619795073, 4063101.8209849824), (359828.2619795073, 4063091.8209849824), (359828.2619795073, 4063081.8209849824), (359828.2619795073, 4063071.8209849824), (359828.2619795073, 4063061.8209849824), (359828.2619795073, 4063051.8209849824), (359828.2619795073, 4063041.8209849824), (359828.2619795073, 4063031.8209849824), (359828.2619795073, 4063021.8209849824), (359828.2619795073, 4063011.8209849824), (359828.2619795073, 4063001.8209849824), (359828.2619795073, 4062991.8209849824), (359828.2619795073, 4062981.8209849824), (359828.2619795073, 4062971.8209849824), (359828.2619795073, 4062961.8209849824), (359828.2619795073, 4062951.8209849824), (359828.2619795073, 4062941.8209849824), (359828.2619795073, 4062931.8209849824), (359828.2619795073, 4062921.8209849824), (359828.2619795073, 4062911.8209849824), (359828.2619795073, 4062901.8209849824), (359828.2619795073, 4062891.8209849824), (359828.2619795073, 4062881.8209849824), (359828.2619795073, 4062871.8209849824), (359828.2619795073, 4062861.8209849824), (359828.2619795073, 4062851.8209849824), (359828.2619795073, 4062841.8209849824), (359828.2619795073, 4062831.8209849824), (359828.2619795073, 4062821.8209849824), (359828.2619795073, 4062811.8209849824), (359828.2619795073, 4062801.8209849824), (359828.2619795073, 4062791.8209849824), (359828.2619795073, 4062781.8209849824), (359828.2619795073, 4062771.8209849824), (359828.2619795073, 4062761.8209849824), (359828.2619795073, 4062751.8209849824), (359828.2619795073, 4062741.8209849824), (359828.2619795073, 4062731.8209849824), (359828.2619795073, 4062721.8209849824), (359828.2619795073, 4062711.8209849824), (359828.2619795073, 4062701.8209849824), (359828.2619795073, 4062691.8209849824), (359828.2619795073, 4062681.8209849824), (359828.2619795073, 4062671.8209849824), (359828.2619795073, 4062661.8209849824), (359828.2619795073, 4062651.8209849824), (359828.2619795073, 4062641.8209849824), (359828.2619795073, 4062631.8209849824), (359828.2619795073, 4062621.8209849824), (359828.2619795073, 4062611.8209849824), (359828.2619795073, 4062601.8209849824), (359828.2619795073, 
4062591.8209849824), (359828.2619795073, 4062581.8209849824), (359828.2619795073, 4062571.8209849824), (359828.2619795073, 4062561.8209849824), (359828.2619795073, 4062551.8209849824), (359828.2619795073, 4062541.8209849824), (359828.2619795073, 4062531.8209849824), (359828.2619795073, 4062521.8209849824), (359828.2619795073, 4062511.8209849824), (359828.2619795073, 4062501.8209849824), (359828.2619795073, 4062491.8209849824), (359828.2619795073, 4062481.8209849824), (359828.2619795073, 4062471.8209849824), (359828.2619795073, 4062461.8209849824), (359828.2619795073, 4062451.8209849824), (359828.2619795073, 4062441.8209849824), (359828.2619795073, 4062431.8209849824), (359828.2619795073, 4062421.8209849824), (359828.2619795073, 4062411.8209849824), (359828.2619795073, 4062401.8209849824), (359828.2619795073, 4062391.8209849824), (359828.2619795073, 4062381.8209849824), (359828.2619795073, 4062371.8209849824), (359828.2619795073, 4062361.8209849824), (359828.2619795073, 4062351.8209849824), (359828.2619795073, 4062341.8209849824), (359828.2619795073, 4062331.8209849824), (359828.2619795073, 4062321.8209849824), (359828.2619795073, 4062311.8209849824), (359828.2619795073, 4062301.8209849824), (359828.2619795073, 4062291.8209849824), (359828.2619795073, 4062281.8209849824), (359828.2619795073, 4062271.8209849824), (359828.2619795073, 4062261.8209849824), (359828.2619795073, 4062251.8209849824), (359828.2619795073, 4062241.8209849824), (359828.2619795073, 4062231.8209849824), (359828.2619795073, 4062221.8209849824), (359828.2619795073, 4062211.8209849824), (359828.2619795073, 4062201.8209849824), (359828.2619795073, 4062191.8209849824), (359828.2619795073, 4062181.8209849824), (359828.2619795073, 4062171.8209849824), (359828.2619795073, 4062161.8209849824), (359828.2619795073, 4062151.8209849824), (359828.2619795073, 4062141.8209849824), (359828.2619795073, 4062131.8209849824), (359828.2619795073, 4062121.8209849824), (359828.2619795073, 4062111.8209849824), (359828.2619795073, 4062101.8209849824), (359828.2619795073, 4062091.8209849824), (359828.2619795073, 4062081.8209849824), (359828.2619795073, 4062071.8209849824), (359828.2619795073, 4062061.8209849824), (359838.2619795073, 4065321.8209849824), (359838.2619795073, 4065311.8209849824), (359838.2619795073, 4065301.8209849824), (359838.2619795073, 4065291.8209849824), (359838.2619795073, 4065281.8209849824), (359838.2619795073, 4065271.8209849824), (359838.2619795073, 4065261.8209849824), (359838.2619795073, 4065251.8209849824), (359838.2619795073, 4065241.8209849824), (359838.2619795073, 4065231.8209849824), (359838.2619795073, 4065221.8209849824), (359838.2619795073, 4065211.8209849824), (359838.2619795073, 4065201.8209849824), (359838.2619795073, 4065191.8209849824), (359838.2619795073, 4065181.8209849824), (359838.2619795073, 4065171.8209849824), 
(359838.2619795073, 4065161.8209849824), (359838.2619795073, 4065151.8209849824), (359838.2619795073, 4065141.8209849824), (359838.2619795073, 4065131.8209849824), (359838.2619795073, 4065121.8209849824), (359838.2619795073, 4065111.8209849824), (359838.2619795073, 4065101.8209849824), (359838.2619795073, 4065091.8209849824), (359838.2619795073, 4065081.8209849824), (359838.2619795073, 4065071.8209849824), (359838.2619795073, 4065061.8209849824), (359838.2619795073, 4065051.8209849824), (359838.2619795073, 4065041.8209849824), (359838.2619795073, 4065031.8209849824), (359838.2619795073, 4065021.8209849824), (359838.2619795073, 4065011.8209849824), (359838.2619795073, 4065001.8209849824), (359838.2619795073, 4064991.8209849824), (359838.2619795073, 4064981.8209849824), (359838.2619795073, 4064971.8209849824), (359838.2619795073, 4064961.8209849824), (359838.2619795073, 4064951.8209849824), (359838.2619795073, 4064941.8209849824), (359838.2619795073, 4064931.8209849824), (359838.2619795073, 4064921.8209849824), (359838.2619795073, 4064911.8209849824), (359838.2619795073, 4064901.8209849824), (359838.2619795073, 4064891.8209849824), (359838.2619795073, 4064881.8209849824), (359838.2619795073, 4064871.8209849824), (359838.2619795073, 4064861.8209849824), (359838.2619795073, 4064851.8209849824), (359838.2619795073, 4064841.8209849824), (359838.2619795073, 4064831.8209849824), (359838.2619795073, 4064821.8209849824), (359838.2619795073, 4064811.8209849824), (359838.2619795073, 4064801.8209849824), (359838.2619795073, 4064791.8209849824), (359838.2619795073, 4064781.8209849824), (359838.2619795073, 4064771.8209849824), (359838.2619795073, 4064761.8209849824), (359838.2619795073, 4064751.8209849824), (359838.2619795073, 4064741.8209849824), (359838.2619795073, 4064731.8209849824), (359838.2619795073, 4064721.8209849824), (359838.2619795073, 4064711.8209849824), (359838.2619795073, 4064701.8209849824), (359838.2619795073, 4064691.8209849824), (359838.2619795073, 4064681.8209849824), (359838.2619795073, 4064671.8209849824), (359838.2619795073, 4064661.8209849824), (359838.2619795073, 4064651.8209849824), (359838.2619795073, 4064641.8209849824), (359838.2619795073, 4064631.8209849824), (359838.2619795073, 4064621.8209849824), (359838.2619795073, 4064611.8209849824), (359838.2619795073, 4064601.8209849824), (359838.2619795073, 4064591.8209849824), (359838.2619795073, 4064581.8209849824), (359838.2619795073, 4064571.8209849824), (359838.2619795073, 4064561.8209849824), (359838.2619795073, 4064551.8209849824), (359838.2619795073, 4064541.8209849824), (359838.2619795073, 4064531.8209849824), (359838.2619795073, 4064521.8209849824), (359838.2619795073, 4064511.8209849824), (359838.2619795073, 4064501.8209849824), (359838.2619795073, 4064491.8209849824), (359838.2619795073, 4064481.8209849824), (359838.2619795073, 4064471.8209849824), (359838.2619795073, 4064461.8209849824), (359838.2619795073, 4064451.8209849824), (359838.2619795073, 4064441.8209849824), (359838.2619795073, 4064431.8209849824), (359838.2619795073, 4064421.8209849824), (359838.2619795073, 4064411.8209849824), (359838.2619795073, 4064401.8209849824), (359838.2619795073, 4064391.8209849824), (359838.2619795073, 4064381.8209849824), (359838.2619795073, 4064371.8209849824), (359838.2619795073, 
4064361.8209849824), (359838.2619795073, 4064351.8209849824), (359838.2619795073, 4064341.8209849824), (359838.2619795073, 4064331.8209849824), (359838.2619795073, 4064321.8209849824), (359838.2619795073, 4064311.8209849824), (359838.2619795073, 4064301.8209849824), (359838.2619795073, 4064291.8209849824), (359838.2619795073, 4064281.8209849824), (359838.2619795073, 4064271.8209849824), (359838.2619795073, 4064261.8209849824), (359838.2619795073, 4064251.8209849824), (359838.2619795073, 4064241.8209849824), (359838.2619795073, 4064231.8209849824), (359838.2619795073, 4064221.8209849824), (359838.2619795073, 4064211.8209849824), (359838.2619795073, 4064201.8209849824), (359838.2619795073, 4064191.8209849824), (359838.2619795073, 4064181.8209849824), (359838.2619795073, 4064171.8209849824), (359838.2619795073, 4064161.8209849824), (359838.2619795073, 4064151.8209849824), (359838.2619795073, 4064141.8209849824), (359838.2619795073, 4064131.8209849824), (359838.2619795073, 4064121.8209849824), (359838.2619795073, 4064111.8209849824), (359838.2619795073, 4064101.8209849824), (359838.2619795073, 4064091.8209849824), (359838.2619795073, 4064081.8209849824), (359838.2619795073, 4064071.8209849824), (359838.2619795073, 4064061.8209849824), (359838.2619795073, 4064051.8209849824), (359838.2619795073, 4064041.8209849824), (359838.2619795073, 4064031.8209849824), (359838.2619795073, 4064021.8209849824), (359838.2619795073, 4064011.8209849824), (359838.2619795073, 4064001.8209849824), (359838.2619795073, 4063991.8209849824), (359838.2619795073, 4063981.8209849824), (359838.2619795073, 4063971.8209849824), (359838.2619795073, 4063961.8209849824), (359838.2619795073, 4063951.8209849824), (359838.2619795073, 4063941.8209849824), (359838.2619795073, 4063931.8209849824), (359838.2619795073, 4063921.8209849824), (359838.2619795073, 4063911.8209849824), (359838.2619795073, 4063901.8209849824), (359838.2619795073, 4063891.8209849824), (359838.2619795073, 4063881.8209849824), (359838.2619795073, 4063871.8209849824), (359838.2619795073, 4063861.8209849824), (359838.2619795073, 4063851.8209849824), (359838.2619795073, 4063841.8209849824), (359838.2619795073, 4063831.8209849824), (359838.2619795073, 4063821.8209849824), (359838.2619795073, 4063811.8209849824), (359838.2619795073, 4063801.8209849824), (359838.2619795073, 4063791.8209849824), (359838.2619795073, 4063781.8209849824), (359838.2619795073, 4063771.8209849824), (359838.2619795073, 4063761.8209849824), (359838.2619795073, 4063751.8209849824), (359838.2619795073, 4063741.8209849824), (359838.2619795073, 4063731.8209849824), (359838.2619795073, 4063721.8209849824), (359838.2619795073, 4063711.8209849824), (359838.2619795073, 4063701.8209849824), (359838.2619795073, 4063691.8209849824), (359838.2619795073, 4063681.8209849824), (359838.2619795073, 4063671.8209849824), 
(359838.2619795073, 4063661.8209849824), (359838.2619795073, 4063651.8209849824), (359838.2619795073, 4063641.8209849824), (359838.2619795073, 4063631.8209849824), (359838.2619795073, 4063621.8209849824), (359838.2619795073, 4063611.8209849824), (359838.2619795073, 4063601.8209849824), (359838.2619795073, 4063591.8209849824), (359838.2619795073, 4063581.8209849824), (359838.2619795073, 4063571.8209849824), (359838.2619795073, 4063561.8209849824), (359838.2619795073, 4063551.8209849824), (359838.2619795073, 4063541.8209849824), (359838.2619795073, 4063531.8209849824), (359838.2619795073, 4063521.8209849824), (359838.2619795073, 4063511.8209849824), (359838.2619795073, 4063501.8209849824), (359838.2619795073, 4063491.8209849824), (359838.2619795073, 4063481.8209849824), (359838.2619795073, 4063471.8209849824), (359838.2619795073, 4063461.8209849824), (359838.2619795073, 4063451.8209849824), (359838.2619795073, 4063441.8209849824), (359838.2619795073, 4063431.8209849824), (359838.2619795073, 4063421.8209849824), (359838.2619795073, 4063411.8209849824), (359838.2619795073, 4063401.8209849824), (359838.2619795073, 4063391.8209849824), (359838.2619795073, 4063381.8209849824), (359838.2619795073, 4063371.8209849824), (359838.2619795073, 4063361.8209849824), (359838.2619795073, 4063351.8209849824), (359838.2619795073, 4063341.8209849824), (359838.2619795073, 4063331.8209849824), (359838.2619795073, 4063321.8209849824), (359838.2619795073, 4063311.8209849824), (359838.2619795073, 4063301.8209849824), (359838.2619795073, 4063291.8209849824), (359838.2619795073, 4063281.8209849824), (359838.2619795073, 4063271.8209849824), (359838.2619795073, 4063261.8209849824), (359838.2619795073, 4063251.8209849824), (359838.2619795073, 4063241.8209849824), (359838.2619795073, 4063231.8209849824), (359838.2619795073, 4063221.8209849824), (359838.2619795073, 4063211.8209849824), (359838.2619795073, 4063201.8209849824), (359838.2619795073, 4063191.8209849824), (359838.2619795073, 4063181.8209849824), (359838.2619795073, 4063171.8209849824), (359838.2619795073, 4063161.8209849824), (359838.2619795073, 4063151.8209849824), (359838.2619795073, 4063141.8209849824), (359838.2619795073, 4063131.8209849824), (359838.2619795073, 4063121.8209849824), (359838.2619795073, 4063111.8209849824), (359838.2619795073, 4063101.8209849824), (359838.2619795073, 4063091.8209849824), (359838.2619795073, 4063081.8209849824), (359838.2619795073, 4063071.8209849824), (359838.2619795073, 4063061.8209849824), (359838.2619795073, 4063051.8209849824), (359838.2619795073, 4063041.8209849824), (359838.2619795073, 4063031.8209849824), (359838.2619795073, 4063021.8209849824), (359838.2619795073, 4063011.8209849824), (359838.2619795073, 4063001.8209849824), (359838.2619795073, 4062991.8209849824), (359838.2619795073, 4062981.8209849824), (359838.2619795073, 4062971.8209849824), (359838.2619795073, 4062961.8209849824), (359838.2619795073, 4062951.8209849824), (359838.2619795073, 4062941.8209849824), (359838.2619795073, 4062931.8209849824), (359838.2619795073, 4062921.8209849824), (359838.2619795073, 4062911.8209849824), (359838.2619795073, 4062901.8209849824), (359838.2619795073, 4062891.8209849824), (359838.2619795073, 4062881.8209849824), (359838.2619795073, 4062871.8209849824), (359838.2619795073, 
4062861.8209849824), (359838.2619795073, 4062851.8209849824), (359838.2619795073, 4062841.8209849824), (359838.2619795073, 4062831.8209849824), (359838.2619795073, 4062821.8209849824), (359838.2619795073, 4062811.8209849824), (359838.2619795073, 4062801.8209849824), (359838.2619795073, 4062791.8209849824), (359838.2619795073, 4062781.8209849824), (359838.2619795073, 4062771.8209849824), (359838.2619795073, 4062761.8209849824), (359838.2619795073, 4062751.8209849824), (359838.2619795073, 4062741.8209849824), (359838.2619795073, 4062731.8209849824), (359838.2619795073, 4062721.8209849824), (359838.2619795073, 4062711.8209849824), (359838.2619795073, 4062701.8209849824), (359838.2619795073, 4062691.8209849824), (359838.2619795073, 4062681.8209849824), (359838.2619795073, 4062671.8209849824), (359838.2619795073, 4062661.8209849824), (359838.2619795073, 4062651.8209849824), (359838.2619795073, 4062641.8209849824), (359838.2619795073, 4062631.8209849824), (359838.2619795073, 4062621.8209849824), (359838.2619795073, 4062611.8209849824), (359838.2619795073, 4062601.8209849824), (359838.2619795073, 4062591.8209849824), (359838.2619795073, 4062581.8209849824), (359838.2619795073, 4062571.8209849824), (359838.2619795073, 4062561.8209849824), (359838.2619795073, 4062551.8209849824), (359838.2619795073, 4062541.8209849824), (359838.2619795073, 4062531.8209849824), (359838.2619795073, 4062521.8209849824), (359838.2619795073, 4062511.8209849824), (359838.2619795073, 4062501.8209849824), (359838.2619795073, 4062491.8209849824), (359838.2619795073, 4062481.8209849824), (359838.2619795073, 4062471.8209849824), (359838.2619795073, 4062461.8209849824), (359838.2619795073, 4062451.8209849824), (359838.2619795073, 4062441.8209849824), (359838.2619795073, 4062431.8209849824), (359838.2619795073, 4062421.8209849824), (359838.2619795073, 4062411.8209849824), (359838.2619795073, 4062401.8209849824), (359838.2619795073, 4062391.8209849824), (359838.2619795073, 4062381.8209849824), (359838.2619795073, 4062371.8209849824), (359838.2619795073, 4062361.8209849824), (359838.2619795073, 4062351.8209849824), (359838.2619795073, 4062341.8209849824), (359838.2619795073, 4062331.8209849824), (359838.2619795073, 4062321.8209849824), (359838.2619795073, 4062311.8209849824), (359838.2619795073, 4062301.8209849824), (359838.2619795073, 4062291.8209849824), (359838.2619795073, 4062281.8209849824), (359838.2619795073, 4062271.8209849824), (359838.2619795073, 4062261.8209849824), (359838.2619795073, 4062251.8209849824), (359838.2619795073, 4062241.8209849824), (359838.2619795073, 4062231.8209849824), (359838.2619795073, 4062221.8209849824), (359838.2619795073, 4062211.8209849824), (359838.2619795073, 4062201.8209849824), (359838.2619795073, 4062191.8209849824), (359838.2619795073, 4062181.8209849824), (359838.2619795073, 4062171.8209849824), 
(359838.2619795073, 4062161.8209849824), (359838.2619795073, 4062151.8209849824), (359838.2619795073, 4062141.8209849824), (359838.2619795073, 4062131.8209849824), (359838.2619795073, 4062121.8209849824), (359838.2619795073, 4062111.8209849824), (359838.2619795073, 4062101.8209849824), (359838.2619795073, 4062091.8209849824), (359838.2619795073, 4062081.8209849824), (359838.2619795073, 4062071.8209849824), (359838.2619795073, 4062061.8209849824), (359838.2619795073, 4062051.8209849824), (359848.2619795073, 4065321.8209849824), (359848.2619795073, 4065311.8209849824), (359848.2619795073, 4065301.8209849824), (359848.2619795073, 4065291.8209849824), (359848.2619795073, 4065281.8209849824), (359848.2619795073, 4065271.8209849824), (359848.2619795073, 4065261.8209849824), (359848.2619795073, 4065251.8209849824), (359848.2619795073, 4065241.8209849824), (359848.2619795073, 4065231.8209849824), (359848.2619795073, 4065221.8209849824), (359848.2619795073, 4065211.8209849824), (359848.2619795073, 4065201.8209849824), (359848.2619795073, 4065191.8209849824), (359848.2619795073, 4065181.8209849824), (359848.2619795073, 4065171.8209849824), (359848.2619795073, 4065161.8209849824), (359848.2619795073, 4065151.8209849824), (359848.2619795073, 4065141.8209849824), (359848.2619795073, 4065131.8209849824), (359848.2619795073, 4065121.8209849824), (359848.2619795073, 4065111.8209849824), (359848.2619795073, 4065101.8209849824), (359848.2619795073, 4065091.8209849824), (359848.2619795073, 4065081.8209849824), (359848.2619795073, 4065071.8209849824), (359848.2619795073, 4065061.8209849824), (359848.2619795073, 4065051.8209849824), (359848.2619795073, 4065041.8209849824), (359848.2619795073, 4065031.8209849824), (359848.2619795073, 4065021.8209849824), (359848.2619795073, 4065011.8209849824), (359848.2619795073, 4065001.8209849824), (359848.2619795073, 4064991.8209849824), (359848.2619795073, 4064981.8209849824), (359848.2619795073, 4064971.8209849824), (359848.2619795073, 4064961.8209849824), (359848.2619795073, 4064951.8209849824), (359848.2619795073, 4064941.8209849824), (359848.2619795073, 4064931.8209849824), (359848.2619795073, 4064921.8209849824), (359848.2619795073, 4064911.8209849824), (359848.2619795073, 4064901.8209849824), (359848.2619795073, 4064891.8209849824), (359848.2619795073, 4064881.8209849824), (359848.2619795073, 4064871.8209849824), (359848.2619795073, 4064861.8209849824), (359848.2619795073, 4064851.8209849824), (359848.2619795073, 4064841.8209849824), (359848.2619795073, 4064831.8209849824), (359848.2619795073, 4064821.8209849824), (359848.2619795073, 4064811.8209849824), (359848.2619795073, 4064801.8209849824), (359848.2619795073, 4064791.8209849824), (359848.2619795073, 4064781.8209849824), (359848.2619795073, 4064771.8209849824), (359848.2619795073, 4064761.8209849824), (359848.2619795073, 4064751.8209849824), (359848.2619795073, 4064741.8209849824), (359848.2619795073, 4064731.8209849824), (359848.2619795073, 4064721.8209849824), (359848.2619795073, 4064711.8209849824), (359848.2619795073, 4064701.8209849824), (359848.2619795073, 4064691.8209849824), (359848.2619795073, 4064681.8209849824), (359848.2619795073, 4064671.8209849824), (359848.2619795073, 4064661.8209849824), (359848.2619795073, 4064651.8209849824), (359848.2619795073, 
4064641.8209849824), (359848.2619795073, 4064631.8209849824), (359848.2619795073, 4064621.8209849824), (359848.2619795073, 4064611.8209849824), (359848.2619795073, 4064601.8209849824), (359848.2619795073, 4064591.8209849824), (359848.2619795073, 4064581.8209849824), (359848.2619795073, 4064571.8209849824), (359848.2619795073, 4064561.8209849824), (359848.2619795073, 4064551.8209849824), (359848.2619795073, 4064541.8209849824), (359848.2619795073, 4064531.8209849824), (359848.2619795073, 4064521.8209849824), (359848.2619795073, 4064511.8209849824), (359848.2619795073, 4064501.8209849824), (359848.2619795073, 4064491.8209849824), (359848.2619795073, 4064481.8209849824), (359848.2619795073, 4064471.8209849824), (359848.2619795073, 4064461.8209849824), (359848.2619795073, 4064451.8209849824), (359848.2619795073, 4064441.8209849824), (359848.2619795073, 4064431.8209849824), (359848.2619795073, 4064421.8209849824), (359848.2619795073, 4064411.8209849824), (359848.2619795073, 4064401.8209849824), (359848.2619795073, 4064391.8209849824), (359848.2619795073, 4064381.8209849824), (359848.2619795073, 4064371.8209849824), (359848.2619795073, 4064361.8209849824), (359848.2619795073, 4064351.8209849824), (359848.2619795073, 4064341.8209849824), (359848.2619795073, 4064331.8209849824), (359848.2619795073, 4064321.8209849824), (359848.2619795073, 4064311.8209849824), (359848.2619795073, 4064301.8209849824), (359848.2619795073, 4064291.8209849824), (359848.2619795073, 4064281.8209849824), (359848.2619795073, 4064271.8209849824), (359848.2619795073, 4064261.8209849824), (359848.2619795073, 4064251.8209849824), (359848.2619795073, 4064241.8209849824), (359848.2619795073, 4064231.8209849824), (359848.2619795073, 4064221.8209849824), (359848.2619795073, 4064211.8209849824), (359848.2619795073, 4064201.8209849824), (359848.2619795073, 4064191.8209849824), (359848.2619795073, 4064181.8209849824), (359848.2619795073, 4064171.8209849824), (359848.2619795073, 4064161.8209849824), (359848.2619795073, 4064151.8209849824), (359848.2619795073, 4064141.8209849824), (359848.2619795073, 4064131.8209849824), (359848.2619795073, 4064121.8209849824), (359848.2619795073, 4064111.8209849824), (359848.2619795073, 4064101.8209849824), (359848.2619795073, 4064091.8209849824), (359848.2619795073, 4064081.8209849824), (359848.2619795073, 4064071.8209849824), (359848.2619795073, 4064061.8209849824), (359848.2619795073, 4064051.8209849824), (359848.2619795073, 4064041.8209849824), (359848.2619795073, 4064031.8209849824), (359848.2619795073, 4064021.8209849824), (359848.2619795073, 4064011.8209849824), (359848.2619795073, 4064001.8209849824), (359848.2619795073, 4063991.8209849824), (359848.2619795073, 4063981.8209849824), (359848.2619795073, 4063971.8209849824), (359848.2619795073, 4063961.8209849824), (359848.2619795073, 4063951.8209849824), 
(359848.2619795073, 4063941.8209849824), (359848.2619795073, 4063931.8209849824), (359848.2619795073, 4063921.8209849824), (359848.2619795073, 4063911.8209849824), (359848.2619795073, 4063901.8209849824), (359848.2619795073, 4063891.8209849824), (359848.2619795073, 4063881.8209849824), (359848.2619795073, 4063871.8209849824), (359848.2619795073, 4063861.8209849824), (359848.2619795073, 4063851.8209849824), (359848.2619795073, 4063841.8209849824), (359848.2619795073, 4063831.8209849824), (359848.2619795073, 4063821.8209849824), (359848.2619795073, 4063811.8209849824), (359848.2619795073, 4063801.8209849824), (359848.2619795073, 4063791.8209849824), (359848.2619795073, 4063781.8209849824), (359848.2619795073, 4063771.8209849824), (359848.2619795073, 4063761.8209849824), (359848.2619795073, 4063751.8209849824), (359848.2619795073, 4063741.8209849824), (359848.2619795073, 4063731.8209849824), (359848.2619795073, 4063721.8209849824), (359848.2619795073, 4063711.8209849824), (359848.2619795073, 4063701.8209849824), (359848.2619795073, 4063691.8209849824), (359848.2619795073, 4063681.8209849824), (359848.2619795073, 4063671.8209849824), (359848.2619795073, 4063661.8209849824), (359848.2619795073, 4063651.8209849824), (359848.2619795073, 4063641.8209849824), (359848.2619795073, 4063631.8209849824), (359848.2619795073, 4063621.8209849824), (359848.2619795073, 4063611.8209849824), (359848.2619795073, 4063601.8209849824), (359848.2619795073, 4063591.8209849824), (359848.2619795073, 4063581.8209849824), (359848.2619795073, 4063571.8209849824), (359848.2619795073, 4063561.8209849824), (359848.2619795073, 4063551.8209849824), (359848.2619795073, 4063541.8209849824), (359848.2619795073, 4063531.8209849824), (359848.2619795073, 4063521.8209849824), (359848.2619795073, 4063511.8209849824), (359848.2619795073, 4063501.8209849824), (359848.2619795073, 4063491.8209849824), (359848.2619795073, 4063481.8209849824), (359848.2619795073, 4063471.8209849824), (359848.2619795073, 4063461.8209849824), (359848.2619795073, 4063451.8209849824), (359848.2619795073, 4063441.8209849824), (359848.2619795073, 4063431.8209849824), (359848.2619795073, 4063421.8209849824), (359848.2619795073, 4063411.8209849824), (359848.2619795073, 4063401.8209849824), (359848.2619795073, 4063391.8209849824), (359848.2619795073, 4063381.8209849824), (359848.2619795073, 4063371.8209849824), (359848.2619795073, 4063361.8209849824), (359848.2619795073, 4063351.8209849824), (359848.2619795073, 4063341.8209849824), (359848.2619795073, 4063331.8209849824), (359848.2619795073, 4063321.8209849824), (359848.2619795073, 4063311.8209849824), (359848.2619795073, 4063301.8209849824), (359848.2619795073, 4063291.8209849824), (359848.2619795073, 4063281.8209849824), (359848.2619795073, 4063271.8209849824), (359848.2619795073, 4063261.8209849824), (359848.2619795073, 4063251.8209849824), (359848.2619795073, 4063241.8209849824), (359848.2619795073, 4063231.8209849824), (359848.2619795073, 4063221.8209849824), (359848.2619795073, 4063211.8209849824), (359848.2619795073, 4063201.8209849824), (359848.2619795073, 4063191.8209849824), (359848.2619795073, 4063181.8209849824), (359848.2619795073, 4063171.8209849824), (359848.2619795073, 4063161.8209849824), (359848.2619795073, 4063151.8209849824), (359848.2619795073, 
4063141.8209849824), (359848.2619795073, 4063131.8209849824), (359848.2619795073, 4063121.8209849824), (359848.2619795073, 4063111.8209849824), (359848.2619795073, 4063101.8209849824), (359848.2619795073, 4063091.8209849824), (359848.2619795073, 4063081.8209849824), (359848.2619795073, 4063071.8209849824), (359848.2619795073, 4063051.8209849824), (359848.2619795073, 4063041.8209849824), (359848.2619795073, 4063031.8209849824), (359848.2619795073, 4063021.8209849824), (359848.2619795073, 4063011.8209849824), (359848.2619795073, 4063001.8209849824), (359848.2619795073, 4062991.8209849824), (359848.2619795073, 4062981.8209849824), (359848.2619795073, 4062971.8209849824), (359848.2619795073, 4062961.8209849824), (359848.2619795073, 4062951.8209849824), (359848.2619795073, 4062941.8209849824), (359848.2619795073, 4062931.8209849824), (359848.2619795073, 4062921.8209849824), (359848.2619795073, 4062911.8209849824), (359848.2619795073, 4062901.8209849824), (359848.2619795073, 4062891.8209849824), (359848.2619795073, 4062881.8209849824), (359848.2619795073, 4062871.8209849824), (359848.2619795073, 4062861.8209849824), (359848.2619795073, 4062851.8209849824), (359848.2619795073, 4062841.8209849824), (359848.2619795073, 4062831.8209849824), (359848.2619795073, 4062821.8209849824), (359848.2619795073, 4062811.8209849824), (359848.2619795073, 4062801.8209849824), (359848.2619795073, 4062791.8209849824), (359848.2619795073, 4062781.8209849824), (359848.2619795073, 4062771.8209849824), (359848.2619795073, 4062761.8209849824), (359848.2619795073, 4062751.8209849824), (359848.2619795073, 4062741.8209849824), (359848.2619795073, 4062731.8209849824), (359848.2619795073, 4062721.8209849824), (359848.2619795073, 4062711.8209849824), (359848.2619795073, 4062701.8209849824), (359848.2619795073, 4062691.8209849824), (359848.2619795073, 4062681.8209849824), (359848.2619795073, 4062671.8209849824), (359848.2619795073, 4062661.8209849824), (359848.2619795073, 4062651.8209849824), (359848.2619795073, 4062641.8209849824), (359848.2619795073, 4062631.8209849824), (359848.2619795073, 4062621.8209849824), (359848.2619795073, 4062611.8209849824), (359848.2619795073, 4062601.8209849824), (359848.2619795073, 4062591.8209849824), (359848.2619795073, 4062581.8209849824), (359848.2619795073, 4062571.8209849824), (359848.2619795073, 4062561.8209849824), (359848.2619795073, 4062551.8209849824), (359848.2619795073, 4062541.8209849824), (359848.2619795073, 4062531.8209849824), (359848.2619795073, 4062521.8209849824), (359848.2619795073, 4062511.8209849824), (359848.2619795073, 4062501.8209849824), (359848.2619795073, 4062491.8209849824), (359848.2619795073, 4062481.8209849824), (359848.2619795073, 4062471.8209849824), (359848.2619795073, 4062461.8209849824), (359848.2619795073, 4062451.8209849824), (359848.2619795073, 4062441.8209849824), 
(359848.2619795073, 4062431.8209849824), (359848.2619795073, 4062421.8209849824), (359848.2619795073, 4062411.8209849824), (359848.2619795073, 4062401.8209849824), (359848.2619795073, 4062391.8209849824), (359848.2619795073, 4062381.8209849824), (359848.2619795073, 4062371.8209849824), (359848.2619795073, 4062361.8209849824), (359848.2619795073, 4062351.8209849824), (359848.2619795073, 4062341.8209849824), (359848.2619795073, 4062331.8209849824), (359848.2619795073, 4062321.8209849824), (359848.2619795073, 4062311.8209849824), (359848.2619795073, 4062301.8209849824), (359848.2619795073, 4062291.8209849824), (359848.2619795073, 4062281.8209849824), (359848.2619795073, 4062271.8209849824), (359848.2619795073, 4062261.8209849824), (359848.2619795073, 4062251.8209849824), (359848.2619795073, 4062241.8209849824), (359848.2619795073, 4062231.8209849824), (359848.2619795073, 4062221.8209849824), (359848.2619795073, 4062211.8209849824), (359848.2619795073, 4062201.8209849824), (359848.2619795073, 4062191.8209849824), (359848.2619795073, 4062181.8209849824), (359848.2619795073, 4062171.8209849824), (359848.2619795073, 4062161.8209849824), (359848.2619795073, 4062151.8209849824), (359848.2619795073, 4062141.8209849824), (359848.2619795073, 4062131.8209849824), (359848.2619795073, 4062121.8209849824), (359848.2619795073, 4062111.8209849824), (359848.2619795073, 4062101.8209849824), (359848.2619795073, 4062091.8209849824), (359848.2619795073, 4062081.8209849824), (359848.2619795073, 4062071.8209849824), (359848.2619795073, 4062061.8209849824), (359848.2619795073, 4062051.8209849824), (359858.2619795073, 4065331.8209849824), (359858.2619795073, 4065321.8209849824), (359858.2619795073, 4065311.8209849824), (359858.2619795073, 4065301.8209849824), (359858.2619795073, 4065291.8209849824), (359858.2619795073, 4065281.8209849824), (359858.2619795073, 4065271.8209849824), (359858.2619795073, 4065261.8209849824), (359858.2619795073, 4065251.8209849824), (359858.2619795073, 4065241.8209849824), (359858.2619795073, 4065231.8209849824), (359858.2619795073, 4065221.8209849824), (359858.2619795073, 4065211.8209849824), (359858.2619795073, 4065201.8209849824), (359858.2619795073, 4065191.8209849824), (359858.2619795073, 4065181.8209849824), (359858.2619795073, 4065171.8209849824), (359858.2619795073, 4065161.8209849824), (359858.2619795073, 4065151.8209849824), (359858.2619795073, 4065141.8209849824), (359858.2619795073, 4065131.8209849824), (359858.2619795073, 4065121.8209849824), (359858.2619795073, 4065111.8209849824), (359858.2619795073, 4065101.8209849824), (359858.2619795073, 4065091.8209849824), (359858.2619795073, 4065081.8209849824), (359858.2619795073, 4065071.8209849824), (359858.2619795073, 4065061.8209849824), (359858.2619795073, 4065051.8209849824), (359858.2619795073, 4065041.8209849824), (359858.2619795073, 4065031.8209849824), (359858.2619795073, 4065021.8209849824), (359858.2619795073, 4065011.8209849824), (359858.2619795073, 4065001.8209849824), (359858.2619795073, 4064991.8209849824), (359858.2619795073, 4064981.8209849824), (359858.2619795073, 4064971.8209849824), (359858.2619795073, 4064961.8209849824), (359858.2619795073, 4064951.8209849824), (359858.2619795073, 4064941.8209849824), (359858.2619795073, 4064931.8209849824), (359858.2619795073, 
4064921.8209849824), (359858.2619795073, 4064911.8209849824), (359858.2619795073, 4064901.8209849824), (359858.2619795073, 4064891.8209849824), (359858.2619795073, 4064881.8209849824), (359858.2619795073, 4064871.8209849824), (359858.2619795073, 4064861.8209849824), (359858.2619795073, 4064851.8209849824), (359858.2619795073, 4064841.8209849824), (359858.2619795073, 4064831.8209849824), (359858.2619795073, 4064821.8209849824), (359858.2619795073, 4064811.8209849824), (359858.2619795073, 4064801.8209849824), (359858.2619795073, 4064791.8209849824), (359858.2619795073, 4064781.8209849824), (359858.2619795073, 4064771.8209849824), (359858.2619795073, 4064761.8209849824), (359858.2619795073, 4064751.8209849824), (359858.2619795073, 4064741.8209849824), (359858.2619795073, 4064731.8209849824), (359858.2619795073, 4064721.8209849824), (359858.2619795073, 4064711.8209849824), (359858.2619795073, 4064701.8209849824), (359858.2619795073, 4064691.8209849824), (359858.2619795073, 4064681.8209849824), (359858.2619795073, 4064671.8209849824), (359858.2619795073, 4064661.8209849824), (359858.2619795073, 4064651.8209849824), (359858.2619795073, 4064641.8209849824), (359858.2619795073, 4064631.8209849824), (359858.2619795073, 4064621.8209849824), (359858.2619795073, 4064611.8209849824), (359858.2619795073, 4064601.8209849824), (359858.2619795073, 4064591.8209849824), (359858.2619795073, 4064581.8209849824), (359858.2619795073, 4064571.8209849824), (359858.2619795073, 4064561.8209849824), (359858.2619795073, 4064551.8209849824), (359858.2619795073, 4064541.8209849824), (359858.2619795073, 4064531.8209849824), (359858.2619795073, 4064521.8209849824), (359858.2619795073, 4064511.8209849824), (359858.2619795073, 4064501.8209849824), (359858.2619795073, 4064491.8209849824), (359858.2619795073, 4064481.8209849824), (359858.2619795073, 4064471.8209849824), (359858.2619795073, 4064461.8209849824), (359858.2619795073, 4064451.8209849824), (359858.2619795073, 4064441.8209849824), (359858.2619795073, 4064431.8209849824), (359858.2619795073, 4064421.8209849824), (359858.2619795073, 4064411.8209849824), (359858.2619795073, 4064401.8209849824), (359858.2619795073, 4064391.8209849824), (359858.2619795073, 4064381.8209849824), (359858.2619795073, 4064371.8209849824), (359858.2619795073, 4064361.8209849824), (359858.2619795073, 4064351.8209849824), (359858.2619795073, 4064341.8209849824), (359858.2619795073, 4064331.8209849824), (359858.2619795073, 4064321.8209849824), (359858.2619795073, 4064311.8209849824), (359858.2619795073, 4064301.8209849824), (359858.2619795073, 4064291.8209849824), (359858.2619795073, 4064281.8209849824), (359858.2619795073, 4064271.8209849824), (359858.2619795073, 4064261.8209849824), (359858.2619795073, 4064251.8209849824), (359858.2619795073, 4064241.8209849824), (359858.2619795073, 4064231.8209849824), 
(359858.2619795073, 4064221.8209849824), (359858.2619795073, 4064211.8209849824), (359858.2619795073, 4064201.8209849824), (359858.2619795073, 4064191.8209849824), (359858.2619795073, 4064181.8209849824), (359858.2619795073, 4064171.8209849824), (359858.2619795073, 4064161.8209849824), (359858.2619795073, 4064151.8209849824), (359858.2619795073, 4064141.8209849824), (359858.2619795073, 4064131.8209849824), (359858.2619795073, 4064121.8209849824), (359858.2619795073, 4064111.8209849824), (359858.2619795073, 4064101.8209849824), (359858.2619795073, 4064091.8209849824), (359858.2619795073, 4064081.8209849824), (359858.2619795073, 4064071.8209849824), (359858.2619795073, 4064061.8209849824), (359858.2619795073, 4064051.8209849824), (359858.2619795073, 4064041.8209849824), (359858.2619795073, 4064031.8209849824), (359858.2619795073, 4064021.8209849824), (359858.2619795073, 4064011.8209849824), (359858.2619795073, 4064001.8209849824), (359858.2619795073, 4063991.8209849824), (359858.2619795073, 4063981.8209849824), (359858.2619795073, 4063971.8209849824), (359858.2619795073, 4063961.8209849824), (359858.2619795073, 4063951.8209849824), (359858.2619795073, 4063941.8209849824), (359858.2619795073, 4063931.8209849824), (359858.2619795073, 4063921.8209849824), (359858.2619795073, 4063911.8209849824), (359858.2619795073, 4063901.8209849824), (359858.2619795073, 4063891.8209849824), (359858.2619795073, 4063881.8209849824), (359858.2619795073, 4063871.8209849824), (359858.2619795073, 4063861.8209849824), (359858.2619795073, 4063851.8209849824), (359858.2619795073, 4063841.8209849824), (359858.2619795073, 4063831.8209849824), (359858.2619795073, 4063821.8209849824), (359858.2619795073, 4063811.8209849824), (359858.2619795073, 4063801.8209849824), (359858.2619795073, 4063791.8209849824), (359858.2619795073, 4063781.8209849824), (359858.2619795073, 4063771.8209849824), (359858.2619795073, 4063761.8209849824), (359858.2619795073, 4063751.8209849824), (359858.2619795073, 4063741.8209849824), (359858.2619795073, 4063731.8209849824), (359858.2619795073, 4063721.8209849824), (359858.2619795073, 4063711.8209849824), (359858.2619795073, 4063701.8209849824), (359858.2619795073, 4063691.8209849824), (359858.2619795073, 4063681.8209849824), (359858.2619795073, 4063671.8209849824), (359858.2619795073, 4063661.8209849824), (359858.2619795073, 4063651.8209849824), (359858.2619795073, 4063641.8209849824), (359858.2619795073, 4063631.8209849824), (359858.2619795073, 4063621.8209849824), (359858.2619795073, 4063611.8209849824), (359858.2619795073, 4063601.8209849824), (359858.2619795073, 4063591.8209849824), (359858.2619795073, 4063581.8209849824), (359858.2619795073, 4063571.8209849824), (359858.2619795073, 4063561.8209849824), (359858.2619795073, 4063551.8209849824), (359858.2619795073, 4063541.8209849824), (359858.2619795073, 4063531.8209849824), (359858.2619795073, 4063521.8209849824), (359858.2619795073, 4063511.8209849824), (359858.2619795073, 4063501.8209849824), (359858.2619795073, 4063491.8209849824), (359858.2619795073, 4063481.8209849824), (359858.2619795073, 4063471.8209849824), (359858.2619795073, 4063461.8209849824), (359858.2619795073, 4063451.8209849824), (359858.2619795073, 4063441.8209849824), (359858.2619795073, 4063431.8209849824), (359858.2619795073, 
4063421.8209849824), (359858.2619795073, 4063411.8209849824), (359858.2619795073, 4063401.8209849824), (359858.2619795073, 4063391.8209849824), (359858.2619795073, 4063381.8209849824), (359858.2619795073, 4063371.8209849824), (359858.2619795073, 4063361.8209849824), (359858.2619795073, 4063351.8209849824), (359858.2619795073, 4063341.8209849824), (359858.2619795073, 4063331.8209849824), (359858.2619795073, 4063321.8209849824), (359858.2619795073, 4063311.8209849824), (359858.2619795073, 4063301.8209849824), (359858.2619795073, 4063291.8209849824), (359858.2619795073, 4063281.8209849824), (359858.2619795073, 4063271.8209849824), (359858.2619795073, 4063261.8209849824), (359858.2619795073, 4063251.8209849824), (359858.2619795073, 4063241.8209849824), (359858.2619795073, 4063231.8209849824), (359858.2619795073, 4063221.8209849824), (359858.2619795073, 4063211.8209849824), (359858.2619795073, 4063201.8209849824), (359858.2619795073, 4063191.8209849824), (359858.2619795073, 4063181.8209849824), (359858.2619795073, 4063171.8209849824), (359858.2619795073, 4063161.8209849824), (359858.2619795073, 4063151.8209849824), (359858.2619795073, 4063141.8209849824), (359858.2619795073, 4063131.8209849824), (359858.2619795073, 4063121.8209849824), (359858.2619795073, 4063111.8209849824), (359858.2619795073, 4063101.8209849824), (359858.2619795073, 4063091.8209849824), (359858.2619795073, 4063081.8209849824), (359858.2619795073, 4063071.8209849824), (359858.2619795073, 4063061.8209849824), (359858.2619795073, 4063051.8209849824), (359858.2619795073, 4063041.8209849824), (359858.2619795073, 4063031.8209849824), (359858.2619795073, 4063021.8209849824), (359858.2619795073, 4063011.8209849824), (359858.2619795073, 4063001.8209849824), (359858.2619795073, 4062991.8209849824), (359858.2619795073, 4062981.8209849824), (359858.2619795073, 4062971.8209849824), (359858.2619795073, 4062961.8209849824), (359858.2619795073, 4062951.8209849824), (359858.2619795073, 4062941.8209849824), (359858.2619795073, 4062931.8209849824), (359858.2619795073, 4062921.8209849824), (359858.2619795073, 4062911.8209849824), (359858.2619795073, 4062901.8209849824), (359858.2619795073, 4062891.8209849824), (359858.2619795073, 4062881.8209849824), (359858.2619795073, 4062871.8209849824), (359858.2619795073, 4062861.8209849824), (359858.2619795073, 4062851.8209849824), (359858.2619795073, 4062841.8209849824), (359858.2619795073, 4062831.8209849824), (359858.2619795073, 4062821.8209849824), (359858.2619795073, 4062811.8209849824), (359858.2619795073, 4062801.8209849824), (359858.2619795073, 4062791.8209849824), (359858.2619795073, 4062781.8209849824), (359858.2619795073, 4062771.8209849824), (359858.2619795073, 4062761.8209849824), (359858.2619795073, 4062751.8209849824), (359858.2619795073, 4062741.8209849824), (359858.2619795073, 4062731.8209849824), 
(359858.2619795073, 4062721.8209849824), (359858.2619795073, 4062711.8209849824), (359858.2619795073, 4062701.8209849824), (359858.2619795073, 4062691.8209849824), (359858.2619795073, 4062681.8209849824), (359858.2619795073, 4062671.8209849824), (359858.2619795073, 4062661.8209849824), (359858.2619795073, 4062651.8209849824), (359858.2619795073, 4062641.8209849824), (359858.2619795073, 4062631.8209849824), (359858.2619795073, 4062621.8209849824), (359858.2619795073, 4062611.8209849824), (359858.2619795073, 4062601.8209849824), (359858.2619795073, 4062591.8209849824), (359858.2619795073, 4062581.8209849824), (359858.2619795073, 4062571.8209849824), (359858.2619795073, 4062561.8209849824), (359858.2619795073, 4062551.8209849824), (359858.2619795073, 4062541.8209849824), (359858.2619795073, 4062531.8209849824), (359858.2619795073, 4062521.8209849824), (359858.2619795073, 4062511.8209849824), (359858.2619795073, 4062501.8209849824), (359858.2619795073, 4062491.8209849824), (359858.2619795073, 4062481.8209849824), (359858.2619795073, 4062471.8209849824), (359858.2619795073, 4062461.8209849824), (359858.2619795073, 4062451.8209849824), (359858.2619795073, 4062441.8209849824), (359858.2619795073, 4062431.8209849824), (359858.2619795073, 4062421.8209849824), (359858.2619795073, 4062411.8209849824), (359858.2619795073, 4062401.8209849824), (359858.2619795073, 4062391.8209849824), (359858.2619795073, 4062381.8209849824), (359858.2619795073, 4062371.8209849824), (359858.2619795073, 4062361.8209849824), (359858.2619795073, 4062351.8209849824), (359858.2619795073, 4062341.8209849824), (359858.2619795073, 4062331.8209849824), (359858.2619795073, 4062321.8209849824), (359858.2619795073, 4062311.8209849824), (359858.2619795073, 4062301.8209849824), (359858.2619795073, 4062291.8209849824), (359858.2619795073, 4062281.8209849824), (359858.2619795073, 4062271.8209849824), (359858.2619795073, 4062261.8209849824), (359858.2619795073, 4062251.8209849824), (359858.2619795073, 4062241.8209849824), (359858.2619795073, 4062231.8209849824), (359858.2619795073, 4062221.8209849824), (359858.2619795073, 4062211.8209849824), (359858.2619795073, 4062201.8209849824), (359858.2619795073, 4062191.8209849824), (359858.2619795073, 4062181.8209849824), (359858.2619795073, 4062171.8209849824), (359858.2619795073, 4062161.8209849824), (359858.2619795073, 4062151.8209849824), (359858.2619795073, 4062141.8209849824), (359858.2619795073, 4062131.8209849824), (359858.2619795073, 4062121.8209849824), (359858.2619795073, 4062111.8209849824), (359858.2619795073, 4062101.8209849824), (359858.2619795073, 4062091.8209849824), (359858.2619795073, 4062081.8209849824), (359858.2619795073, 4062071.8209849824), (359858.2619795073, 4062061.8209849824), (359858.2619795073, 4062051.8209849824), (359868.2619795073, 4065331.8209849824), (359868.2619795073, 4065321.8209849824), (359868.2619795073, 4065311.8209849824), (359868.2619795073, 4065301.8209849824), (359868.2619795073, 4065291.8209849824), (359868.2619795073, 4065281.8209849824), (359868.2619795073, 4065271.8209849824), (359868.2619795073, 4065261.8209849824), (359868.2619795073, 4065251.8209849824), (359868.2619795073, 4065241.8209849824), (359868.2619795073, 4065231.8209849824), (359868.2619795073, 4065221.8209849824), (359868.2619795073, 
4065211.8209849824), (359868.2619795073, 4065201.8209849824), (359868.2619795073, 4065191.8209849824), (359868.2619795073, 4065181.8209849824), (359868.2619795073, 4065171.8209849824), (359868.2619795073, 4065161.8209849824), (359868.2619795073, 4065151.8209849824), (359868.2619795073, 4065141.8209849824), (359868.2619795073, 4065131.8209849824), (359868.2619795073, 4065121.8209849824), (359868.2619795073, 4065111.8209849824), (359868.2619795073, 4065101.8209849824), (359868.2619795073, 4065091.8209849824), (359868.2619795073, 4065081.8209849824), (359868.2619795073, 4065071.8209849824), (359868.2619795073, 4065061.8209849824), (359868.2619795073, 4065051.8209849824), (359868.2619795073, 4065041.8209849824), (359868.2619795073, 4065031.8209849824), (359868.2619795073, 4065021.8209849824), (359868.2619795073, 4065011.8209849824), (359868.2619795073, 4065001.8209849824), (359868.2619795073, 4064991.8209849824), (359868.2619795073, 4064981.8209849824), (359868.2619795073, 4064971.8209849824), (359868.2619795073, 4064961.8209849824), (359868.2619795073, 4064951.8209849824), (359868.2619795073, 4064941.8209849824), (359868.2619795073, 4064931.8209849824), (359868.2619795073, 4064921.8209849824), (359868.2619795073, 4064911.8209849824), (359868.2619795073, 4064901.8209849824), (359868.2619795073, 4064891.8209849824), (359868.2619795073, 4064881.8209849824), (359868.2619795073, 4064871.8209849824), (359868.2619795073, 4064861.8209849824), (359868.2619795073, 4064851.8209849824), (359868.2619795073, 4064841.8209849824), (359868.2619795073, 4064831.8209849824), (359868.2619795073, 4064821.8209849824), (359868.2619795073, 4064811.8209849824), (359868.2619795073, 4064801.8209849824), (359868.2619795073, 4064791.8209849824), (359868.2619795073, 4064781.8209849824), (359868.2619795073, 4064771.8209849824), (359868.2619795073, 4064761.8209849824), (359868.2619795073, 4064751.8209849824), (359868.2619795073, 4064741.8209849824), (359868.2619795073, 4064731.8209849824), (359868.2619795073, 4064721.8209849824), (359868.2619795073, 4064711.8209849824), (359868.2619795073, 4064701.8209849824), (359868.2619795073, 4064691.8209849824), (359868.2619795073, 4064681.8209849824), (359868.2619795073, 4064671.8209849824), (359868.2619795073, 4064661.8209849824), (359868.2619795073, 4064651.8209849824), (359868.2619795073, 4064641.8209849824), (359868.2619795073, 4064631.8209849824), (359868.2619795073, 4064621.8209849824), (359868.2619795073, 4064611.8209849824), (359868.2619795073, 4064601.8209849824), (359868.2619795073, 4064591.8209849824), (359868.2619795073, 4064581.8209849824), (359868.2619795073, 4064571.8209849824), (359868.2619795073, 4064561.8209849824), (359868.2619795073, 4064551.8209849824), (359868.2619795073, 4064541.8209849824), (359868.2619795073, 4064531.8209849824), (359868.2619795073, 4064521.8209849824), 
(359868.2619795073, 4064511.8209849824), (359868.2619795073, 4064501.8209849824), (359868.2619795073, 4064491.8209849824), (359868.2619795073, 4064481.8209849824), (359868.2619795073, 4064471.8209849824), (359868.2619795073, 4064461.8209849824), (359868.2619795073, 4064451.8209849824), (359868.2619795073, 4064441.8209849824), (359868.2619795073, 4064431.8209849824), (359868.2619795073, 4064421.8209849824), (359868.2619795073, 4064411.8209849824), (359868.2619795073, 4064401.8209849824), (359868.2619795073, 4064391.8209849824), (359868.2619795073, 4064381.8209849824), (359868.2619795073, 4064371.8209849824), (359868.2619795073, 4064361.8209849824), (359868.2619795073, 4064351.8209849824), (359868.2619795073, 4064341.8209849824), (359868.2619795073, 4064331.8209849824), (359868.2619795073, 4064321.8209849824), (359868.2619795073, 4064311.8209849824), (359868.2619795073, 4064301.8209849824), (359868.2619795073, 4064291.8209849824), (359868.2619795073, 4064281.8209849824), (359868.2619795073, 4064271.8209849824), (359868.2619795073, 4064261.8209849824), (359868.2619795073, 4064251.8209849824), (359868.2619795073, 4064241.8209849824), (359868.2619795073, 4064231.8209849824), (359868.2619795073, 4064221.8209849824), (359868.2619795073, 4064211.8209849824), (359868.2619795073, 4064201.8209849824), (359868.2619795073, 4064191.8209849824), (359868.2619795073, 4064181.8209849824), (359868.2619795073, 4064171.8209849824), (359868.2619795073, 4064161.8209849824), (359868.2619795073, 4064151.8209849824), (359868.2619795073, 4064141.8209849824), (359868.2619795073, 4064131.8209849824), (359868.2619795073, 4064121.8209849824), (359868.2619795073, 4064111.8209849824), (359868.2619795073, 4064101.8209849824), (359868.2619795073, 4064091.8209849824), (359868.2619795073, 4064081.8209849824), (359868.2619795073, 4064071.8209849824), (359868.2619795073, 4064061.8209849824), (359868.2619795073, 4064051.8209849824), (359868.2619795073, 4064041.8209849824), (359868.2619795073, 4064031.8209849824), (359868.2619795073, 4064021.8209849824), (359868.2619795073, 4064011.8209849824), (359868.2619795073, 4064001.8209849824), (359868.2619795073, 4063991.8209849824), (359868.2619795073, 4063981.8209849824), (359868.2619795073, 4063971.8209849824), (359868.2619795073, 4063961.8209849824), (359868.2619795073, 4063951.8209849824), (359868.2619795073, 4063941.8209849824), (359868.2619795073, 4063931.8209849824), (359868.2619795073, 4063921.8209849824), (359868.2619795073, 4063911.8209849824), (359868.2619795073, 4063901.8209849824), (359868.2619795073, 4063891.8209849824), (359868.2619795073, 4063881.8209849824), (359868.2619795073, 4063871.8209849824), (359868.2619795073, 4063861.8209849824), (359868.2619795073, 4063851.8209849824), (359868.2619795073, 4063841.8209849824), (359868.2619795073, 4063831.8209849824), (359868.2619795073, 4063821.8209849824), (359868.2619795073, 4063811.8209849824), (359868.2619795073, 4063801.8209849824), (359868.2619795073, 4063791.8209849824), (359868.2619795073, 4063781.8209849824), (359868.2619795073, 4063771.8209849824), (359868.2619795073, 4063761.8209849824), (359868.2619795073, 4063751.8209849824), (359868.2619795073, 4063741.8209849824), (359868.2619795073, 4063731.8209849824), (359868.2619795073, 4063721.8209849824), (359868.2619795073, 
4063711.8209849824), (359868.2619795073, 4063701.8209849824), (359868.2619795073, 4063691.8209849824), (359868.2619795073, 4063681.8209849824), (359868.2619795073, 4063671.8209849824), (359868.2619795073, 4063661.8209849824), (359868.2619795073, 4063651.8209849824), (359868.2619795073, 4063641.8209849824), (359868.2619795073, 4063631.8209849824), (359868.2619795073, 4063621.8209849824), (359868.2619795073, 4063611.8209849824), (359868.2619795073, 4063601.8209849824), (359868.2619795073, 4063591.8209849824), (359868.2619795073, 4063581.8209849824), (359868.2619795073, 4063571.8209849824), (359868.2619795073, 4063561.8209849824), (359868.2619795073, 4063551.8209849824), (359868.2619795073, 4063541.8209849824), (359868.2619795073, 4063531.8209849824), (359868.2619795073, 4063521.8209849824), (359868.2619795073, 4063511.8209849824), (359868.2619795073, 4063501.8209849824), (359868.2619795073, 4063491.8209849824), (359868.2619795073, 4063481.8209849824), (359868.2619795073, 4063471.8209849824), (359868.2619795073, 4063461.8209849824), (359868.2619795073, 4063451.8209849824), (359868.2619795073, 4063441.8209849824), (359868.2619795073, 4063431.8209849824), (359868.2619795073, 4063421.8209849824), (359868.2619795073, 4063411.8209849824), (359868.2619795073, 4063401.8209849824), (359868.2619795073, 4063391.8209849824), (359868.2619795073, 4063381.8209849824), (359868.2619795073, 4063371.8209849824), (359868.2619795073, 4063361.8209849824), (359868.2619795073, 4063351.8209849824), (359868.2619795073, 4063341.8209849824), (359868.2619795073, 4063331.8209849824), (359868.2619795073, 4063321.8209849824), (359868.2619795073, 4063311.8209849824), (359868.2619795073, 4063301.8209849824), (359868.2619795073, 4063291.8209849824), (359868.2619795073, 4063281.8209849824), (359868.2619795073, 4063271.8209849824), (359868.2619795073, 4063261.8209849824), (359868.2619795073, 4063251.8209849824), (359868.2619795073, 4063241.8209849824), (359868.2619795073, 4063231.8209849824), (359868.2619795073, 4063221.8209849824), (359868.2619795073, 4063211.8209849824), (359868.2619795073, 4063201.8209849824), (359868.2619795073, 4063191.8209849824), (359868.2619795073, 4063181.8209849824), (359868.2619795073, 4063171.8209849824), (359868.2619795073, 4063161.8209849824), (359868.2619795073, 4063151.8209849824), (359868.2619795073, 4063141.8209849824), (359868.2619795073, 4063131.8209849824), (359868.2619795073, 4063121.8209849824), (359868.2619795073, 4063101.8209849824), (359868.2619795073, 4063091.8209849824), (359868.2619795073, 4063081.8209849824), (359868.2619795073, 4063071.8209849824), (359868.2619795073, 4063061.8209849824), (359868.2619795073, 4063051.8209849824), (359868.2619795073, 4063041.8209849824), (359868.2619795073, 4063031.8209849824), (359868.2619795073, 4063021.8209849824), (359868.2619795073, 4063011.8209849824), 
(359868.2619795073, 4063001.8209849824), (359868.2619795073, 4062991.8209849824), (359868.2619795073, 4062981.8209849824), (359868.2619795073, 4062971.8209849824), (359868.2619795073, 4062961.8209849824), (359868.2619795073, 4062951.8209849824), (359868.2619795073, 4062941.8209849824), (359868.2619795073, 4062931.8209849824), (359868.2619795073, 4062921.8209849824), (359868.2619795073, 4062911.8209849824), (359868.2619795073, 4062901.8209849824), (359868.2619795073, 4062891.8209849824), (359868.2619795073, 4062881.8209849824), (359868.2619795073, 4062871.8209849824), (359868.2619795073, 4062861.8209849824), (359868.2619795073, 4062851.8209849824), (359868.2619795073, 4062841.8209849824), (359868.2619795073, 4062831.8209849824), (359868.2619795073, 4062821.8209849824), (359868.2619795073, 4062811.8209849824), (359868.2619795073, 4062801.8209849824), (359868.2619795073, 4062791.8209849824), (359868.2619795073, 4062781.8209849824), (359868.2619795073, 4062771.8209849824), (359868.2619795073, 4062761.8209849824), (359868.2619795073, 4062751.8209849824), (359868.2619795073, 4062741.8209849824), (359868.2619795073, 4062731.8209849824), (359868.2619795073, 4062721.8209849824), (359868.2619795073, 4062711.8209849824), (359868.2619795073, 4062701.8209849824), (359868.2619795073, 4062691.8209849824), (359868.2619795073, 4062681.8209849824), (359868.2619795073, 4062671.8209849824), (359868.2619795073, 4062661.8209849824), (359868.2619795073, 4062651.8209849824), (359868.2619795073, 4062641.8209849824), (359868.2619795073, 4062631.8209849824), (359868.2619795073, 4062621.8209849824), (359868.2619795073, 4062611.8209849824), (359868.2619795073, 4062601.8209849824), (359868.2619795073, 4062591.8209849824), (359868.2619795073, 4062581.8209849824), (359868.2619795073, 4062571.8209849824), (359868.2619795073, 4062561.8209849824), (359868.2619795073, 4062551.8209849824), (359868.2619795073, 4062541.8209849824), (359868.2619795073, 4062531.8209849824), (359868.2619795073, 4062521.8209849824), (359868.2619795073, 4062511.8209849824), (359868.2619795073, 4062501.8209849824), (359868.2619795073, 4062491.8209849824), (359868.2619795073, 4062481.8209849824), (359868.2619795073, 4062471.8209849824), (359868.2619795073, 4062461.8209849824), (359868.2619795073, 4062451.8209849824), (359868.2619795073, 4062441.8209849824), (359868.2619795073, 4062431.8209849824), (359868.2619795073, 4062421.8209849824), (359868.2619795073, 4062411.8209849824), (359868.2619795073, 4062401.8209849824), (359868.2619795073, 4062391.8209849824), (359868.2619795073, 4062381.8209849824), (359868.2619795073, 4062371.8209849824), (359868.2619795073, 4062361.8209849824), (359868.2619795073, 4062351.8209849824), (359868.2619795073, 4062341.8209849824), (359868.2619795073, 4062331.8209849824), (359868.2619795073, 4062321.8209849824), (359868.2619795073, 4062311.8209849824), (359868.2619795073, 4062301.8209849824), (359868.2619795073, 4062291.8209849824), (359868.2619795073, 4062281.8209849824), (359868.2619795073, 4062271.8209849824), (359868.2619795073, 4062261.8209849824), (359868.2619795073, 4062251.8209849824), (359868.2619795073, 4062241.8209849824), (359868.2619795073, 4062231.8209849824), (359868.2619795073, 4062221.8209849824), (359868.2619795073, 4062211.8209849824), (359868.2619795073, 
4062201.8209849824), (359868.2619795073, 4062191.8209849824), (359868.2619795073, 4062181.8209849824), (359868.2619795073, 4062171.8209849824), (359868.2619795073, 4062161.8209849824), (359868.2619795073, 4062151.8209849824), (359868.2619795073, 4062141.8209849824), (359868.2619795073, 4062131.8209849824), (359868.2619795073, 4062121.8209849824), (359868.2619795073, 4062111.8209849824), (359868.2619795073, 4062101.8209849824), (359868.2619795073, 4062091.8209849824), (359868.2619795073, 4062081.8209849824), (359868.2619795073, 4062071.8209849824), (359868.2619795073, 4062061.8209849824), (359868.2619795073, 4062051.8209849824), (359878.2619795073, 4065341.8209849824), (359878.2619795073, 4065331.8209849824), (359878.2619795073, 4065321.8209849824), (359878.2619795073, 4065311.8209849824), (359878.2619795073, 4065301.8209849824), (359878.2619795073, 4065291.8209849824), (359878.2619795073, 4065281.8209849824), (359878.2619795073, 4065271.8209849824), (359878.2619795073, 4065261.8209849824), (359878.2619795073, 4065251.8209849824), (359878.2619795073, 4065241.8209849824), (359878.2619795073, 4065231.8209849824), (359878.2619795073, 4065221.8209849824), (359878.2619795073, 4065211.8209849824), (359878.2619795073, 4065201.8209849824), (359878.2619795073, 4065191.8209849824), (359878.2619795073, 4065181.8209849824), (359878.2619795073, 4065171.8209849824), (359878.2619795073, 4065161.8209849824), (359878.2619795073, 4065151.8209849824), (359878.2619795073, 4065141.8209849824), (359878.2619795073, 4065131.8209849824), (359878.2619795073, 4065121.8209849824), (359878.2619795073, 4065111.8209849824), (359878.2619795073, 4065101.8209849824), (359878.2619795073, 4065091.8209849824), (359878.2619795073, 4065081.8209849824), (359878.2619795073, 4065071.8209849824), (359878.2619795073, 4065061.8209849824), (359878.2619795073, 4065051.8209849824), (359878.2619795073, 4065041.8209849824), (359878.2619795073, 4065031.8209849824), (359878.2619795073, 4065021.8209849824), (359878.2619795073, 4065011.8209849824), (359878.2619795073, 4065001.8209849824), (359878.2619795073, 4064991.8209849824), (359878.2619795073, 4064981.8209849824), (359878.2619795073, 4064971.8209849824), (359878.2619795073, 4064961.8209849824), (359878.2619795073, 4064951.8209849824), (359878.2619795073, 4064941.8209849824), (359878.2619795073, 4064931.8209849824), (359878.2619795073, 4064921.8209849824), (359878.2619795073, 4064911.8209849824), (359878.2619795073, 4064901.8209849824), (359878.2619795073, 4064891.8209849824), (359878.2619795073, 4064881.8209849824), (359878.2619795073, 4064871.8209849824), (359878.2619795073, 4064861.8209849824), (359878.2619795073, 4064851.8209849824), (359878.2619795073, 4064841.8209849824), (359878.2619795073, 4064831.8209849824), (359878.2619795073, 4064821.8209849824), (359878.2619795073, 4064811.8209849824), 
(359878.2619795073, 4064801.8209849824), (359878.2619795073, 4064791.8209849824), (359878.2619795073, 4064781.8209849824), (359878.2619795073, 4064771.8209849824), (359878.2619795073, 4064761.8209849824), (359878.2619795073, 4064751.8209849824), (359878.2619795073, 4064741.8209849824), (359878.2619795073, 4064731.8209849824), (359878.2619795073, 4064721.8209849824), (359878.2619795073, 4064711.8209849824), (359878.2619795073, 4064701.8209849824), (359878.2619795073, 4064691.8209849824), (359878.2619795073, 4064681.8209849824), (359878.2619795073, 4064671.8209849824), (359878.2619795073, 4064661.8209849824), (359878.2619795073, 4064651.8209849824), (359878.2619795073, 4064641.8209849824), (359878.2619795073, 4064631.8209849824), (359878.2619795073, 4064621.8209849824), (359878.2619795073, 4064611.8209849824), (359878.2619795073, 4064601.8209849824), (359878.2619795073, 4064591.8209849824), (359878.2619795073, 4064581.8209849824), (359878.2619795073, 4064571.8209849824), (359878.2619795073, 4064561.8209849824), (359878.2619795073, 4064551.8209849824), (359878.2619795073, 4064541.8209849824), (359878.2619795073, 4064531.8209849824), (359878.2619795073, 4064521.8209849824), (359878.2619795073, 4064511.8209849824), (359878.2619795073, 4064501.8209849824), (359878.2619795073, 4064491.8209849824), (359878.2619795073, 4064481.8209849824), (359878.2619795073, 4064471.8209849824), (359878.2619795073, 4064461.8209849824), (359878.2619795073, 4064451.8209849824), (359878.2619795073, 4064441.8209849824), (359878.2619795073, 4064431.8209849824), (359878.2619795073, 4064421.8209849824), (359878.2619795073, 4064411.8209849824), (359878.2619795073, 4064401.8209849824), (359878.2619795073, 4064391.8209849824), (359878.2619795073, 4064381.8209849824), (359878.2619795073, 4064371.8209849824), (359878.2619795073, 4064361.8209849824), (359878.2619795073, 4064351.8209849824), (359878.2619795073, 4064341.8209849824), (359878.2619795073, 4064331.8209849824), (359878.2619795073, 4064321.8209849824), (359878.2619795073, 4064311.8209849824), (359878.2619795073, 4064301.8209849824), (359878.2619795073, 4064291.8209849824), (359878.2619795073, 4064281.8209849824), (359878.2619795073, 4064271.8209849824), (359878.2619795073, 4064261.8209849824), (359878.2619795073, 4064251.8209849824), (359878.2619795073, 4064241.8209849824), (359878.2619795073, 4064231.8209849824), (359878.2619795073, 4064221.8209849824), (359878.2619795073, 4064211.8209849824), (359878.2619795073, 4064201.8209849824), (359878.2619795073, 4064191.8209849824), (359878.2619795073, 4064181.8209849824), (359878.2619795073, 4064171.8209849824), (359878.2619795073, 4064161.8209849824), (359878.2619795073, 4064151.8209849824), (359878.2619795073, 4064141.8209849824), (359878.2619795073, 4064131.8209849824), (359878.2619795073, 4064121.8209849824), (359878.2619795073, 4064111.8209849824), (359878.2619795073, 4064101.8209849824), (359878.2619795073, 4064091.8209849824), (359878.2619795073, 4064081.8209849824), (359878.2619795073, 4064071.8209849824), (359878.2619795073, 4064061.8209849824), (359878.2619795073, 4064051.8209849824), (359878.2619795073, 4064041.8209849824), (359878.2619795073, 4064031.8209849824), (359878.2619795073, 4064021.8209849824), (359878.2619795073, 4064011.8209849824), (359878.2619795073, 
4064001.8209849824), (359878.2619795073, 4063991.8209849824), (359878.2619795073, 4063981.8209849824), (359878.2619795073, 4063971.8209849824), (359878.2619795073, 4063961.8209849824), (359878.2619795073, 4063951.8209849824), (359878.2619795073, 4063941.8209849824), (359878.2619795073, 4063931.8209849824), (359878.2619795073, 4063921.8209849824), (359878.2619795073, 4063911.8209849824), (359878.2619795073, 4063901.8209849824), (359878.2619795073, 4063891.8209849824), (359878.2619795073, 4063881.8209849824), (359878.2619795073, 4063871.8209849824), (359878.2619795073, 4063861.8209849824), (359878.2619795073, 4063851.8209849824), (359878.2619795073, 4063841.8209849824), (359878.2619795073, 4063831.8209849824), (359878.2619795073, 4063821.8209849824), (359878.2619795073, 4063811.8209849824), (359878.2619795073, 4063801.8209849824), (359878.2619795073, 4063791.8209849824), (359878.2619795073, 4063781.8209849824), (359878.2619795073, 4063771.8209849824), (359878.2619795073, 4063761.8209849824), (359878.2619795073, 4063751.8209849824), (359878.2619795073, 4063741.8209849824), (359878.2619795073, 4063731.8209849824), (359878.2619795073, 4063721.8209849824), (359878.2619795073, 4063711.8209849824), (359878.2619795073, 4063701.8209849824), (359878.2619795073, 4063691.8209849824), (359878.2619795073, 4063681.8209849824), (359878.2619795073, 4063671.8209849824), (359878.2619795073, 4063661.8209849824), (359878.2619795073, 4063651.8209849824), (359878.2619795073, 4063641.8209849824), (359878.2619795073, 4063631.8209849824), (359878.2619795073, 4063621.8209849824), (359878.2619795073, 4063611.8209849824), (359878.2619795073, 4063601.8209849824), (359878.2619795073, 4063591.8209849824), (359878.2619795073, 4063581.8209849824), (359878.2619795073, 4063571.8209849824), (359878.2619795073, 4063561.8209849824), (359878.2619795073, 4063551.8209849824), (359878.2619795073, 4063541.8209849824), (359878.2619795073, 4063531.8209849824), (359878.2619795073, 4063521.8209849824), (359878.2619795073, 4063511.8209849824), (359878.2619795073, 4063501.8209849824), (359878.2619795073, 4063491.8209849824), (359878.2619795073, 4063481.8209849824), (359878.2619795073, 4063471.8209849824), (359878.2619795073, 4063461.8209849824), (359878.2619795073, 4063451.8209849824), (359878.2619795073, 4063441.8209849824), (359878.2619795073, 4063431.8209849824), (359878.2619795073, 4063421.8209849824), (359878.2619795073, 4063411.8209849824), (359878.2619795073, 4063401.8209849824), (359878.2619795073, 4063391.8209849824), (359878.2619795073, 4063381.8209849824), (359878.2619795073, 4063371.8209849824), (359878.2619795073, 4063361.8209849824), (359878.2619795073, 4063351.8209849824), (359878.2619795073, 4063341.8209849824), (359878.2619795073, 4063331.8209849824), (359878.2619795073, 4063321.8209849824), (359878.2619795073, 4063311.8209849824), 
(359878.2619795073, 4063301.8209849824), (359878.2619795073, 4063291.8209849824), (359878.2619795073, 4063281.8209849824), (359878.2619795073, 4063271.8209849824), (359878.2619795073, 4063261.8209849824), (359878.2619795073, 4063251.8209849824), (359878.2619795073, 4063241.8209849824), (359878.2619795073, 4063231.8209849824), (359878.2619795073, 4063221.8209849824), (359878.2619795073, 4063211.8209849824), (359878.2619795073, 4063201.8209849824), (359878.2619795073, 4063191.8209849824), (359878.2619795073, 4063181.8209849824), (359878.2619795073, 4063171.8209849824), (359878.2619795073, 4063161.8209849824), (359878.2619795073, 4063151.8209849824), (359878.2619795073, 4063141.8209849824), (359878.2619795073, 4063131.8209849824), (359878.2619795073, 4063121.8209849824), (359878.2619795073, 4063111.8209849824), (359878.2619795073, 4063101.8209849824), (359878.2619795073, 4063091.8209849824), (359878.2619795073, 4063081.8209849824), (359878.2619795073, 4063071.8209849824), (359878.2619795073, 4063061.8209849824), (359878.2619795073, 4063051.8209849824), (359878.2619795073, 4063041.8209849824), (359878.2619795073, 4063031.8209849824), (359878.2619795073, 4063021.8209849824), (359878.2619795073, 4063011.8209849824), (359878.2619795073, 4063001.8209849824), (359878.2619795073, 4062991.8209849824), (359878.2619795073, 4062981.8209849824), (359878.2619795073, 4062971.8209849824), (359878.2619795073, 4062961.8209849824), (359878.2619795073, 4062951.8209849824), (359878.2619795073, 4062941.8209849824), (359878.2619795073, 4062931.8209849824), (359878.2619795073, 4062921.8209849824), (359878.2619795073, 4062911.8209849824), (359878.2619795073, 4062901.8209849824), (359878.2619795073, 4062891.8209849824), (359878.2619795073, 4062881.8209849824), (359878.2619795073, 4062871.8209849824), (359878.2619795073, 4062861.8209849824), (359878.2619795073, 4062851.8209849824), (359878.2619795073, 4062841.8209849824), (359878.2619795073, 4062831.8209849824), (359878.2619795073, 4062821.8209849824), (359878.2619795073, 4062811.8209849824), (359878.2619795073, 4062801.8209849824), (359878.2619795073, 4062791.8209849824), (359878.2619795073, 4062781.8209849824), (359878.2619795073, 4062771.8209849824), (359878.2619795073, 4062761.8209849824), (359878.2619795073, 4062751.8209849824), (359878.2619795073, 4062741.8209849824), (359878.2619795073, 4062731.8209849824), (359878.2619795073, 4062721.8209849824), (359878.2619795073, 4062711.8209849824), (359878.2619795073, 4062701.8209849824), (359878.2619795073, 4062691.8209849824), (359878.2619795073, 4062681.8209849824), (359878.2619795073, 4062671.8209849824), (359878.2619795073, 4062661.8209849824), (359878.2619795073, 4062651.8209849824), (359878.2619795073, 4062641.8209849824), (359878.2619795073, 4062631.8209849824), (359878.2619795073, 4062621.8209849824), (359878.2619795073, 4062611.8209849824), (359878.2619795073, 4062601.8209849824), (359878.2619795073, 4062591.8209849824), (359878.2619795073, 4062581.8209849824), (359878.2619795073, 4062571.8209849824), (359878.2619795073, 4062561.8209849824), (359878.2619795073, 4062551.8209849824), (359878.2619795073, 4062541.8209849824), (359878.2619795073, 4062531.8209849824), (359878.2619795073, 4062521.8209849824), (359878.2619795073, 4062511.8209849824), (359878.2619795073, 
4062501.8209849824), (359878.2619795073, 4062491.8209849824), (359878.2619795073, 4062481.8209849824), (359878.2619795073, 4062471.8209849824), (359878.2619795073, 4062461.8209849824), (359878.2619795073, 4062451.8209849824), (359878.2619795073, 4062441.8209849824), (359878.2619795073, 4062431.8209849824), (359878.2619795073, 4062421.8209849824), (359878.2619795073, 4062411.8209849824), (359878.2619795073, 4062401.8209849824), (359878.2619795073, 4062391.8209849824), (359878.2619795073, 4062381.8209849824), (359878.2619795073, 4062371.8209849824), (359878.2619795073, 4062361.8209849824), (359878.2619795073, 4062351.8209849824), (359878.2619795073, 4062341.8209849824), (359878.2619795073, 4062331.8209849824), (359878.2619795073, 4062321.8209849824), (359878.2619795073, 4062311.8209849824), (359878.2619795073, 4062301.8209849824), (359878.2619795073, 4062291.8209849824), (359878.2619795073, 4062281.8209849824), (359878.2619795073, 4062271.8209849824), (359878.2619795073, 4062261.8209849824), (359878.2619795073, 4062251.8209849824), (359878.2619795073, 4062241.8209849824), (359878.2619795073, 4062231.8209849824), (359878.2619795073, 4062221.8209849824), (359878.2619795073, 4062211.8209849824), (359878.2619795073, 4062201.8209849824), (359878.2619795073, 4062191.8209849824), (359878.2619795073, 4062181.8209849824), (359878.2619795073, 4062171.8209849824), (359878.2619795073, 4062161.8209849824), (359878.2619795073, 4062151.8209849824), (359878.2619795073, 4062141.8209849824), (359878.2619795073, 4062131.8209849824), (359878.2619795073, 4062121.8209849824), (359878.2619795073, 4062111.8209849824), (359878.2619795073, 4062101.8209849824), (359878.2619795073, 4062091.8209849824), (359878.2619795073, 4062081.8209849824), (359878.2619795073, 4062071.8209849824), (359878.2619795073, 4062061.8209849824), (359878.2619795073, 4062051.8209849824), (359888.2619795073, 4065341.8209849824), (359888.2619795073, 4065331.8209849824), (359888.2619795073, 4065321.8209849824), (359888.2619795073, 4065311.8209849824), (359888.2619795073, 4065301.8209849824), (359888.2619795073, 4065291.8209849824), (359888.2619795073, 4065281.8209849824), (359888.2619795073, 4065271.8209849824), (359888.2619795073, 4065261.8209849824), (359888.2619795073, 4065251.8209849824), (359888.2619795073, 4065241.8209849824), (359888.2619795073, 4065231.8209849824), (359888.2619795073, 4065221.8209849824), (359888.2619795073, 4065211.8209849824), (359888.2619795073, 4065201.8209849824), (359888.2619795073, 4065191.8209849824), (359888.2619795073, 4065181.8209849824), (359888.2619795073, 4065171.8209849824), (359888.2619795073, 4065161.8209849824), (359888.2619795073, 4065151.8209849824), (359888.2619795073, 4065141.8209849824), (359888.2619795073, 4065131.8209849824), (359888.2619795073, 4065121.8209849824), (359888.2619795073, 4065111.8209849824), 
(359888.2619795073, 4065101.8209849824), (359888.2619795073, 4065091.8209849824), (359888.2619795073, 4065081.8209849824), (359888.2619795073, 4065071.8209849824), (359888.2619795073, 4065061.8209849824), (359888.2619795073, 4065051.8209849824), (359888.2619795073, 4065041.8209849824), (359888.2619795073, 4065031.8209849824), (359888.2619795073, 4065021.8209849824), (359888.2619795073, 4065011.8209849824), (359888.2619795073, 4065001.8209849824), (359888.2619795073, 4064991.8209849824), (359888.2619795073, 4064981.8209849824), (359888.2619795073, 4064971.8209849824), (359888.2619795073, 4064961.8209849824), (359888.2619795073, 4064951.8209849824), (359888.2619795073, 4064941.8209849824), (359888.2619795073, 4064931.8209849824), (359888.2619795073, 4064921.8209849824), (359888.2619795073, 4064911.8209849824), (359888.2619795073, 4064901.8209849824), (359888.2619795073, 4064891.8209849824), (359888.2619795073, 4064881.8209849824), (359888.2619795073, 4064871.8209849824), (359888.2619795073, 4064861.8209849824), (359888.2619795073, 4064851.8209849824), (359888.2619795073, 4064841.8209849824), (359888.2619795073, 4064831.8209849824), (359888.2619795073, 4064821.8209849824), (359888.2619795073, 4064811.8209849824), (359888.2619795073, 4064801.8209849824), (359888.2619795073, 4064791.8209849824), (359888.2619795073, 4064781.8209849824), (359888.2619795073, 4064771.8209849824), (359888.2619795073, 4064761.8209849824), (359888.2619795073, 4064751.8209849824), (359888.2619795073, 4064741.8209849824), (359888.2619795073, 4064731.8209849824), (359888.2619795073, 4064721.8209849824), (359888.2619795073, 4064711.8209849824), (359888.2619795073, 4064701.8209849824), (359888.2619795073, 4064691.8209849824), (359888.2619795073, 4064681.8209849824), (359888.2619795073, 4064671.8209849824), (359888.2619795073, 4064661.8209849824), (359888.2619795073, 4064651.8209849824), (359888.2619795073, 4064641.8209849824), (359888.2619795073, 4064631.8209849824), (359888.2619795073, 4064621.8209849824), (359888.2619795073, 4064611.8209849824), (359888.2619795073, 4064601.8209849824), (359888.2619795073, 4064591.8209849824), (359888.2619795073, 4064581.8209849824), (359888.2619795073, 4064571.8209849824), (359888.2619795073, 4064561.8209849824), (359888.2619795073, 4064551.8209849824), (359888.2619795073, 4064541.8209849824), (359888.2619795073, 4064531.8209849824), (359888.2619795073, 4064521.8209849824), (359888.2619795073, 4064511.8209849824), (359888.2619795073, 4064501.8209849824), (359888.2619795073, 4064491.8209849824), (359888.2619795073, 4064481.8209849824), (359888.2619795073, 4064471.8209849824), (359888.2619795073, 4064461.8209849824), (359888.2619795073, 4064451.8209849824), (359888.2619795073, 4064441.8209849824), (359888.2619795073, 4064431.8209849824), (359888.2619795073, 4064421.8209849824), (359888.2619795073, 4064411.8209849824), (359888.2619795073, 4064401.8209849824), (359888.2619795073, 4064391.8209849824), (359888.2619795073, 4064381.8209849824), (359888.2619795073, 4064371.8209849824), (359888.2619795073, 4064361.8209849824), (359888.2619795073, 4064351.8209849824), (359888.2619795073, 4064341.8209849824), (359888.2619795073, 4064331.8209849824), (359888.2619795073, 4064321.8209849824), (359888.2619795073, 4064311.8209849824), (359888.2619795073, 
4064301.8209849824), (359888.2619795073, 4064291.8209849824), (359888.2619795073, 4064281.8209849824), (359888.2619795073, 4064271.8209849824), (359888.2619795073, 4064261.8209849824), (359888.2619795073, 4064251.8209849824), (359888.2619795073, 4064241.8209849824), (359888.2619795073, 4064231.8209849824), (359888.2619795073, 4064221.8209849824), (359888.2619795073, 4064211.8209849824), (359888.2619795073, 4064201.8209849824), (359888.2619795073, 4064191.8209849824), (359888.2619795073, 4064181.8209849824), (359888.2619795073, 4064171.8209849824), (359888.2619795073, 4064161.8209849824), (359888.2619795073, 4064151.8209849824), (359888.2619795073, 4064141.8209849824), (359888.2619795073, 4064131.8209849824), (359888.2619795073, 4064121.8209849824), (359888.2619795073, 4064111.8209849824), (359888.2619795073, 4064101.8209849824), (359888.2619795073, 4064091.8209849824), (359888.2619795073, 4064081.8209849824), (359888.2619795073, 4064071.8209849824), (359888.2619795073, 4064061.8209849824), (359888.2619795073, 4064051.8209849824), (359888.2619795073, 4064041.8209849824), (359888.2619795073, 4064031.8209849824), (359888.2619795073, 4064021.8209849824), (359888.2619795073, 4064011.8209849824), (359888.2619795073, 4064001.8209849824), (359888.2619795073, 4063991.8209849824), (359888.2619795073, 4063981.8209849824), (359888.2619795073, 4063971.8209849824), (359888.2619795073, 4063961.8209849824), (359888.2619795073, 4063951.8209849824), (359888.2619795073, 4063941.8209849824), (359888.2619795073, 4063931.8209849824), (359888.2619795073, 4063921.8209849824), (359888.2619795073, 4063911.8209849824), (359888.2619795073, 4063901.8209849824), (359888.2619795073, 4063891.8209849824), (359888.2619795073, 4063881.8209849824), (359888.2619795073, 4063871.8209849824), (359888.2619795073, 4063861.8209849824), (359888.2619795073, 4063851.8209849824), (359888.2619795073, 4063841.8209849824), (359888.2619795073, 4063831.8209849824), (359888.2619795073, 4063821.8209849824), (359888.2619795073, 4063811.8209849824), (359888.2619795073, 4063801.8209849824), (359888.2619795073, 4063791.8209849824), (359888.2619795073, 4063781.8209849824), (359888.2619795073, 4063771.8209849824), (359888.2619795073, 4063761.8209849824), (359888.2619795073, 4063751.8209849824), (359888.2619795073, 4063741.8209849824), (359888.2619795073, 4063731.8209849824), (359888.2619795073, 4063721.8209849824), (359888.2619795073, 4063711.8209849824), (359888.2619795073, 4063701.8209849824), (359888.2619795073, 4063691.8209849824), (359888.2619795073, 4063681.8209849824), (359888.2619795073, 4063671.8209849824), (359888.2619795073, 4063661.8209849824), (359888.2619795073, 4063651.8209849824), (359888.2619795073, 4063641.8209849824), (359888.2619795073, 4063631.8209849824), (359888.2619795073, 4063621.8209849824), (359888.2619795073, 4063611.8209849824), 
(359888.2619795073, 4063601.8209849824), (359888.2619795073, 4063591.8209849824), (359888.2619795073, 4063581.8209849824), (359888.2619795073, 4063571.8209849824), (359888.2619795073, 4063561.8209849824), (359888.2619795073, 4063551.8209849824), (359888.2619795073, 4063541.8209849824), (359888.2619795073, 4063531.8209849824), (359888.2619795073, 4063521.8209849824), (359888.2619795073, 4063511.8209849824), (359888.2619795073, 4063501.8209849824), (359888.2619795073, 4063491.8209849824), (359888.2619795073, 4063481.8209849824), (359888.2619795073, 4063471.8209849824), (359888.2619795073, 4063461.8209849824), (359888.2619795073, 4063451.8209849824), (359888.2619795073, 4063441.8209849824), (359888.2619795073, 4063431.8209849824), (359888.2619795073, 4063421.8209849824), (359888.2619795073, 4063411.8209849824), (359888.2619795073, 4063401.8209849824), (359888.2619795073, 4063391.8209849824), (359888.2619795073, 4063381.8209849824), (359888.2619795073, 4063371.8209849824), (359888.2619795073, 4063361.8209849824), (359888.2619795073, 4063351.8209849824), (359888.2619795073, 4063341.8209849824), (359888.2619795073, 4063331.8209849824), (359888.2619795073, 4063321.8209849824), (359888.2619795073, 4063311.8209849824), (359888.2619795073, 4063301.8209849824), (359888.2619795073, 4063291.8209849824), (359888.2619795073, 4063281.8209849824), (359888.2619795073, 4063271.8209849824), (359888.2619795073, 4063261.8209849824), (359888.2619795073, 4063251.8209849824), (359888.2619795073, 4063241.8209849824), (359888.2619795073, 4063231.8209849824), (359888.2619795073, 4063221.8209849824), (359888.2619795073, 4063211.8209849824), (359888.2619795073, 4063201.8209849824), (359888.2619795073, 4063191.8209849824), (359888.2619795073, 4063181.8209849824), (359888.2619795073, 4063171.8209849824), (359888.2619795073, 4063161.8209849824), (359888.2619795073, 4063151.8209849824), (359888.2619795073, 4063141.8209849824), (359888.2619795073, 4063131.8209849824), (359888.2619795073, 4063121.8209849824), (359888.2619795073, 4063111.8209849824), (359888.2619795073, 4063101.8209849824), (359888.2619795073, 4063091.8209849824), (359888.2619795073, 4063081.8209849824), (359888.2619795073, 4063071.8209849824), (359888.2619795073, 4063061.8209849824), (359888.2619795073, 4063051.8209849824), (359888.2619795073, 4063041.8209849824), (359888.2619795073, 4063031.8209849824), (359888.2619795073, 4063021.8209849824), (359888.2619795073, 4063011.8209849824), (359888.2619795073, 4063001.8209849824), (359888.2619795073, 4062991.8209849824), (359888.2619795073, 4062981.8209849824), (359888.2619795073, 4062971.8209849824), (359888.2619795073, 4062961.8209849824), (359888.2619795073, 4062951.8209849824), (359888.2619795073, 4062941.8209849824), (359888.2619795073, 4062931.8209849824), (359888.2619795073, 4062921.8209849824), (359888.2619795073, 4062911.8209849824), (359888.2619795073, 4062901.8209849824), (359888.2619795073, 4062891.8209849824), (359888.2619795073, 4062881.8209849824), (359888.2619795073, 4062871.8209849824), (359888.2619795073, 4062861.8209849824), (359888.2619795073, 4062851.8209849824), (359888.2619795073, 4062841.8209849824), (359888.2619795073, 4062831.8209849824), (359888.2619795073, 4062821.8209849824), (359888.2619795073, 4062811.8209849824), (359888.2619795073, 
4062801.8209849824), (359888.2619795073, 4062791.8209849824), (359888.2619795073, 4062781.8209849824), (359888.2619795073, 4062771.8209849824), (359888.2619795073, 4062761.8209849824), (359888.2619795073, 4062751.8209849824), (359888.2619795073, 4062741.8209849824), (359888.2619795073, 4062731.8209849824), (359888.2619795073, 4062721.8209849824), (359888.2619795073, 4062711.8209849824), (359888.2619795073, 4062701.8209849824), (359888.2619795073, 4062691.8209849824), (359888.2619795073, 4062681.8209849824), (359888.2619795073, 4062671.8209849824), (359888.2619795073, 4062661.8209849824), (359888.2619795073, 4062651.8209849824), (359888.2619795073, 4062641.8209849824), (359888.2619795073, 4062631.8209849824), (359888.2619795073, 4062621.8209849824), (359888.2619795073, 4062611.8209849824), (359888.2619795073, 4062601.8209849824), (359888.2619795073, 4062591.8209849824), (359888.2619795073, 4062581.8209849824), (359888.2619795073, 4062571.8209849824), (359888.2619795073, 4062561.8209849824), (359888.2619795073, 4062551.8209849824), (359888.2619795073, 4062541.8209849824), (359888.2619795073, 4062531.8209849824), (359888.2619795073, 4062521.8209849824), (359888.2619795073, 4062511.8209849824), (359888.2619795073, 4062501.8209849824), (359888.2619795073, 4062491.8209849824), (359888.2619795073, 4062481.8209849824), (359888.2619795073, 4062471.8209849824), (359888.2619795073, 4062461.8209849824), (359888.2619795073, 4062451.8209849824), (359888.2619795073, 4062441.8209849824), (359888.2619795073, 4062431.8209849824), (359888.2619795073, 4062421.8209849824), (359888.2619795073, 4062411.8209849824), (359888.2619795073, 4062401.8209849824), (359888.2619795073, 4062391.8209849824), (359888.2619795073, 4062381.8209849824), (359888.2619795073, 4062371.8209849824), (359888.2619795073, 4062361.8209849824), (359888.2619795073, 4062351.8209849824), (359888.2619795073, 4062341.8209849824), (359888.2619795073, 4062331.8209849824), (359888.2619795073, 4062321.8209849824), (359888.2619795073, 4062311.8209849824), (359888.2619795073, 4062301.8209849824), (359888.2619795073, 4062291.8209849824), (359888.2619795073, 4062281.8209849824), (359888.2619795073, 4062271.8209849824), (359888.2619795073, 4062261.8209849824), (359888.2619795073, 4062251.8209849824), (359888.2619795073, 4062241.8209849824), (359888.2619795073, 4062231.8209849824), (359888.2619795073, 4062221.8209849824), (359888.2619795073, 4062211.8209849824), (359888.2619795073, 4062201.8209849824), (359888.2619795073, 4062191.8209849824), (359888.2619795073, 4062181.8209849824), (359888.2619795073, 4062171.8209849824), (359888.2619795073, 4062161.8209849824), (359888.2619795073, 4062151.8209849824), (359888.2619795073, 4062141.8209849824), (359888.2619795073, 4062131.8209849824), (359888.2619795073, 4062121.8209849824), (359888.2619795073, 4062111.8209849824), 
(359888.2619795073, 4062101.8209849824), (359888.2619795073, 4062091.8209849824), (359888.2619795073, 4062081.8209849824), (359888.2619795073, 4062071.8209849824), (359888.2619795073, 4062061.8209849824), (359888.2619795073, 4062051.8209849824), (359898.2619795073, 4065341.8209849824), (359898.2619795073, 4065331.8209849824), (359898.2619795073, 4065321.8209849824), (359898.2619795073, 4065311.8209849824), (359898.2619795073, 4065301.8209849824), (359898.2619795073, 4065291.8209849824), (359898.2619795073, 4065281.8209849824), (359898.2619795073, 4065271.8209849824), (359898.2619795073, 4065261.8209849824), (359898.2619795073, 4065251.8209849824), (359898.2619795073, 4065241.8209849824), (359898.2619795073, 4065231.8209849824), (359898.2619795073, 4065221.8209849824), (359898.2619795073, 4065211.8209849824), (359898.2619795073, 4065201.8209849824), (359898.2619795073, 4065191.8209849824), (359898.2619795073, 4065181.8209849824), (359898.2619795073, 4065171.8209849824), (359898.2619795073, 4065161.8209849824), (359898.2619795073, 4065151.8209849824), (359898.2619795073, 4065141.8209849824), (359898.2619795073, 4065131.8209849824), (359898.2619795073, 4065121.8209849824), (359898.2619795073, 4065111.8209849824), (359898.2619795073, 4065101.8209849824), (359898.2619795073, 4065091.8209849824), (359898.2619795073, 4065081.8209849824), (359898.2619795073, 4065071.8209849824), (359898.2619795073, 4065061.8209849824), (359898.2619795073, 4065051.8209849824), (359898.2619795073, 4065041.8209849824), (359898.2619795073, 4065031.8209849824), (359898.2619795073, 4065021.8209849824), (359898.2619795073, 4065011.8209849824), (359898.2619795073, 4065001.8209849824), (359898.2619795073, 4064991.8209849824), (359898.2619795073, 4064981.8209849824), (359898.2619795073, 4064971.8209849824), (359898.2619795073, 4064961.8209849824), (359898.2619795073, 4064951.8209849824), (359898.2619795073, 4064941.8209849824), (359898.2619795073, 4064931.8209849824), (359898.2619795073, 4064921.8209849824), (359898.2619795073, 4064911.8209849824), (359898.2619795073, 4064901.8209849824), (359898.2619795073, 4064891.8209849824), (359898.2619795073, 4064881.8209849824), (359898.2619795073, 4064871.8209849824), (359898.2619795073, 4064861.8209849824), (359898.2619795073, 4064851.8209849824), (359898.2619795073, 4064841.8209849824), (359898.2619795073, 4064831.8209849824), (359898.2619795073, 4064821.8209849824), (359898.2619795073, 4064811.8209849824), (359898.2619795073, 4064801.8209849824), (359898.2619795073, 4064791.8209849824), (359898.2619795073, 4064781.8209849824), (359898.2619795073, 4064771.8209849824), (359898.2619795073, 4064761.8209849824), (359898.2619795073, 4064751.8209849824), (359898.2619795073, 4064741.8209849824), (359898.2619795073, 4064731.8209849824), (359898.2619795073, 4064721.8209849824), (359898.2619795073, 4064711.8209849824), (359898.2619795073, 4064701.8209849824), (359898.2619795073, 4064691.8209849824), (359898.2619795073, 4064681.8209849824), (359898.2619795073, 4064671.8209849824), (359898.2619795073, 4064661.8209849824), (359898.2619795073, 4064651.8209849824), (359898.2619795073, 4064641.8209849824), (359898.2619795073, 4064631.8209849824), (359898.2619795073, 4064621.8209849824), (359898.2619795073, 4064611.8209849824), (359898.2619795073, 
4064601.8209849824), (359898.2619795073, 4064591.8209849824), (359898.2619795073, 4064581.8209849824), (359898.2619795073, 4064571.8209849824), (359898.2619795073, 4064561.8209849824), (359898.2619795073, 4064551.8209849824), (359898.2619795073, 4064541.8209849824), (359898.2619795073, 4064531.8209849824), (359898.2619795073, 4064521.8209849824), (359898.2619795073, 4064511.8209849824), (359898.2619795073, 4064501.8209849824), (359898.2619795073, 4064491.8209849824), (359898.2619795073, 4064481.8209849824), (359898.2619795073, 4064471.8209849824), (359898.2619795073, 4064461.8209849824), (359898.2619795073, 4064451.8209849824), (359898.2619795073, 4064441.8209849824), (359898.2619795073, 4064431.8209849824), (359898.2619795073, 4064421.8209849824), (359898.2619795073, 4064411.8209849824), (359898.2619795073, 4064401.8209849824), (359898.2619795073, 4064391.8209849824), (359898.2619795073, 4064381.8209849824), (359898.2619795073, 4064371.8209849824), (359898.2619795073, 4064361.8209849824), (359898.2619795073, 4064351.8209849824), (359898.2619795073, 4064341.8209849824), (359898.2619795073, 4064331.8209849824), (359898.2619795073, 4064321.8209849824), (359898.2619795073, 4064311.8209849824), (359898.2619795073, 4064301.8209849824), (359898.2619795073, 4064291.8209849824), (359898.2619795073, 4064281.8209849824), (359898.2619795073, 4064271.8209849824), (359898.2619795073, 4064261.8209849824), (359898.2619795073, 4064251.8209849824), (359898.2619795073, 4064241.8209849824), (359898.2619795073, 4064231.8209849824), (359898.2619795073, 4064221.8209849824), (359898.2619795073, 4064211.8209849824), (359898.2619795073, 4064201.8209849824), (359898.2619795073, 4064191.8209849824), (359898.2619795073, 4064181.8209849824), (359898.2619795073, 4064171.8209849824), (359898.2619795073, 4064161.8209849824), (359898.2619795073, 4064151.8209849824), (359898.2619795073, 4064141.8209849824), (359898.2619795073, 4064131.8209849824), (359898.2619795073, 4064121.8209849824), (359898.2619795073, 4064111.8209849824), (359898.2619795073, 4064101.8209849824), (359898.2619795073, 4064091.8209849824), (359898.2619795073, 4064081.8209849824), (359898.2619795073, 4064071.8209849824), (359898.2619795073, 4064061.8209849824), (359898.2619795073, 4064051.8209849824), (359898.2619795073, 4064041.8209849824), (359898.2619795073, 4064031.8209849824), (359898.2619795073, 4064021.8209849824), (359898.2619795073, 4064011.8209849824), (359898.2619795073, 4064001.8209849824), (359898.2619795073, 4063991.8209849824), (359898.2619795073, 4063981.8209849824), (359898.2619795073, 4063971.8209849824), (359898.2619795073, 4063961.8209849824), (359898.2619795073, 4063951.8209849824), (359898.2619795073, 4063941.8209849824), (359898.2619795073, 4063931.8209849824), (359898.2619795073, 4063921.8209849824), (359898.2619795073, 4063911.8209849824), 
(359898.2619795073, 4063901.8209849824), (359898.2619795073, 4063891.8209849824), (359898.2619795073, 4063881.8209849824), (359898.2619795073, 4063871.8209849824), (359898.2619795073, 4063861.8209849824), (359898.2619795073, 4063851.8209849824), (359898.2619795073, 4063841.8209849824), (359898.2619795073, 4063831.8209849824), (359898.2619795073, 4063821.8209849824), (359898.2619795073, 4063811.8209849824), (359898.2619795073, 4063801.8209849824), (359898.2619795073, 4063791.8209849824), (359898.2619795073, 4063781.8209849824), (359898.2619795073, 4063771.8209849824), (359898.2619795073, 4063761.8209849824), (359898.2619795073, 4063751.8209849824), (359898.2619795073, 4063741.8209849824), (359898.2619795073, 4063731.8209849824), (359898.2619795073, 4063721.8209849824), (359898.2619795073, 4063711.8209849824), (359898.2619795073, 4063701.8209849824), (359898.2619795073, 4063691.8209849824), (359898.2619795073, 4063681.8209849824), (359898.2619795073, 4063671.8209849824), (359898.2619795073, 4063661.8209849824), (359898.2619795073, 4063651.8209849824), (359898.2619795073, 4063641.8209849824), (359898.2619795073, 4063631.8209849824), (359898.2619795073, 4063621.8209849824), (359898.2619795073, 4063611.8209849824), (359898.2619795073, 4063601.8209849824), (359898.2619795073, 4063591.8209849824), (359898.2619795073, 4063581.8209849824), (359898.2619795073, 4063571.8209849824), (359898.2619795073, 4063561.8209849824), (359898.2619795073, 4063551.8209849824), (359898.2619795073, 4063541.8209849824), (359898.2619795073, 4063531.8209849824), (359898.2619795073, 4063521.8209849824), (359898.2619795073, 4063511.8209849824), (359898.2619795073, 4063501.8209849824), (359898.2619795073, 4063491.8209849824), (359898.2619795073, 4063481.8209849824), (359898.2619795073, 4063471.8209849824), (359898.2619795073, 4063461.8209849824), (359898.2619795073, 4063451.8209849824), (359898.2619795073, 4063441.8209849824), (359898.2619795073, 4063431.8209849824), (359898.2619795073, 4063421.8209849824), (359898.2619795073, 4063411.8209849824), (359898.2619795073, 4063401.8209849824), (359898.2619795073, 4063391.8209849824), (359898.2619795073, 4063381.8209849824), (359898.2619795073, 4063371.8209849824), (359898.2619795073, 4063361.8209849824), (359898.2619795073, 4063351.8209849824), (359898.2619795073, 4063341.8209849824), (359898.2619795073, 4063331.8209849824), (359898.2619795073, 4063321.8209849824), (359898.2619795073, 4063311.8209849824), (359898.2619795073, 4063301.8209849824), (359898.2619795073, 4063291.8209849824), (359898.2619795073, 4063281.8209849824), (359898.2619795073, 4063271.8209849824), (359898.2619795073, 4063261.8209849824), (359898.2619795073, 4063251.8209849824), (359898.2619795073, 4063241.8209849824), (359898.2619795073, 4063231.8209849824), (359898.2619795073, 4063221.8209849824), (359898.2619795073, 4063211.8209849824), (359898.2619795073, 4063201.8209849824), (359898.2619795073, 4063191.8209849824), (359898.2619795073, 4063181.8209849824), (359898.2619795073, 4063171.8209849824), (359898.2619795073, 4063161.8209849824), (359898.2619795073, 4063151.8209849824), (359898.2619795073, 4063141.8209849824), (359898.2619795073, 4063131.8209849824), (359898.2619795073, 4063121.8209849824), (359898.2619795073, 4063111.8209849824), (359898.2619795073, 
4063101.8209849824), (359898.2619795073, 4063091.8209849824), (359898.2619795073, 4063081.8209849824), (359898.2619795073, 4063071.8209849824), (359898.2619795073, 4063061.8209849824), (359898.2619795073, 4063051.8209849824), (359898.2619795073, 4063041.8209849824), (359898.2619795073, 4063031.8209849824), (359898.2619795073, 4063021.8209849824), (359898.2619795073, 4063011.8209849824), (359898.2619795073, 4063001.8209849824), (359898.2619795073, 4062991.8209849824), (359898.2619795073, 4062981.8209849824), (359898.2619795073, 4062971.8209849824), (359898.2619795073, 4062961.8209849824), (359898.2619795073, 4062951.8209849824), (359898.2619795073, 4062941.8209849824), (359898.2619795073, 4062931.8209849824), (359898.2619795073, 4062921.8209849824), (359898.2619795073, 4062911.8209849824), (359898.2619795073, 4062901.8209849824), (359898.2619795073, 4062891.8209849824), (359898.2619795073, 4062881.8209849824), (359898.2619795073, 4062871.8209849824), (359898.2619795073, 4062861.8209849824), (359898.2619795073, 4062851.8209849824), (359898.2619795073, 4062841.8209849824), (359898.2619795073, 4062831.8209849824), (359898.2619795073, 4062821.8209849824), (359898.2619795073, 4062811.8209849824), (359898.2619795073, 4062801.8209849824), (359898.2619795073, 4062791.8209849824), (359898.2619795073, 4062781.8209849824), (359898.2619795073, 4062771.8209849824), (359898.2619795073, 4062761.8209849824), (359898.2619795073, 4062751.8209849824), (359898.2619795073, 4062741.8209849824), (359898.2619795073, 4062731.8209849824), (359898.2619795073, 4062721.8209849824), (359898.2619795073, 4062711.8209849824), (359898.2619795073, 4062701.8209849824), (359898.2619795073, 4062691.8209849824), (359898.2619795073, 4062681.8209849824), (359898.2619795073, 4062671.8209849824), (359898.2619795073, 4062661.8209849824), (359898.2619795073, 4062651.8209849824), (359898.2619795073, 4062641.8209849824), (359898.2619795073, 4062631.8209849824), (359898.2619795073, 4062621.8209849824), (359898.2619795073, 4062611.8209849824), (359898.2619795073, 4062601.8209849824), (359898.2619795073, 4062591.8209849824), (359898.2619795073, 4062581.8209849824), (359898.2619795073, 4062571.8209849824), (359898.2619795073, 4062561.8209849824), (359898.2619795073, 4062551.8209849824), (359898.2619795073, 4062541.8209849824), (359898.2619795073, 4062531.8209849824), (359898.2619795073, 4062521.8209849824), (359898.2619795073, 4062511.8209849824), (359898.2619795073, 4062501.8209849824), (359898.2619795073, 4062491.8209849824), (359898.2619795073, 4062481.8209849824), (359898.2619795073, 4062471.8209849824), (359898.2619795073, 4062461.8209849824), (359898.2619795073, 4062451.8209849824), (359898.2619795073, 4062441.8209849824), (359898.2619795073, 4062431.8209849824), (359898.2619795073, 4062421.8209849824), (359898.2619795073, 4062411.8209849824), 
(359898.2619795073, 4062401.8209849824), (359898.2619795073, 4062391.8209849824), (359898.2619795073, 4062381.8209849824), (359898.2619795073, 4062371.8209849824), (359898.2619795073, 4062361.8209849824), (359898.2619795073, 4062351.8209849824), (359898.2619795073, 4062341.8209849824), (359898.2619795073, 4062331.8209849824), (359898.2619795073, 4062321.8209849824), (359898.2619795073, 4062311.8209849824), (359898.2619795073, 4062301.8209849824), (359898.2619795073, 4062291.8209849824), (359898.2619795073, 4062281.8209849824), (359898.2619795073, 4062271.8209849824), (359898.2619795073, 4062261.8209849824), (359898.2619795073, 4062251.8209849824), (359898.2619795073, 4062241.8209849824), (359898.2619795073, 4062231.8209849824), (359898.2619795073, 4062221.8209849824), (359898.2619795073, 4062211.8209849824), (359898.2619795073, 4062201.8209849824), (359898.2619795073, 4062191.8209849824), (359898.2619795073, 4062181.8209849824), (359898.2619795073, 4062171.8209849824), (359898.2619795073, 4062161.8209849824), (359898.2619795073, 4062151.8209849824), (359898.2619795073, 4062141.8209849824), (359898.2619795073, 4062131.8209849824), (359898.2619795073, 4062121.8209849824), (359898.2619795073, 4062111.8209849824), (359898.2619795073, 4062101.8209849824), (359898.2619795073, 4062091.8209849824), (359898.2619795073, 4062081.8209849824), (359898.2619795073, 4062071.8209849824), (359898.2619795073, 4062061.8209849824), (359898.2619795073, 4062051.8209849824), (359908.2619795073, 4065351.8209849824), (359908.2619795073, 4065341.8209849824), (359908.2619795073, 4065331.8209849824), (359908.2619795073, 4065321.8209849824), (359908.2619795073, 4065311.8209849824), (359908.2619795073, 4065301.8209849824), (359908.2619795073, 4065291.8209849824), (359908.2619795073, 4065281.8209849824), (359908.2619795073, 4065271.8209849824), (359908.2619795073, 4065261.8209849824), (359908.2619795073, 4065251.8209849824), (359908.2619795073, 4065241.8209849824), (359908.2619795073, 4065231.8209849824), (359908.2619795073, 4065221.8209849824), (359908.2619795073, 4065211.8209849824), (359908.2619795073, 4065201.8209849824), (359908.2619795073, 4065191.8209849824), (359908.2619795073, 4065181.8209849824), (359908.2619795073, 4065171.8209849824), (359908.2619795073, 4065161.8209849824), (359908.2619795073, 4065151.8209849824), (359908.2619795073, 4065141.8209849824), (359908.2619795073, 4065131.8209849824), (359908.2619795073, 4065121.8209849824), (359908.2619795073, 4065111.8209849824), (359908.2619795073, 4065101.8209849824), (359908.2619795073, 4065091.8209849824), (359908.2619795073, 4065081.8209849824), (359908.2619795073, 4065071.8209849824), (359908.2619795073, 4065061.8209849824), (359908.2619795073, 4065051.8209849824), (359908.2619795073, 4065041.8209849824), (359908.2619795073, 4065031.8209849824), (359908.2619795073, 4065021.8209849824), (359908.2619795073, 4065011.8209849824), (359908.2619795073, 4065001.8209849824), (359908.2619795073, 4064991.8209849824), (359908.2619795073, 4064981.8209849824), (359908.2619795073, 4064971.8209849824), (359908.2619795073, 4064961.8209849824), (359908.2619795073, 4064951.8209849824), (359908.2619795073, 4064941.8209849824), (359908.2619795073, 4064931.8209849824), (359908.2619795073, 4064921.8209849824), (359908.2619795073, 
4064911.8209849824), (359908.2619795073, 4064901.8209849824), (359908.2619795073, 4064891.8209849824), (359908.2619795073, 4064881.8209849824), (359908.2619795073, 4064871.8209849824), (359908.2619795073, 4064861.8209849824), (359908.2619795073, 4064851.8209849824), (359908.2619795073, 4064841.8209849824), (359908.2619795073, 4064831.8209849824), (359908.2619795073, 4064821.8209849824), (359908.2619795073, 4064811.8209849824), (359908.2619795073, 4064801.8209849824), (359908.2619795073, 4064791.8209849824), (359908.2619795073, 4064781.8209849824), (359908.2619795073, 4064771.8209849824), (359908.2619795073, 4064761.8209849824), (359908.2619795073, 4064751.8209849824), (359908.2619795073, 4064741.8209849824), (359908.2619795073, 4064731.8209849824), (359908.2619795073, 4064721.8209849824), (359908.2619795073, 4064711.8209849824), (359908.2619795073, 4064701.8209849824), (359908.2619795073, 4064691.8209849824), (359908.2619795073, 4064681.8209849824), (359908.2619795073, 4064671.8209849824), (359908.2619795073, 4064661.8209849824), (359908.2619795073, 4064651.8209849824), (359908.2619795073, 4064641.8209849824), (359908.2619795073, 4064631.8209849824), (359908.2619795073, 4064621.8209849824), (359908.2619795073, 4064611.8209849824), (359908.2619795073, 4064601.8209849824), (359908.2619795073, 4064591.8209849824), (359908.2619795073, 4064581.8209849824), (359908.2619795073, 4064571.8209849824), (359908.2619795073, 4064561.8209849824), (359908.2619795073, 4064551.8209849824), (359908.2619795073, 4064541.8209849824), (359908.2619795073, 4064531.8209849824), (359908.2619795073, 4064521.8209849824), (359908.2619795073, 4064511.8209849824), (359908.2619795073, 4064501.8209849824), (359908.2619795073, 4064491.8209849824), (359908.2619795073, 4064481.8209849824), (359908.2619795073, 4064471.8209849824), (359908.2619795073, 4064461.8209849824), (359908.2619795073, 4064451.8209849824), (359908.2619795073, 4064441.8209849824), (359908.2619795073, 4064431.8209849824), (359908.2619795073, 4064421.8209849824), (359908.2619795073, 4064411.8209849824), (359908.2619795073, 4064401.8209849824), (359908.2619795073, 4064391.8209849824), (359908.2619795073, 4064381.8209849824), (359908.2619795073, 4064371.8209849824), (359908.2619795073, 4064361.8209849824), (359908.2619795073, 4064351.8209849824), (359908.2619795073, 4064341.8209849824), (359908.2619795073, 4064331.8209849824), (359908.2619795073, 4064321.8209849824), (359908.2619795073, 4064311.8209849824), (359908.2619795073, 4064301.8209849824), (359908.2619795073, 4064291.8209849824), (359908.2619795073, 4064281.8209849824), (359908.2619795073, 4064271.8209849824), (359908.2619795073, 4064261.8209849824), (359908.2619795073, 4064251.8209849824), (359908.2619795073, 4064241.8209849824), (359908.2619795073, 4064231.8209849824), (359908.2619795073, 4064221.8209849824), 
(359908.2619795073, 4064211.8209849824), (359908.2619795073, 4064201.8209849824), (359908.2619795073, 4064191.8209849824), (359908.2619795073, 4064181.8209849824), (359908.2619795073, 4064171.8209849824), (359908.2619795073, 4064161.8209849824), (359908.2619795073, 4064151.8209849824), (359908.2619795073, 4064141.8209849824), (359908.2619795073, 4064131.8209849824), (359908.2619795073, 4064121.8209849824), (359908.2619795073, 4064111.8209849824), (359908.2619795073, 4064101.8209849824), (359908.2619795073, 4064091.8209849824), (359908.2619795073, 4064081.8209849824), (359908.2619795073, 4064071.8209849824), (359908.2619795073, 4064061.8209849824), (359908.2619795073, 4064051.8209849824), (359908.2619795073, 4064041.8209849824), (359908.2619795073, 4064031.8209849824), (359908.2619795073, 4064021.8209849824), (359908.2619795073, 4064011.8209849824), (359908.2619795073, 4064001.8209849824), (359908.2619795073, 4063991.8209849824), (359908.2619795073, 4063981.8209849824), (359908.2619795073, 4063971.8209849824), (359908.2619795073, 4063961.8209849824), (359908.2619795073, 4063951.8209849824), (359908.2619795073, 4063941.8209849824), (359908.2619795073, 4063931.8209849824), (359908.2619795073, 4063921.8209849824), (359908.2619795073, 4063911.8209849824), (359908.2619795073, 4063901.8209849824), (359908.2619795073, 4063891.8209849824), (359908.2619795073, 4063881.8209849824), (359908.2619795073, 4063871.8209849824), (359908.2619795073, 4063861.8209849824), (359908.2619795073, 4063851.8209849824), (359908.2619795073, 4063841.8209849824), (359908.2619795073, 4063831.8209849824), (359908.2619795073, 4063821.8209849824), (359908.2619795073, 4063811.8209849824), (359908.2619795073, 4063801.8209849824), (359908.2619795073, 4063791.8209849824), (359908.2619795073, 4063781.8209849824), (359908.2619795073, 4063771.8209849824), (359908.2619795073, 4063761.8209849824), (359908.2619795073, 4063751.8209849824), (359908.2619795073, 4063741.8209849824), (359908.2619795073, 4063731.8209849824), (359908.2619795073, 4063721.8209849824), (359908.2619795073, 4063711.8209849824), (359908.2619795073, 4063701.8209849824), (359908.2619795073, 4063691.8209849824), (359908.2619795073, 4063681.8209849824), (359908.2619795073, 4063671.8209849824), (359908.2619795073, 4063661.8209849824), (359908.2619795073, 4063651.8209849824), (359908.2619795073, 4063641.8209849824), (359908.2619795073, 4063631.8209849824), (359908.2619795073, 4063621.8209849824), (359908.2619795073, 4063611.8209849824), (359908.2619795073, 4063601.8209849824), (359908.2619795073, 4063591.8209849824), (359908.2619795073, 4063581.8209849824), (359908.2619795073, 4063571.8209849824), (359908.2619795073, 4063561.8209849824), (359908.2619795073, 4063551.8209849824), (359908.2619795073, 4063541.8209849824), (359908.2619795073, 4063531.8209849824), (359908.2619795073, 4063521.8209849824), (359908.2619795073, 4063511.8209849824), (359908.2619795073, 4063501.8209849824), (359908.2619795073, 4063491.8209849824), (359908.2619795073, 4063481.8209849824), (359908.2619795073, 4063471.8209849824), (359908.2619795073, 4063461.8209849824), (359908.2619795073, 4063451.8209849824), (359908.2619795073, 4063441.8209849824), (359908.2619795073, 4063431.8209849824), (359908.2619795073, 4063421.8209849824), (359908.2619795073, 
4063411.8209849824), (359908.2619795073, 4063401.8209849824), (359908.2619795073, 4063391.8209849824), (359908.2619795073, 4063381.8209849824), (359908.2619795073, 4063371.8209849824), (359908.2619795073, 4063361.8209849824), (359908.2619795073, 4063351.8209849824), (359908.2619795073, 4063341.8209849824), (359908.2619795073, 4063331.8209849824), (359908.2619795073, 4063321.8209849824), (359908.2619795073, 4063311.8209849824), (359908.2619795073, 4063301.8209849824), (359908.2619795073, 4063291.8209849824), (359908.2619795073, 4063281.8209849824), (359908.2619795073, 4063271.8209849824), (359908.2619795073, 4063261.8209849824), (359908.2619795073, 4063251.8209849824), (359908.2619795073, 4063241.8209849824), (359908.2619795073, 4063231.8209849824), (359908.2619795073, 4063221.8209849824), (359908.2619795073, 4063211.8209849824), (359908.2619795073, 4063201.8209849824), (359908.2619795073, 4063191.8209849824), (359908.2619795073, 4063181.8209849824), (359908.2619795073, 4063171.8209849824), (359908.2619795073, 4063161.8209849824), (359908.2619795073, 4063151.8209849824), (359908.2619795073, 4063141.8209849824), (359908.2619795073, 4063131.8209849824), (359908.2619795073, 4063121.8209849824), (359908.2619795073, 4063111.8209849824), (359908.2619795073, 4063101.8209849824), (359908.2619795073, 4063091.8209849824), (359908.2619795073, 4063081.8209849824), (359908.2619795073, 4063071.8209849824), (359908.2619795073, 4063061.8209849824), (359908.2619795073, 4063051.8209849824), (359908.2619795073, 4063041.8209849824), (359908.2619795073, 4063031.8209849824), (359908.2619795073, 4063021.8209849824), (359908.2619795073, 4063011.8209849824), (359908.2619795073, 4063001.8209849824), (359908.2619795073, 4062991.8209849824), (359908.2619795073, 4062981.8209849824), (359908.2619795073, 4062971.8209849824), (359908.2619795073, 4062961.8209849824), (359908.2619795073, 4062951.8209849824), (359908.2619795073, 4062941.8209849824), (359908.2619795073, 4062931.8209849824), (359908.2619795073, 4062921.8209849824), (359908.2619795073, 4062911.8209849824), (359908.2619795073, 4062901.8209849824), (359908.2619795073, 4062891.8209849824), (359908.2619795073, 4062881.8209849824), (359908.2619795073, 4062871.8209849824), (359908.2619795073, 4062861.8209849824), (359908.2619795073, 4062851.8209849824), (359908.2619795073, 4062841.8209849824), (359908.2619795073, 4062831.8209849824), (359908.2619795073, 4062821.8209849824), (359908.2619795073, 4062811.8209849824), (359908.2619795073, 4062801.8209849824), (359908.2619795073, 4062791.8209849824), (359908.2619795073, 4062781.8209849824), (359908.2619795073, 4062771.8209849824), (359908.2619795073, 4062761.8209849824), (359908.2619795073, 4062751.8209849824), (359908.2619795073, 4062741.8209849824), (359908.2619795073, 4062731.8209849824), (359908.2619795073, 4062721.8209849824), 
(359908.2619795073, 4062711.8209849824), (359908.2619795073, 4062701.8209849824), (359908.2619795073, 4062691.8209849824), (359908.2619795073, 4062681.8209849824), (359908.2619795073, 4062671.8209849824), (359908.2619795073, 4062661.8209849824), (359908.2619795073, 4062651.8209849824), (359908.2619795073, 4062641.8209849824), (359908.2619795073, 4062631.8209849824), (359908.2619795073, 4062621.8209849824), (359908.2619795073, 4062611.8209849824), (359908.2619795073, 4062601.8209849824), (359908.2619795073, 4062591.8209849824), (359908.2619795073, 4062581.8209849824), (359908.2619795073, 4062571.8209849824), (359908.2619795073, 4062561.8209849824), (359908.2619795073, 4062551.8209849824), (359908.2619795073, 4062541.8209849824), (359908.2619795073, 4062531.8209849824), (359908.2619795073, 4062521.8209849824), (359908.2619795073, 4062511.8209849824), (359908.2619795073, 4062501.8209849824), (359908.2619795073, 4062491.8209849824), (359908.2619795073, 4062481.8209849824), (359908.2619795073, 4062471.8209849824), (359908.2619795073, 4062461.8209849824), (359908.2619795073, 4062451.8209849824), (359908.2619795073, 4062441.8209849824), (359908.2619795073, 4062431.8209849824), (359908.2619795073, 4062421.8209849824), (359908.2619795073, 4062411.8209849824), (359908.2619795073, 4062401.8209849824), (359908.2619795073, 4062391.8209849824), (359908.2619795073, 4062381.8209849824), (359908.2619795073, 4062371.8209849824), (359908.2619795073, 4062361.8209849824), (359908.2619795073, 4062351.8209849824), (359908.2619795073, 4062341.8209849824), (359908.2619795073, 4062331.8209849824), (359908.2619795073, 4062321.8209849824), (359908.2619795073, 4062311.8209849824), (359908.2619795073, 4062301.8209849824), (359908.2619795073, 4062291.8209849824), (359908.2619795073, 4062281.8209849824), (359908.2619795073, 4062271.8209849824), (359908.2619795073, 4062261.8209849824), (359908.2619795073, 4062251.8209849824), (359908.2619795073, 4062241.8209849824), (359908.2619795073, 4062231.8209849824), (359908.2619795073, 4062221.8209849824), (359908.2619795073, 4062211.8209849824), (359908.2619795073, 4062201.8209849824), (359908.2619795073, 4062191.8209849824), (359908.2619795073, 4062181.8209849824), (359908.2619795073, 4062171.8209849824), (359908.2619795073, 4062161.8209849824), (359908.2619795073, 4062151.8209849824), (359908.2619795073, 4062141.8209849824), (359908.2619795073, 4062131.8209849824), (359908.2619795073, 4062121.8209849824), (359908.2619795073, 4062111.8209849824), (359908.2619795073, 4062101.8209849824), (359908.2619795073, 4062091.8209849824), (359908.2619795073, 4062081.8209849824), (359908.2619795073, 4062071.8209849824), (359908.2619795073, 4062061.8209849824), (359908.2619795073, 4062051.8209849824), (359918.2619795073, 4065351.8209849824), (359918.2619795073, 4065341.8209849824), (359918.2619795073, 4065331.8209849824), (359918.2619795073, 4065321.8209849824), (359918.2619795073, 4065311.8209849824), (359918.2619795073, 4065301.8209849824), (359918.2619795073, 4065291.8209849824), (359918.2619795073, 4065281.8209849824), (359918.2619795073, 4065271.8209849824), (359918.2619795073, 4065261.8209849824), (359918.2619795073, 4065251.8209849824), (359918.2619795073, 4065241.8209849824), (359918.2619795073, 4065231.8209849824), (359918.2619795073, 
4065221.8209849824), (359918.2619795073, 4065211.8209849824), (359918.2619795073, 4065201.8209849824), (359918.2619795073, 4065191.8209849824), (359918.2619795073, 4065181.8209849824), (359918.2619795073, 4065171.8209849824), (359918.2619795073, 4065161.8209849824), (359918.2619795073, 4065151.8209849824), (359918.2619795073, 4065141.8209849824), (359918.2619795073, 4065131.8209849824), (359918.2619795073, 4065121.8209849824), (359918.2619795073, 4065111.8209849824), (359918.2619795073, 4065101.8209849824), (359918.2619795073, 4065091.8209849824), (359918.2619795073, 4065081.8209849824), (359918.2619795073, 4065071.8209849824), (359918.2619795073, 4065061.8209849824), (359918.2619795073, 4065051.8209849824), (359918.2619795073, 4065041.8209849824), (359918.2619795073, 4065031.8209849824), (359918.2619795073, 4065021.8209849824), (359918.2619795073, 4065011.8209849824), (359918.2619795073, 4065001.8209849824), (359918.2619795073, 4064991.8209849824), (359918.2619795073, 4064981.8209849824), (359918.2619795073, 4064971.8209849824), (359918.2619795073, 4064961.8209849824), (359918.2619795073, 4064951.8209849824), (359918.2619795073, 4064941.8209849824), (359918.2619795073, 4064931.8209849824), (359918.2619795073, 4064921.8209849824), (359918.2619795073, 4064911.8209849824), (359918.2619795073, 4064901.8209849824), (359918.2619795073, 4064891.8209849824), (359918.2619795073, 4064881.8209849824), (359918.2619795073, 4064871.8209849824), (359918.2619795073, 4064861.8209849824), (359918.2619795073, 4064851.8209849824), (359918.2619795073, 4064841.8209849824), (359918.2619795073, 4064831.8209849824), (359918.2619795073, 4064821.8209849824), (359918.2619795073, 4064811.8209849824), (359918.2619795073, 4064801.8209849824), (359918.2619795073, 4064791.8209849824), (359918.2619795073, 4064781.8209849824), (359918.2619795073, 4064771.8209849824), (359918.2619795073, 4064761.8209849824), (359918.2619795073, 4064751.8209849824), (359918.2619795073, 4064741.8209849824), (359918.2619795073, 4064731.8209849824), (359918.2619795073, 4064721.8209849824), (359918.2619795073, 4064711.8209849824), (359918.2619795073, 4064701.8209849824), (359918.2619795073, 4064691.8209849824), (359918.2619795073, 4064681.8209849824), (359918.2619795073, 4064671.8209849824), (359918.2619795073, 4064661.8209849824), (359918.2619795073, 4064651.8209849824), (359918.2619795073, 4064641.8209849824), (359918.2619795073, 4064631.8209849824), (359918.2619795073, 4064621.8209849824), (359918.2619795073, 4064611.8209849824), (359918.2619795073, 4064601.8209849824), (359918.2619795073, 4064591.8209849824), (359918.2619795073, 4064581.8209849824), (359918.2619795073, 4064571.8209849824), (359918.2619795073, 4064561.8209849824), (359918.2619795073, 4064551.8209849824), (359918.2619795073, 4064541.8209849824), (359918.2619795073, 4064531.8209849824), 
(359918.2619795073, 4064521.8209849824), (359918.2619795073, 4064511.8209849824), (359918.2619795073, 4064501.8209849824), (359918.2619795073, 4064491.8209849824), (359918.2619795073, 4064481.8209849824), (359918.2619795073, 4064471.8209849824), (359918.2619795073, 4064461.8209849824), (359918.2619795073, 4064451.8209849824), (359918.2619795073, 4064441.8209849824), (359918.2619795073, 4064431.8209849824), (359918.2619795073, 4064421.8209849824), (359918.2619795073, 4064411.8209849824), (359918.2619795073, 4064401.8209849824), (359918.2619795073, 4064391.8209849824), (359918.2619795073, 4064381.8209849824), (359918.2619795073, 4064371.8209849824), (359918.2619795073, 4064361.8209849824), (359918.2619795073, 4064351.8209849824), (359918.2619795073, 4064341.8209849824), (359918.2619795073, 4064331.8209849824), (359918.2619795073, 4064321.8209849824), (359918.2619795073, 4064311.8209849824), (359918.2619795073, 4064301.8209849824), (359918.2619795073, 4064291.8209849824), (359918.2619795073, 4064281.8209849824), (359918.2619795073, 4064271.8209849824), (359918.2619795073, 4064261.8209849824), (359918.2619795073, 4064251.8209849824), (359918.2619795073, 4064241.8209849824), (359918.2619795073, 4064231.8209849824)]

# def inside(xmin,ymin,xmax,ymax, lista_pixels):
#     index_points=[]

#     for i in range(len(lista_pixels)-3000000):
#         if xmin < lista_pixels[i][0] < xmax and ymin < lista_pixels[i][1] < ymax:
#             index_points.append(i)
#     return index_points

def core_algorithm(point,polygon):
    # A point is in a polygon if a line from the point to infinity crosses the polygon an odd number of times
    odd = False
    i = 0
    j = len(polygon) - 1
    while i < len(polygon) - 1:
        i = i + 1
        if (((polygon[i][1] > point[1]) != (polygon[j][1] > point[1])) and (point[0] < (
                (polygon[j][0] - polygon[i][0]) * (point[1] - polygon[i][1]) / (polygon[j][1] - polygon[i][1])) +
                                                                            polygon[i][0])):
            odd = not odd #* Invert odd
        j = i
    return odd #* If odd point is in polygon

# ind_values = []
def index_values(lista_points,polygon):
    index_points=[]
    for i in range(len(lista_points)): 
        bool = core_algorithm(lista_points[i],polygon)
        if bool: 
            index_points.append(i)
    return index_points

# def get_band_matrix2(path,points_list,arr,transformer):
def get_band_matrix2(path,points_list,arr,transformer):
    cont = 0
    # chunked_list = np.array_split(points_list, 10)
    # print(len(chunked_list[0]))
    with fiona.open(path) as layer:
        for feature in tqdm(layer):
            ord_dict = feature['properties']
            for key in ord_dict.values():
                if key in cod_uso: 
                    cd_uso = get_id_codigo_uso(key) #* codigo sigpac para cada iteraccion del shapefile
            geometry = feature["geometry"]['coordinates']
            for g in geometry:
                polygon = Polygon(g)
                bounds = polygon.bounds
                xmin = bounds[0]
                ymin = bounds[1]
                xmax = bounds[2]
                ymax = bounds[3]
                # ind_values = inside(xmin,ymin,xmax,ymax, points_list)
                ind_values = index_values(points_list, g)
                # if __name__ == '__main__':
                #     p1 = multiprocessing.Process(target=index_values, args=(chunked_list[0],g))
                #     p2 = multiprocessing.Process(target=index_values, args=(chunked_list[1],g))
                #     p3 = multiprocessing.Process(target=index_values, args=(chunked_list[2],g))
                #     p4 = multiprocessing.Process(target=index_values, args=(chunked_list[3],g))
                #     p1.start()
                #     p2.start()
                #     p3.start()
                #     p4.start()
                #     p1.join()
                #     p2.join()
                #     p3.join()
                #     p4.join()

                if len(ind_values) != 0: #* no todas las parcelas tienen al menos un pixel del arr en su interior
                    print(ind_values)
                    for ind in ind_values:
                        ind_arr = transformer.rowcol(points_list[ind][0],points_list[ind][1])
                        arr[ind_arr[0],ind_arr[1]] = cd_uso #* sustituye el valor de la banda anterior por el del SIGPAC
            cont+=1
            # print(cont)
    return arr
    
# get_band_matrix2("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp", list_pixels2)




with rasterio.open("C:\TFG_resources\satelite_images_sigpac\maskSUF_raster_29900.tif") as src:
    profile = src.profile #* raster meta-data
    arr = src.read(1) #* band
    transformer = rasterio.transform.AffineTransformer(profile['transform'])

    not_zero_indices = np.nonzero(arr) #* Get all indexes of non-zero values in array to reduce its size

    # points_list = [transformer.xy(not_zero_indices[0][i],not_zero_indices[1][i]) for i in tqdm(range(len(not_zero_indices[0])))] #* coordinates list of src values
    # print(len(points_list)) 

    # matrix = get_band_matrix2("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp",
    #                         points_list,arr,transformer)

    matrix = get_band_matrix2("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp", list_pixels2,arr,transformer)

    with rasterio.open('pruebas2_pixels.tif', 'w', **profile) as dst:
        dst.write(matrix, 1)

# def prueba(path):
#     with fiona.open(path) as layer:
#         for feature in layer:
#             ord_dict = feature['properties']
#             print(ord_dict)
#             # for key in ord_dict.values():
#             #     if key in cod_uso: 
#             #         cd_uso = get_id_codigo_uso(key)
#             # geometry = feature["geometry"]['coordinates']

# prueba("C:\\TFG_resources\\shape_files\\Municipio29_Malaga\\SeparadosMunicipios\\SP20_REC_29900.shp")
