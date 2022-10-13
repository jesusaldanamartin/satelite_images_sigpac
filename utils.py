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
from shapely.geometry import Polygon, MultiPolygon,MultiLineString
import fiona
import geojson
import rasterio.mask
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

#! 07/09

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

# south_spain_rasterio = merge_tiles_rasterio("merge-Great")
merged_file = PATH+f"/malagaMask.tif"
# merge_tiles_tifftools("merg")

#* show(south_spain_rasterio, cmap='gist_earth', title='satelite view') # More cmap parameters: viridis, Greens, Blues, Reds, jet
#* _, file_paths = get_tiles_merge(folder_files)

#! 01/09/2022

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
    out = []
    for item in x:
        if isinstance(item, list):
            out.append(swapCoords(item))
        else:
            return [x[1], x[0]]
    return out

def geojson_coords_into_lat_lon():
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

    # with fiona.open(path, layer="recinto") as layer:
    #     multipolygon = [feature['geometry'] for feature in layer]
    #     counter=0
    #     while counter < len(multipolygon):
    #         i=0
    #         for coord in multipolygon[counter]['coordinates'][0]:
    #             multipolygon[counter]['coordinates'][0][i] = utm_to_wgs(False, coord)
    #             i+=1
    #         counter+=1

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


"""FIRST CONVERT VECTOR DATA INTO RASTER DATA"""



# get_raster_from_shapefile()


    #  if mask_geometry:
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

# src= rasterio.open("C:\TFG_resources\satelite_img\classification_30SUF.tif")
# "C:\TFG_resources\satelite_images_sigpac\malagaMask.tif"

# def get_polygon_dataframe_geometries():
#     with fiona.open("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp") as layer:

#     return 

# create_raster_from_geojson()
# create_geo_dataframe_geometries()
# print(df.head())


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

def get_corners_geometry():
    """
    Get the coordinates of the 4 corners of the bounding box of a geometry
    """
    shapefile = "C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp"
    with fiona.open(shapefile) as layer:
        for geometry in layer:
            coordinates = geometry["geometry"]['coordinates']
            if geometry["type"] == "MultiPolygon":
                coordinates = coordinates[0]  # TODO multiple polygons in a geometry
            lon = []
            lat = []
            if geometry["type"] == "Point":
                lon.append(coordinates[0])
                lat.append(coordinates[1])
            else:
                coordinates = coordinates[
                    0
                ]  # Takes only the outer ring of the polygon: https://geojson.org/geojson-spec.html#polygon
                for coordinate in coordinates:
                    lon.append(coordinate[0])
                    lat.append(coordinate[1])

    max_lon = max(lon)
    min_lon = min(lon)
    max_lat = max(lat)
    min_lat = min(lat)

    return {
        "top_left": (max_lat, min_lon),
        "top_right": (max_lat, max_lon),
        "bottom_left": (min_lat, min_lon),
        "bottom_right": (min_lat, max_lon),
    }

# print(get_corners_geometry())


def get_polygon_corners():

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

# print(get_polygon_corners())
# corners = get_polygon_corners()
# top_left_corner = corners["top_left"]
# top_left_corner = (top_left_corner[1], top_left_corner[0])
# project = pyproj.Transformer.from_crs(
#     pyproj.CRS.from_epsg(4326), crs_to=4326,always_xy=True).transform
# top_left_corner = transform(project, Point(top_left_corner))
# transform = rasterio.Affine(
#     10,
#     0.0,
#     top_left_corner.x,
#     0.0,
#     -10,
#     top_left_corner.y,)
# print(top_left_corner)
# print(project)
# print(transform)

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

    src = rasterio.open(filepath)
    crs = src.crs
    array = src.read()
    xmin, ymax = np.around(src.xy(0.00, 0.00), 9) # Get the center of the raster
    xmax, ymin = np.around(src.xy(src.height-1, src.width-1), 9)  

    x = np.linspace(xmin, xmax, src.width)
    # print(xmin)
    # print(xmax)
    # print(src.width)
    y = np.linspace(ymax, ymin, src.height) 

    # create 2D arrays
    xs, ys = np.meshgrid(x, y)

    data = {"X": pd.Series(xs.ravel()),
            "Y": pd.Series(ys.ravel())}

    raster_dataframe = pd.DataFrame(data=data)
    raster_dataframe['band1'] = array[0].ravel() 
    np.round(raster_dataframe["X"],9)
    np.round(raster_dataframe["Y"],9)

    # print(len(raster_dataframe))
    # print(raster_dataframe.head(20))

    return raster_dataframe.to_csv("df_raster")

# get_numpy_array_from_raster("C:\\TFG_resources\\satelite_images_sigpac\\primera_prueba")
    # geometry = gpd.points_from_xy(raster_dataframe.X, raster_dataframe.Y)
    # gdf = gpd.GeoDataFrame(raster_dataframe, crs=crs, geometry=geometry)

    # print(gdf.head(20))

# def tuple_values_xy(row):

#     data = []
#     data.append(list(zip(row["X", "Y"])))
#     print(data)
        # raster_df['xy'] = raster_df.apply(tuple_values_xy, axis=1)

#     return data

def get_codigo_for_pixel():

    polygon_df = pd.read_csv("df_codigo_polygon.csv", sep =",")
    # polygon_df = polygon_df[polygon_df['cod_uso'].isin(cod_uso)]
    print(len(polygon_df))
    raster_points_df = pd.read_csv("df_raster_points.csv", sep=",")
    print(len(raster_points_df))
    raster_df = pd.read_csv("df_raster.csv",sep=",")
    raster_df = raster_df[raster_df['band1'] == 255]

    # print(raster_points_df.head())
    # print(polygon_df.head())
    #3967902
    #3964000
    for i in range(len(raster_points_df)-3966479):
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
            # print("------------------",j,"-----------------")
            break

    return raster_df.to_csv("raster_with_bands.csv")

# get_codigo_for_pixel()

def get_id_codigo_uso(key):
    if key == 'AG' : return 1
    if key == 'CA' : return 2
    if key == 'CF' : return 3
    if key == 'CI' : return 4
    if key == 'CS' : return 5
    if key == 'CV' : return 6
    if key == 'ED' : return 7
    if key == 'EP' : return 8
    if key == 'FF' : return 9
    if key == 'FL' : return 10
    if key == 'FO' : return 11
    if key == 'FS' : return 12
    if key == 'FV' : return 13
    if key == 'FY' : return 14
    if key == 'IM' : return 15
    if key == 'IV' : return 16
    if key == 'OC' : return 17
    if key == 'OF' : return 18
    if key == 'OV' : return 19
    if key == 'PA' : return 20
    if key == 'PR' : return 21
    if key == 'PS' : return 22
    if key == 'TA' : return 23
    if key == 'TH' : return 24
    if key == 'VF' : return 25
    if key == 'VI' : return 26
    if key == 'VO' : return 27
    if key == 'ZC' : return 28
    if key == 'ZU' : return 29
    if key == 'ZV' : return 30

def x_y_to_point(row):
    return Point(row['X'],row['Y'])

def point_contains_geometry(row):
    if row['polyg'].contains(row['Point']):
        x = row['Point'].x
        y = row['Point'].y
        id = row['band1']
        return x,y,id
    else:
        return 0

def is_in_geometry(polygon,df):
    dataf = pd.DataFrame(columns=['X','Y','band1'])
    dataframe = df

    with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
                # df['Point'] = df.apply(lambda row: Point(row['X'], row['Y']), axis = 1)
                df['Point'] = df.apply(x_y_to_point, axis = 1)
    # df = df.iloc[: , 3:]
    df = df.assign(polyg = polygon)
    df_xy = df.apply(point_contains_geometry, axis = 1)
    df_xy = df.select_dtypes(exclude=['object'])
    # print(df_bool)
    # for i in range(len(dataframe)):
    #     # point = df.loc[i,'Point']
    #     if df_xy[i] != None:
    #     # if polygon.contains(point):
    #         x = df_xy[i][0]
    #         y = df_xy[i][1]
    #         id = dataframe.loc[i,'band1']
    #         dataf.loc[len(dataf)] = [x, y, id]
    return df_xy

def transform_x_y_into_point():
    raster_df = pd.read_csv("df_raster.csv", sep =",")
    raster_df = raster_df[raster_df['band1'] == 255]
    with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
            raster_df['Point'] = raster_df.apply(lambda row: Point(row['X'], row['Y']), axis = 1)
    raster_df = raster_df.iloc[: , 3:]

    return  raster_df.to_csv("df_raster_points")

# transform_x_y_into_point()

def get_numpy_array_each_geometry(path):

    with fiona.open(path) as layer:
        # df = pd.DataFrame(columns=['X','Y','band1'])
        dfs=[]
        cont=0
        for feature in layer:
            ord_dict = feature['properties']
            for key in ord_dict.values():
                if key in cod_uso: 
                    # get_id_codigo_uso(key)
                    cd_uso = get_id_codigo_uso(key)
            geometry = feature["geometry"]['coordinates']
            for g in geometry:
                polygon = Polygon(g)
                xmin, ymin, xmax, ymax = polygon.bounds
                xmin = np.round(xmin,9)
                xmax = np.round(xmax,9)
                ymin = np.round(ymin,9)
                ymax = np.round(ymax,9)
                # width = (xmax-xmin)/10
                # height = (ymax-ymin)/10
                x_res = int((xmax - xmin) / 10)
                y_res = int((ymax - ymin) / 10)
                x = np.linspace(xmin, xmax, x_res)
                y = np.linspace(ymax, ymin, y_res) 

                xs, ys = np.meshgrid(x, y)
                data = {"X": pd.Series(xs.ravel()),
                        "Y": pd.Series(ys.ravel())}
                
                raster_dataframe = pd.DataFrame(data=data)
                raster_dataframe = raster_dataframe.assign(band1 = cd_uso)
                if not raster_dataframe.empty:
                    dataset = is_in_geometry(polygon ,raster_dataframe)
                    # dataset = dataset.align(cd_uso)
                    # dataset['band1'] = cd_uso
                    # print(dataset)
                    dfs.append(dataset)
                    # print(dataset)
                cont+=1
                print(cont)
    return pd.concat(dfs).to_csv("x_y_ID.csv")

# get_numpy_array_each_geometry("C:\TFG_resources\shape_files\Municipio29_Malaga\SeparadosMunicipios\SP20_REC_29900.shp")

#? Numpy array with X-Y-id
#? Create raster from each geometry 


data_frame = pd.read_csv("x_y_id.csv", sep =",")
del(data_frame[data_frame.columns[0]])

xmin=data_frame['X'].min()
xmax=data_frame['X'].max()
ymin=data_frame['Y'].min()
ymax=data_frame['Y'].max()

x_res = int((xmax - xmin) / 10)
y_res = int((ymax - ymin) / 10)

print(x_res)
print(y_res)
my_array = data_frame.to_numpy()
print(my_array)

#! NECESITO driver height width dtype crs transform
from rasterio.transform import from_origin
# from rasterio.crs import crs

driver = "GTIFF"
height = 2846
width = 2933
count = 1
dtype = my_array.dtype
# crs = crs.from_epsg(32630)
transforma = from_origin(xmin,ymax,10,10)
print(transforma)
print(dtype)
print(my_array[:,[2]])
with rasterio.open("resultado7_210922.tif", "w",
                driver=driver,
                height=height,
                width=width,
                count=count,
                dtype=dtype,
                transform=transforma) as dst:
    dst.write(my_array[:,2],1)  
