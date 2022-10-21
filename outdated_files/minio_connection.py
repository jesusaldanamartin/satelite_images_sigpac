from minio import Minio, S3Error
from decouple import config
from http.client import HTTPException
from pathlib import Path
import os



M_PORT = config("M_PORT", default="minio2.khaos.uma.es:9000/")
M_ACCESS = config("M_ACCESS", default="virginia")
M_PASSWORD = config("M_PASSWORD", default="virginia-etc")

# M_PORT = config("M_PORT", default="192.168.219.38:9000")
# M_ACCESS = config("M_ACCESS", default="minioadmin")
# M_PASSWORD = config("M_PASSWORD", default="minioadmin")

def create_minio() -> Minio:
 
    client = Minio(M_PORT, M_ACCESS, M_PASSWORD, secure=False)

    return client

minio = create_minio()
tile = "30SXG"
minio.fget_object(
        "etc-classifications",
        "datos-gabriel/classification_"+f"{tile}.tif",
        "C:\TFG_resources\satelite_img\spain_tile"+f"{tile}.tif"
        )

# tile = "30STF"
# minio.fget_object(
#         "pruebas-descarga-tiff",
#         "masFoto/holamasfotos/classification_"+f"{tile}.tif",
#         "C:\TFG_resources\satelite_img\spain_"+f"{tile}.tif"
#         )

# tile = "classification_30STF"
# minio.fget_object(
#         "pruebas-descarga-tiff",
#         "masFoto/holamasfotos/"+f"{tile}.tif",
#         "C:\TFG_resources\satelite_img\spain_tile"+f"{tile}.tif"
#         )

# def get_list_of_tiles_in_spain():
#     tiles = ['30SYG', '29TPG', '31SCC', '31TDE', '31SBD', '31SBC', '29SPC', '30STH', '30SYJ',
#     '30SYH', '31SCD', '31SED', '31SDD', '29SQC', '29TPF', '30SVH', '30SVJ', '30SWJ',
#     '30STG', '30SUH', '29SPD', '29TPH', '30TUM', '30SUJ', '30SUE', '30TVK', '31TCF',
#     '29SQD', '31TEE', '29SQA', '29SPA', '30SWF', '30SUF', '30TTM', '29TQG', '29TQE',
#     '29SQB', '30TTK', '29TNG', '29SPB', '29SQV', '30SXG', '30SXJ', '30SXH', '30SUG',
#     '30STJ', '30TWL', '29TPE', '30STF', '30SVF', '30STE', '30TWK', '30TUK', '30SWG',
#     '30SVG', '29TQF', '30SWH', '31TBE', '30SXF', '30TTL', '30TVL', '31TBF', '30TUL',
#     '30TYK', '30TXK', '31TDF', '30TYL', '31TBG', '30TYM', '27RYM', '30TXL', '29TNH',
#     '27RYL', '29TQH', '31TCG', '27RYN', '30TXM', '31TDG', '30TUN', '30TVM', '31TFE',
#     '30TWM', '29TNG', '29THN', '29TNJ', '29TPJ', '29TQJ', '30TPU', '30TVP', '30TWP',
#     '30TVN', '30TWN', '30TXN', '30TYN', '31TCH' ]

#     return tiles

# tiles = get_list_of_tiles_in_spain()

# tile = "classification_30STF.tif"
# def download_tiles(bucket_name,object_name,folder_path):
#     for tile in tiles:
#                 minio.fget_object(
#                 bucket_name,
#                 object_name+f"{tile}.tif",
#                 folder_path+f"{tile}.tif"
#                 )


# download_tiles("pruebas-descarga-tiff",
#                 "datos-masFoto/holamasfotos",
#                 "C:\TFG_resources\satelite_img\spain_tile")
