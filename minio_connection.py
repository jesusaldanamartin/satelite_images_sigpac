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
tile = "29TQJ"
minio.fget_object(
        "etc-classifications",
        "datos-gabriel/classification_"+f"{tile}.tif",
        "C:\TFG_resources\satelite_img\spain_tile"+f"{tile}.tif"
        )

tile = "30STF"
minio.fget_object(
        "pruebas-descarga-tiff",
        "masFoto/holamasfotos/classification_"+f"{tile}.tif",
        "C:\TFG_resources\satelite_img\spain_"+f"{tile}.tif"
        )




# tiles = get_list_of_tiles_in_spain()

# def download_tiles(bucket_name,object_name,folder_path):
#     for tile in tiles:
#                 minio.fget_object(
#                 bucket_name,
#                 object_name+f"{tile}.tif",
#                 folder_path+f"{tile}.tif"
#                 )


# download_tiles("etc-classifications",
#                 "datos-gabriel/classification_",
#                 "C:\TFG_resources\satelite_img\spain_tile")
