FROM python:3.11

WORKDIR /app

#COPY FOLDERS NEEDED
COPY /src /app/src/
COPY /json /app/json/

#COPY FILES NEEDED
COPY requirements.txt launch.sh tfg_data/classification_30SUF.tif /app/
COPY tfg_data/Shapefile_Data/MALAGA/SP20_REC_29025.shx tfg_data/Shapefile_Data/MALAGA/SP20_REC_29025.shp /app/

RUN pip install -r requirements.txt

#CMD ["bash", "launch.sh", "-r", "/app/classification_30SUF.tif", "-s", "/app/SP20_REC_29025.shp", "-o", "ejemplo", "-t", "no"]

CMD ["python", "demo.py", "/app/classification_30SUF.tif", "/app/SP20_REC_29025.shp", "ejemplo", "no"]
