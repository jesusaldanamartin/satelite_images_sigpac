FROM python:3.11

WORKDIR /app

#COPY FOLDERS NEEDED
COPY /src /app/src/
COPY /json /app/json/
COPY /demo /app/demo/
COPY requirements.txt /app

#COPY FILES NEEDED
RUN pip install -r requirements.txt

#RUN THE SCRIPT
CMD ["python3", "/app/src/demo.py", \
        "/app/demo/tif/mask_demo.tif", \
         "/app/demo/shp/sp22_REC_41086.shp", \
         "demo", "no"]

