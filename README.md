# Automatic land cover validator

![license](https://img.shields.io/badge/Python-coverage%2068.4%25-gree) ![license](https://img.shields.io/badge/JupyterNotebook-coverage%2024.6%25-orange) ![license](https://img.shields.io/badge/Shell-coverage%207%25-yellow)
![license](https://img.shields.io/badge/rasterio-v1.3.0-red) ![license](https://img.shields.io/badge/Fiona-v1.8.21-red) ![license](https://img.shields.io/badge/license-MIT-blue) 


## Overview
Final degree project for University of Malaga at Khaos Research. Author: Jesús Aldana Martín

## Description

This repository contains the code to validate land cover. In order to execute the app you must have Python3 installed in your machine. The example uploaded use the SIGPAC data of the Ministry of Agriculture, Fisheries and Food of Spain. Data source:

https://www.juntadeandalucia.es/organismos/agriculturapescaaguaydesarrollorural/servicios/sigpac/visor/paginas/sigpac-descarga-informacion-geografica-shapes-provincias.html

## Contents

Code functions:

- Reproject rasters
- Merge tiff images
- Create mask from SHP data
- Point in polygon
- Raster comparison
- Validation metrics

## Setup

To run locally the script you just need to install all libraries specified in the requirements.txt. The code below showw how can you do it.

```Python
python3 pip install -r requirements.txt
```

## Usage

Use all the functions as you wish or run the whole workflow with the launch.sh app. In order to run the script please, replace the 'elements' with your own paths.

```Shell
bash launch.sh -r <raster path> -s <shp path> -o <output path> -t <delete tmp>
```

## Testing

The framework used for the unit tests is pytests. In order to run the tests:

```Python
pytest tests/
```

## Output example

![title](../satelite_images_sigpac/assets/images/sgpc.png)

For extra information check out the [showcase.ipynb](showcase.ipynb) notebook.

---

## The MIT License (MIT)

This project is licensed under the MIT license. See the [LICENSE](LICENSE) file for more info.
