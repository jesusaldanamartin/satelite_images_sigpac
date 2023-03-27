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
pip install -r requirements.txt
```

## Usage

Use all the functions as you wish or run the whole workflow with the launch.sh app. In order to run the script please, replace the 'elements' with your own paths.

```Shell
bash launch.sh -r 'raster path' -s 'shp path' -o 'output path' -t 'delete tmp'
```

## Credits

---

## The MIT License (MIT)

Copyright © 2023 <Yo>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



