#!/bin/bash

if [ ! -d "$data_prueba" ]; then
    mkdir data_prueba
    mkdir data_prueba/tmp
    mkdir data_prueba/tmp/masked
    mkdir data_prueba/tmp/sigpac
    mkdir data_prueba/results
fi

python3 pip install -r requirements.txt

python3 exec.py "$@"