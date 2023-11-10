#!/bin/bash

# Build docker image
docker build -t tfg:latest .

# Run docker image
docker run -v /home/$USER/Documents/resultado_demo:/app/demo/data/results tfg
