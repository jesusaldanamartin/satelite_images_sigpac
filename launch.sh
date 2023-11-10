#!/bin/bash

# Global variables
APP_NAME="Satellite image validator with SIGPAC data"
APP_VERSION="2.1.0"
USAGE="Usage: $0 [-r|--raster_path] [-s|--shapefile_path] [-o|--output] [-t|--delete_tmp]"

# Parse command line options
PARAMS=$(getopt -o hvt:r:s:o: --long help,version,delete_tmp:,raster_path:,shapefile_path:,output: --name "$0" -- "$@")

# Evaluate options
eval set -- "$PARAMS"
while true; do
    case "$1" in
    -h | --help)
        echo ""
        echo "$APP_NAME"
        echo ""
        echo "A python app for the automatic validation of satellite rasters"
        echo ""
        echo "Options:"
        echo "  -h, --help              This will show this help message."
        echo "  -v, --version           The version of the app."
        echo "  -r, --raster_path       Path of the raster we want to validate (Path)."
        echo "  -s, --shapefile_path    Path of the shapefile(s) with the geo data (Path)."
        echo "  -o, --output            Name for the final file (String)."
        echo "  -t, --delete_tmp        Remove all the temporary files created (Yes/No)."
        echo ""
        echo "$USAGE"
        echo ""
        echo "To get more information about the app, check out the docs at https://github.com/jesusaldanamartin/satelite_images_sigpac/"
        exit 0
        ;;
    -v | --version)
        echo "version $APP_VERSION, last update: 23/02/2023"
        exit 0
        ;;
    -r | --raster_path)
        RASTER_PATH="$2"
        shift 2
        ;;
    -s | --shapefile_path)
        SHP_PATH="$2"
        shift 2
        ;;
    -o | --output)
        OUTPUT="$2"
        shift 2
        ;;
    -t | --delete_tmp)
        if ! ([ "$2" == "yes" ] || [ "$2" == "no" ]); then
            echo "ERROR "
            echo ""
            echo "-t argument must be yes or no"
            exit 0
        fi
        tmp="$2"
        break
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "Invalid option: $1"
        echo "$USAGE"
        exit 1
        ;;
    esac
done

# Error handler if none arguments are passed
if [ $# -eq 0 ]; then
    echo ""
    echo "Some arguments are required, please check out the usage guide"
    echo ""
    echo "$USAGE"
    echo "For more information use -h|--help command"
    echo ""
    exit 1
fi

while true; do
    echo ""
    echo "Do you wish to install all the requirements and run the prgram?"
    read -p "Please select [Y/N]: " yn
    case $yn in
    [Yy]*)
        pip install -r ../satelite_images_sigpac/requirements.txt
        echo ""

        if [[ ! -d "$data" ]]; then
            mkdir -p data/tmp/masked
            mkdir -p data/tmp/sigpac
            mkdir -p data/results
        fi

        python3 run.py $RASTER_PATH $SHP_PATH $OUTPUT $tmp

        break
        ;;
    [Nn]*) exit ;;
    *) echo "Please answer yes or no." ;;
    esac
done
