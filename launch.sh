#!/bin/bash
#?wget -r --user-agent="Mozilla" http://ftp.itacyl.es/cartografia/05_SIGPAC/2021_ETRS89/Parcelario_SIGPAC_CyL_Municipios/47_Valladolid/
#? 

# Here are a few suggestions to improve the Bash script:

#     Error messages: When the user enters an invalid option or a required argument is missing, the script currently exits with an error code of 1, but it would be better to print a descriptive error message as well.

#     Consistent variable names: Some variables in the script have inconsistent names, which can make the code harder to read and understand. For example, the script uses both "tmp" and "DELETE_TMP" to refer to the same thing.

#     Use functions: The script could be more modular and easier to read if some of the code were encapsulated in functions. For example, the code that creates the temporary directories could be put in a separate function.

#     Check for dependencies: Before running the script, it would be a good idea to check that all the required dependencies are installed. For example, the script currently assumes that "pip" is available, but this may not always be the case.

#     Improve user interface: The script could be improved by providing more informative messages to the user, such as progress updates or warnings about potential issues.

#     Improve parameter validation: The script currently checks whether the -t argument is "yes" or "no", but it would be better to use a case-insensitive comparison and to provide a more informative error message if an invalid option is entered.

#     Use shellcheck: It is always a good idea to run your Bash scripts through a static analysis tool like shellcheck to catch potential issues.

# Overall, the script looks well-structured and easy to read, but there is room for improvement in terms of error handling, modularity, and user interface.

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
        echo "  -h, --help              This will show this help message"
        echo "  -v, --version           The version of the app"
        echo "  -r, --raster_path       Path of the raster we want to validate (Path)"
        echo "  -s, --shapefile_path    Path of the shapefile(s) with the geo data (Path)"
        echo "  -o, --output            Name for the final file (String)"
        echo "  -t, --delete_tmp        Remove all the temporary files created (Yes/No)"
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
