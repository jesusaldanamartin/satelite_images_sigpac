from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Tuple, IO
import rasterio
import json
import matplotlib.pyplot as plt


def read_needed_files(json_path: str, masked_path: str, sigpac_path: str):# -> Tuple(int, int, str, dict, np.ndarray, np.ndarray):
    '''Read all files needed and call function raster_comparison_cropland() to compare both rasters.

    Args:
        json_path (str): Path to the json file.
        sigpac_path (str): Path to the sigpac processed raster.
        masked_path (str): Path to the classification raster (LAB).

    Returns:
        rows_msk (int): Number of rows.
        cols_msk (int): Number of columns.
        out_meta (str): Output raster metadata.
        style_sheet (dict): Data obtained from json_path. 
        classification_band (np.ndarray): Raster data, read with rasterio.
        sigpac_band (np.ndarray): Raster data, read with rasterio.
    '''

    with open(json_path) as jfile:
        dict_json = json.load(jfile)
        style_sheet = dict_json['style_sheet']['SIGPAC_code']

    with rasterio.open(masked_path) as src2:
        classification_band = src2.read(1)
        rows_msk = classification_band.shape[0]
        cols_msk = classification_band.shape[1]
        out_meta = src2.meta

    with rasterio.open(sigpac_path) as src:
        sigpac_band = src.read(1)
        rows_sgc = sigpac_band.shape[0]
        cols_sgc = sigpac_band.shape[1]

    try:
        if (rows_msk != rows_sgc or cols_msk != cols_sgc):
            raise ValueError
        else:
            return rows_msk, cols_msk, out_meta, style_sheet, classification_band, sigpac_band
    except ValueError as err:
        print(err + ": Both rasters must have the same size")


def raster_comparison(rows: int, cols: int, metadata, output_path: str,
                      style_sheet: dict, classification_band: np.ndarray, sigpac_band: np.ndarray) -> IO:
    '''This function compares the band values of two different raster. These values 
    are linked with the crop_style_sheet.json file. Both rasters must have the same size.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        new_raster_output (np.ndarray): 2D numpy array copy of our input raster.
        style_sheet (dict): Path to the json file.
        classification_band (ndarray): Lab raster band read with rasterio.
        sigpac_band (ndarray): Sigpac raster band read with rasterio.

    Returns:
        This function returns a ndarray where band values have been replaced with 
        the new compared values. Band values:

        Band number 1(green): Same values in both pixels
        Band number 2(red): Different values in both pixels

    '''

    new_raster_output = np.copy(sigpac_band)

    try:
        for x in tqdm(range(rows)):
            for y in range(cols):
                if sigpac_band[x, y] != 0 and classification_band[x, y] != 0:
                    if len(style_sheet[str(sigpac_band[x, y])]) > 1:
                        for item in style_sheet[str(sigpac_band[x, y])]:
                            if classification_band[x, y] == item:
                                new_raster_output[x, y] = 1  # * same
                                continue
                            else:
                                new_raster_output[x, y] = 2  # * diff

                    else:
                        if style_sheet[str(sigpac_band[x, y])] == classification_band[x, y]:
                            new_raster_output[x, y] = 1  # * same

                        else:
                            new_raster_output[x, y] = 2  # * diff
    except IndexError:
        pass

    with rasterio.open(output_path, "w", **metadata) as dest:
        dest.write(new_raster_output, 1)


def raster_comparison_confmatrix(rows: int, cols: int, metadata: str, output_path: str,
                                 style_sheet: dict, classification_band: np.ndarray, sigpac_band: np.ndarray) -> IO:
    '''This function compares the crop zones in both land covers given by parameters.
    These values are linked with the id_style_sheet.json file. Both rasters must have the same size.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        new_raster_output (np.ndarray): 2D numpy array copy of our input raster.
        style_sheet (dict): Path to the json file.
        classification_band (ndarray): Lab raster band read with rasterio.
        sigpac_band (ndarray): Sigpac raster band read with rasterio.

    Returns:
        This function returns a ndarray matrix where band values have been replaced with 
        the new compared values. Band values:

        True Positive band number 1(green) both pixels are crop
        False Positive band number 2(red) correct SAT cropland, wrong SGC
        False Negative band number 3(blue) correct SGP cropland, wrong  SAT
        True Negative band number 4(black) no crop in both pixels
    '''

    new_raster_output = np.copy(sigpac_band)

    try:
        for x in tqdm(range(rows)):
            for y in range(cols):
                if sigpac_band[x, y] != 0 and classification_band[x, y] != 0:
                    # print(style_sheet[str(sigpac_band[x,y])])
                    # print(classification_band[x,y])
                    if style_sheet[str(sigpac_band[x, y])] == 6 and classification_band[x, y] == 6:
                        if style_sheet[str(sigpac_band[x, y])] == classification_band[x, y]:
                            # *True Positives (green)
                            new_raster_output[x, y] = 1

                    elif classification_band[x, y] == 6 and style_sheet[str(sigpac_band[x, y])] != 6:
                        new_raster_output[x, y] = 2  # * False Positive (red)

                    elif classification_band[x, y] != 6 and style_sheet[str(sigpac_band[x, y])] == 6:
                        new_raster_output[x, y] = 3  # * False Negatives (blue)
                    else:
                        new_raster_output[x, y] = 4  # * True Negatives (black)
    except IndexError:
        pass

    with rasterio.open(output_path, "w", **metadata) as dest:
        dest.write(new_raster_output, 1)


def barplot(labels: list, aciertos: list, fallos: list, hr: list, npixels: list, output_path: str):

    # Datos de la tabla
    # labels = ['Citricos Frutal', 'Citricos', 'Citricos-Frutal de cascara', 'Citricos-Viñedo', 'Frutal de Cascara-Frutal', 'Frutal de Cascara-Olivar', 'Frutal de Cascara', 'Frutal de Cascara-Viñedo',
    #          'Frutal', 'Imvernadero y cultivos bajo plastico', 'Olivar-Citricos', 'Olivar-Frutal', 'Olivar', 'Tierra Arable', 'Huerta', 'Frutal-Viñedo', 'Viñedo', 'Olivar-Viñedo']
    # aciertos = [89403, 5856071, 1545, 2950975, 7312, 82857, 9080721, 3806, 4774161,
    #             771053, 36820, 30066, 72449454, 105376719, 570850, 35013, 1122568, 55533]
    # fallos = [979897, 2185685, 7154, 1059, 831503, 196400, 11096748, 8630, 9937804,
    #           4243997, 73883, 63609, 90171231, 55991331, 667416, 63879, 1251335, 92121]
    # hr = [8.36, 72.82, 17.76, 99.96, 0.87, 29.67, 45.0, 30.6,
    #                       32.45, 15.37, 33.26, 32.1, 44.55, 65.3, 46.1, 35.41, 47.29, 37.61]
    # num_pixeles = [1069300, 8041756, 8699, 2952034, 838815, 279257, 20177469, 12436,
    #                14711965, 5015050, 110703, 93675, 162620685, 161368050, 1238266, 98892, 2373903, 147654]

    # Configuración del gráfico
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, aciertos, width, label='Aciertos')
    rects2 = ax.bar(x, fallos, width, label='Fallos')
    rects3 = ax.bar(x + width, hr,
                    width, label='Porcentaje de acierto')

    # Configuración de los ejes y etiquetas
    ax.set_ylabel('Cantidad')
    ax.set_title('Resultados de clasificación')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.legend()

    fig.tight_layout()

    plt.savefig(output_path+"_aciertos_fallos.png")


def create_dataframe_and_graphs(classification_band: np.ndarray, sigpac_band: np.ndarray, output_path: str) -> IO:
    '''Create a Data Frame with all the metrics information obtained.

    Args:
        classification_band (np.ndarray): Raster information Random forest classification from sentinel-2
        sigpac_band (np.ndarray): Raster information from sigpac data.
        output_path (str): Path to the directory where csv will be stored.

    Return:
        out_csv (csv): File with the final metrics of the validation.
    '''

    out_csv = pd.DataFrame(columns=["Aciertos", "Fallos", "Porcentaje de acierto", "Num Pixeles"],
                           index=["Citricos Frutal", "Citricos", "Citricos-Frutal de cascara", "Citricos-Viñedo", "Frutal de Cascara-Frutal",
                                  "Frutal de Cascara-Olivar", "Frutal de Cascara", "Frutal de Cascara-Viñedo", "Frutal", "Invernadero y cultivos bajo plastico",
                                  "Olivar-Citricos", "Olivar-Frutal", "Olivar", "Tierra Arable", "Huerta", "Frutal-Viñedo", "Viñedo", "Olivar-Viñedo"])

    crop_codes = [3, 4, 5, 6, 9, 10, 12, 13,
                  14, 16, 17, 18, 19, 23, 24, 25, 26, 27]
    truep = []
    falsen = []
    hr = []
    for crop_type in tqdm(crop_codes):

        values_cl = classification_band[np.where(sigpac_band == crop_type)]
        tp = len(np.where(values_cl == 6)[0])

        values_sg = sigpac_band[np.where(classification_band != 6)]
        fn = len(np.where(values_sg == crop_type)[0])
        truep.append(tp)
        falsen.append(fn)
        try:
            hit_rate = (tp/(tp+fn))*100
            hr.append(round(hit_rate, 3))

        except ZeroDivisionError:
            hr.append(0)

    labels = out_csv.index
    npixels = [truep[i] + falsen[i] for i in range(len(truep))]

    out_csv["Aciertos"] = truep
    out_csv["Fallos"] = falsen
    out_csv["Porcentaje de acierto"] = hr
    out_csv["Num Pixeles"] = npixels

    barplot(labels, truep, falsen, hr, npixels, "/home/jesus/Documents/TFG/satelite_images_sigpac/assets/images/img")

    print(out_csv)

    return out_csv.to_csv(output_path)


def validation(path: str) -> str:
    '''With a given compared raster, create a 2x2 confusion matrix 
    to validate the lab raster's performance crop classification.

    Args:
        path (str): Raster we want to work with.

    Returns:
        This function writes in the terminal the metrics.
    '''

    with rasterio.open(path) as src:
        band_matrix = src.read()

        tp = len(band_matrix[np.where(band_matrix == 20)])  # * Green
        tn = len(band_matrix[np.where(band_matrix == 21)])  # * Black
        fp = len(band_matrix[np.where(band_matrix == 22)])  # * Red
        fn = len(band_matrix[np.where(band_matrix == 23)])  # * Blue
        na = len(band_matrix[np.where(band_matrix == 0)])  # * White

        accuracy = (tp+tn)/(tp+tn+fp+tn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2/((1/precision)+(1/recall))
        sensitivity = tp/(tp+fn)
        specificity = fp/(fp+tn)
        tp_rate = sensitivity
        fp_rate = 1-specificity

        print("---------------METRICS---------------")
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1-Score:", f1_score)
        print("Sensitivity:", sensitivity)
        print("TruePositiveRate: ", tp_rate)
        print("FalsePositiveRate: ", fp_rate)
        print("Pixels lost in classification (Na): ", na)
        print("-------------------------------------")


# ! exec


# rows, cols, metadata, style, msk_band, sgc_band = read_needed_files(
#     "./satelite_images_sigpac/json/crop_style_sheet.json", "/home/jesus/Documents/TFG/satelite_images_sigpac/results/malaga/malagaMasked.tif", "/home/jesus/Documents/TFG/satelite_images_sigpac/results/malaga/malagaMask_sigpac.tif")

# raster_comparison(rows, cols, metadata, "/home/jesus/Documents/TFG/satelite_images_sigpac/results/malaga/malaga_redg    reen.tif",style, msk_band, sgc_band)

#create_dataframe_and_graphs(msk_band, sgc_band, "/home/jesus/Documents/TFG/satelite_images_sigpac/csv/malaga_metrics.csv")
# print("Generating True/False raster: ")
# raster_comparison(rows, cols, metadata, "/home/jesus/Documents/TFG/satelite_images_sigpac/data/red_green.tif", style, msk_band, sgc_band)

# print("Generating confusion matrix raster: ")
# raster_comparison_confmatrix(
#     rows, cols, metadata, "/home/jesus/Documents/TFG/satelite_images_sigpac/data/conf_matrix_jaen.tif", style, msk_band, sgc_band)

# print("Generating metrics: ")
# create_dataframe_metrics_crops(
#     msk_band, sgc_band, "/home/jesus/Documents/TFG/satelite_images_sigpac/data/jaen/res_jaen_metrics.csv")

# validation("/home/jesus/Documents/TFG/satelite_images_sigpac/data/FPandFN_raster_comparison_olive_jaen.tif")

# graphs()
