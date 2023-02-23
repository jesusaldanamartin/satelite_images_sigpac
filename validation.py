from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Tuple
import rasterio
import json


def read_needed_files(json_path: str, masked_path: str, sigpac_path: str):
    '''Read all files needed and call function raster_comparison_cropland() to compare both rasters.

    Args:
        json_path (str): Path to the json file.
        sigpac_path (str): Path to the sigpac processed raster.
        masked_path (str): Path to the classification raster (LAB).

    Returns:
        style_sheet (dict): Data obtained from json_path. 
        sigpac_band (np.ndarray): Raster data read with rasterio.
        classification_band (np.ndarray): Raster data read with rasterio.
    '''

    with open(json_path) as jfile:
        dict_json = json.load(jfile)
        style_sheet = dict_json['style_sheet']['SIGPAC_code']

    with rasterio.open(sigpac_path) as src:
        sigpac_band = src.read(1)
        rows_sgc = sigpac_band.shape[0]  # * 10654
        cols_sgc = sigpac_band.shape[1]  # * 16555

    with rasterio.open(masked_path) as src2:
        classification_band = src2.read(1)
        rows_msk = classification_band.shape[0]  # * 10654
        cols_msk = classification_band.shape[1]  # * 16555
        out_meta = src2.meta

    try:
        if (rows_msk != rows_sgc or cols_msk != cols_sgc):
            raise ValueError
        else:
            return rows_msk, cols_msk, out_meta, style_sheet, classification_band, sigpac_band
    except ValueError as err:
        print(err + ": Both rasters must have the same size")


# style_sheet, sigpac_band, classification_band = apply_style_sheet_to_raster("json/olive_style_sheet.json",
#     "./results/cadiz/cadizMask_sigpac.tif",
#     "./results/cadiz/cadizMask.tif")

def raster_comparison(rows: int, cols: int, metadata, output_path: str,
                      style_sheet, sigpac_band, classification_band) -> np.ndarray:
    '''This function compares the band values of two different raster. These values 
    are linked with the crop_style_sheet.json file. Both rasters must have the same size.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        new_raster_output (np.ndarray): 2D numpy array copy of our input raster.
        style_sheet (dict): Path to the json file.
        sigpac_band (ndarray): Sigpac raster band read with rasterio.
        classification_band (ndarray): Lab raster band read with rasterio.

    Returns:
        This function returns a ndarray where band values have been replaced with 
        the new compared values.
    '''

    new_raster_output = np.copy(sigpac_band)

    try:
        for x in tqdm(range(rows)):
            for y in range(cols):
                if sigpac_band[x, y] != 0 and classification_band[x, y] != 0:
                    if len(style_sheet[str(sigpac_band[x, y])]) > 1:
                        for item in style_sheet[str(sigpac_band[x, y])]:
                            # print(item)
                            # print(style_sheet[str(sigpac_band[x,y])])
                            # print(classification_band[x,y])
                            if classification_band[x, y] == item:
                                #print("OK",":",item)
                                # * same band value
                                new_raster_output[x, y] = 20
                            else:
                                #print("WRONG",":",classification_band[x,y]," distinto ",style_sheet[str(sigpac_band[x,y])])
                                # * diff band value
                                new_raster_output[x, y] = 21

                    else:
                        if style_sheet[str(sigpac_band[x, y])] == classification_band[x, y]:
                            #print("OK 2",":", classification_band[x,y])
                            new_raster_output[x, y] = 20  # * same band value

                        else:
                            #print("WRONG 2",":",classification_band[x,y]," distinto ",style_sheet[str(sigpac_band[x,y])])
                            new_raster_output[x, y] = 21  # * diff band value
    except IndexError:
        pass

    with rasterio.open(output_path, "w", **metadata) as dest:
        dest.write(new_raster_output, 1)


def raster_comparison_cropland(rows: int, cols: int, metadata, output_path,
                               style_sheet, sigpac_band, classification_band):
    '''This function compares the crop zones in both land covers given by parameters.
    These values are linked with the id_style_sheet.json file. Both rasters must have the same size.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        new_raster_output (np.ndarray): 2D numpy array copy of our input raster.
        style_sheet (dict): Path to the json file.
        sigpac_band (ndarray): Sigpac raster band read with rasterio.
        classification_band (ndarray): Lab raster band read with rasterio.

    Returns:
        This function returns a ndarray matrix where band values have been replaced with 
        the new compared values.
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
                            #print("VERDE")
                            new_raster_output[x, y] = 20  # *True Positives

                    elif classification_band[x, y] == 6 and style_sheet[str(sigpac_band[x, y])] != 6:
                        #print("ROJO")
                        new_raster_output[x, y] = 21  # * False Positive

                    elif classification_band[x, y] != 6 and style_sheet[str(sigpac_band[x, y])] == 6:
                        #print("AZUL")
                        new_raster_output[x, y] = 22  # * False Negatives
                    else:
                        #print("NEGRO")
                        new_raster_output[x, y] = 23  # * True Negatives
    except IndexError:
        pass

    with rasterio.open(output_path, "w", **metadata) as dest:
        dest.write(new_raster_output, 1)


def specific_class_raster_comparison(rows: int, cols: int, new_raster_output,
                                     style_sheet, sigpac_band, classification_band):
    '''This function compares the crop zones in both land covers given by parameters.
    These values are linked with the id_style_sheet.json file. Both rasters must have the same size.

    Args:
        rows (int): Number of rows.
        cols (int): Number of columns.
        new_raster_output (np.ndarray): 2D numpy array copy of our input raster.
        style_sheet (dict): Path to the json file.
        sigpac_band (ndarray): Sigpac raster band read with rasterio.
        classification_band (ndarray): Lab raster band read with rasterio.

    Returns:
        This function returns a ndarray matrix where band values have been replaced with 
        the new compared values.
    '''

    #! TP Acierto seguro (es olivo en sigpac y crop en classification)
    #! FP Cropland no olivo (es cropland pero no mi clase concreta)
    #! TN No cropland no olivo (no es cropland en classification ni olivo en SIGPAC)
    #! FN Fallo seguro (No es ni cropland en sigpac ni forest en classification)
    olive = [10, 17, 18, 19, 27]
    try:
        for x in tqdm(range(rows)):
            for y in range(cols):
                if sigpac_band[x, y] != 0 and classification_band[x, y] != 0:
                    # print(style_sheet[str(sigpac_band[x,y])])
                    # # print(classification_band[x,y])
                    # print(sigpac_band[x,y])
                    # print(style_sheet[str(sigpac_band[x,y])])
                    if sigpac_band[x, y] == 19 and classification_band[x, y] == 6:
                        # print("TP = VERDE")
                        new_raster_output[x, y] = 20  # *True Positives

                    elif classification_band[x, y] == 6 and (6 in style_sheet[str(sigpac_band[x, y])] and sigpac_band[x, y] != 19):
                        # print("FP = ROJO")
                        new_raster_output[x, y] = 21  # * False Positive

                    elif classification_band[x, y] != 6 and (6 in style_sheet[str(sigpac_band[x, y])] and sigpac_band[x, y] == 19):
                        # print("FN = AZUL")
                        new_raster_output[x, y] = 22  # * False Negatives
                    else:
                        # print("TN = NEGRO")
                        new_raster_output[x, y] = 23  # * True Negatives
    except IndexError:
        pass
    return new_raster_output


# *  TP  Banda numero 20(verde) coinciden SGP y SAT
# *  TN  Banda numero 23(negro)  SAT cropland SGP no
# *  FP  Banda numero 21(rojo)  No cropland n
# *  FN  Banda numero 22(azul)  SGP cropland SAT ni en SAT ni SG


def crop_metrics(sigpac_band, classification_band, output_path):
    '''Create a Data Frame with all the metrics information obtained.

    Args:
        sigpac_band (np.ndarray): Raster information from sigpac data.
        classification_band (np.ndarray): Raster information Random forest classification from sentinel-2
        output_path (str): Path to the directory where csv will be stored.

    Return:
        None
    '''

    data_output = pd.DataFrame(columns=["Citricos Frutal", "Citricos", "Citricos-Frutal de cascara", "Citricos-Viñedo", "Frutal de Cascara-Frutal",
                                        "Frutal de Cascara-Olivar", "Frutal de Cascara", "Frutal de Cascara-Viñedo", "Frutal", "Imvernadero y cultivos bajo plastico",
                                        "Olivar-Citricos", "Olivar-Frutal", "Olivar", "Tierra Arable", "Huerta", "Frutal-Viñedo", "Viñedo", "Olivar-Viñedo"],
                               index=["TP", "FN", "Hit rate"])

    crop_codes = [3, 4, 5, 6, 9, 10, 12, 13,
                  14, 16, 17, 18, 19, 23, 24, 25, 26, 27]
    truep = []
    falsen = []
    hr = []
    for crop_type in crop_codes:
        #        print(crop_type)
        # print(cont)
        index_code_sigpac = np.where(sigpac_band == crop_type)
        values_cl = classification_band[index_code_sigpac]
        tp = len(np.where(values_cl == 6)[0])
        truep.append(tp)
        # data_output.loc['TP',cont] = tp
        print(len(index_code_sigpac))
        print(len(values_cl))
        index_not_crop = np.where(classification_band != 6)
        print(len(index_not_crop))
        values_sg = sigpac_band[index_not_crop]
        fn = len(np.where(values_sg == crop_type)[0])
        falsen.append(fn)
        # data_output.loc['FN',cont] = fn
        try:
            hit_rate = tp/(tp+fn)
            hr.append(hit_rate)

        except ZeroDivisionError:
            pass
        # data_output.loc['Hit rate',cont] = hit_rate

        # data_output.loc["TP"] = truep
        # data_output.loc["FN"] = falsen
        # data_output.loc["Hit rate"] = hr

        print(data_output)
        data_output.to_csv(output_path)

# crop_metrics(sigpac_band, classification_band)


def process_dataframe(data_path):
    '''Given the dataframe created in crop_metrics() function the df is processed. 
    It now stores the hit percentage and number of pixels

    Args:
        data_path (str): Path to the dataframe.

    Return:
        The new data will be printed in terminal.
    '''
    dataframe = pd.read_csv(data_path)

    tp = (0, 3, 6, 9, 12, 15, 18, 21)
    tn = (1, 4, 7, 10, 13, 16, 19, 22)
    aciertos = 0
    fallos = 0
    row = 1
    for i in range(18):
        aciertos = 0
        fallos = 0
        porcentaje = 0
        num_pixeles = 0
        for num in tp:
            aciertos += dataframe[dataframe.columns[row]][num]
        for num in tn:
            fallos += dataframe[dataframe.columns[row]][num]
        porcentaje = ((aciertos)/(aciertos+fallos))*100
        num_pixeles = aciertos+fallos
        print(dataframe.columns[row], ",", int(aciertos), ",", int(
            fallos), ",", round(porcentaje, 2), ",", int(num_pixeles))
        row += 1

# process_dataframe("./csv/andalucia_tp_tn.csv")

# "./raster_comparison_malaga.tif"


def validation(path: str) -> Tuple[float, float]:
    '''With a given compared raster, create a 2x2 confusion matrix 
    to validate the lab raster's performance crop classification.

    Args:
        path (str): Raster we want to work with.

    Returns:
        This function writes in the terminal the metrics.
    '''

    with rasterio.open(path) as src:
        band_matrix = src.read()
        # rows = band_matrix.shape[0] #* 10654
        # cols = band_matrix.shape[1] #* 16555
        green = np.where(band_matrix == 20)
        red = np.where(band_matrix == 21)
        blue = np.where(band_matrix == 22)
        black = np.where(band_matrix == 23)
        white = np.where(band_matrix == 0)

        tp = len(band_matrix[green])
        tn = len(band_matrix[black])
        fp = len(band_matrix[red])
        fn = len(band_matrix[blue])
        na = len(band_matrix[white])

        print(tp)
        print(tn)
        print(fp)
        print(fn)
        print(na)
        print("-----------------")
        # print(len(band_matrix[green]))
        # print(band_matrix[green])
        accuracy = (tp+tn)/(tp+tn+fp+tn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2/((1/precision)+(1/recall))
        sensitivity = tp/(tp+fn)
        specificity = fp/(fp+tn)
        tp_rate = sensitivity
        fp_rate = 1-specificity

        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1-Score:", f1_score)
        print("Sensitivity:", sensitivity)
        print("TruePositiveRate: ", tp_rate)
        print("FalsePositiveRate: ", fp_rate)
        print("-------------------------")

    return fp_rate, tp_rate

# x,y = validation("C:\\TFG_resources\\satelite_images_sigpac\\results\\malaga\\raster_comparison_malaga.tif")
# x2,y2 = validation("C:\\TFG_resources\\satelite_images_sigpac\\results\\raster_comparison_cordoba2.tif")
# x3,y3 = validation("C:\\TFG_resources\\satelite_images_sigpac\\results\\raster_comparison_granada.tif")
# x4,y4 = validation("C:\\TFG_resources\\satelite_images_sigpac\\results\\sevilla\\raster_comparison_sevilla.tif")
# x_oliv,y_oliv = validation("/home/jesus/Documents/satelite_images_sigpac/FPandFN_raster_comparison_olive_jaen.tif")

#!---------------------------------------------------------------------------------------------------------------------------------------


def graphs():
    # with open("/home/jesus/Documents/satelite_images_sigpac/csv/andalucia.csv", mode='r') as file:
    #     df = pd.read_csv(file)
    #     print(df.iloc[[1]])
    #     plt.bar(df.iloc[[3]],df.iloc[[0]],'o')
    #     plt.show()
    return

# graphs()

# plt.plot(x2,y2,'o')
# plt.plot(x3,y3,'o')
# plt.plot(x4,y4,'o')
# plt.xlim((0,1))
# plt.ylim((0,1))
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.show()
