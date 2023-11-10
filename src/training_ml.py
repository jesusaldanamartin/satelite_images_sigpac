import json
from os.path import join

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

import joblib
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.ensemble import ForestClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

#* COMIENZA EL SCRIPT

def _feature_reduction(
    df_x: pd.DataFrame, df_y: pd.Series, percentage_columns: int = 100
):
    """Feature reduction method. Receives the training dataset and returns a set of variables."""

    if percentage_columns < 100:
        n_columns = len(df_x.columns.tolist())
        n_features = int(n_columns * percentage_columns / 100)
        model = LogisticRegression(
            penalty="elasticnet", max_iter=10000, solver="saga", n_jobs=-1, l1_ratio=0.5
        )
        rfe = RFE(estimator=model, n_features_to_select=n_features)
        fit = rfe.fit(df_x, df_y)
        used_columns = df_x.columns[fit.support_].tolist()
    else:
        used_columns = df_x.columns.tolist()

    return used_columns

def _write_cells(
    array_df, lin, col, o_text, facecolors, posi, fz, fmt, show_null_values=0
):

    """
        Config cell text and color.
    """

    text_add = []
    text_del = []
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line  and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # tots and percents
        if cell_val != 0:
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif col == ccl - 1:
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            else:
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ["%.2f%%" % per_ok, "100%"][per_ok == 100]

        # text to DEL
        text_del.append(o_text)

        # text to ADD
        font_prop = fm.FontProperties(weight="bold", size=fz)
        text_kwargs = dict(
            color="black",
            ha="center",
            va="center",
            gid="sum",
            fontproperties=font_prop,
        )
        lis_txt = ["%d" % cell_val, per_ok_s, "%.2f%%" % per_err]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy()
        dic["color"] = "g"
        lis_kwa.append(dic)
        dic = text_kwargs.copy()
        dic["color"] = "r"
        lis_kwa.append(dic)
        lis_pos = [
            (o_text._x, o_text._y - 0.3),
            (o_text._x, o_text._y),
            (o_text._x, o_text._y + 0.3),
        ]
        for i in range(len(lis_txt)):
            new_text = dict(
                x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i],
            )
            text_add.append(new_text)

        # set background color for sum cells (last line and last column)
        # doesn't work for matplotlib==3.5.1 (ours currently)
        """carr = [0.27, 0.30, 0.27, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr"""

    else:
        if per > 0:
            txt = f"$/mathbf{{{cell_val}}}$\n{per:.2f}%"
        else:
            if show_null_values == 0:
                txt = ""
            elif show_null_values == 1:
                txt = "0"
            else:
                txt = "$/mathbf{0}$\n0.0%"
        o_text.set_text(txt)

        # main diagonal
        if col == lin:
            # set color of the text in the diagonal to black
            o_text.set_color("black")
            # set background color in the diagonal to blue
            # facecolors[posi] = [0.35, 0.8, 0.55, 1.0]  # doesn't work for matplotlib==3.5.1 (ours currently)
        else:
            o_text.set_color("r")

    return text_add, text_del

def compute_confusion_matrix(y_true, y_test, labels, out_image_path):

    """
        Create confusion matrix structure and make sure it scales properly based on the number of classes.
    """

    df_cm = pd.DataFrame(confusion_matrix(y_true, y_test, labels=labels))

    #* Save the matrix in csv format in case it is needed (is not uploaded to MinIO)

    df_cm.columns = labels
    df_cm.index = labels

    df_cm.to_csv(out_image_path.replace(".png",".csv"),)

    df_len = len(df_cm) + 1

    _compute_matrix(df_cm, cmap="Oranges", out_image_path=out_image_path, figsize=(df_len, df_len))

def _compute_matrix(
    df_cm,
    annot=True,
    cmap="Oranges",
    fmt=".2f",
    fz=11,
    lw=2,
    cbar=False,
    figsize=(30, 30),
    show_null_values=0,
    pred_val_axis="y",
    out_image_path="./confusion_matrix.png",
):

    """
        Computes and saves the confusion matrix.

        params:
          df_cm          dataframe (pandas) without totals
          annot          print text in each cell
          cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu...
                         see: https://matplotlib.org/stable/tutorials/colors/colormaps.html
          fz             fontsize
          lw             linewidth
          pred_val_axis  where to show the prediction values (x or y axis)
                          'col' or 'x': show predicted values in columns (x-axis) instead lines
                          'lin' or 'y': show predicted values in lines   (y-axis)
          out_image_path path where the image will be saved
    """

    if pred_val_axis in ("col", "x"):
        xlbl = "Predicted"
        ylbl = "True"
    else:
        xlbl = "True"
        ylbl = "Predicted"
        df_cm = df_cm.T

    # create "total" row/column
    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append(item_line[1].sum())
    df_cm["sum_lin"] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc["sum_col"] = sum_col

    fig, ax1 = plt.subplots(figsize=figsize)

    ax = sn.heatmap(
        df_cm,
        annot=annot,
        annot_kws={"size": fz},
        linewidths=lw,
        ax=ax1,
        cbar=cbar,
        cmap=cmap,
        linecolor="black",
        fmt=fmt,
    )

    # set tick labels rotation and hide sum row/col label
    x_tick_labels = ax.get_xticklabels()
    x_tick_labels[-1] = ""
    y_tick_labels = ax.get_yticklabels()
    y_tick_labels[-1] = ""
    ax.set_xticklabels(x_tick_labels, fontsize=15)
    ax.set_yticklabels(y_tick_labels, fontsize=15)

    # face colors list
    # doesn't work for matplotlib==3.5.1 (ours currently)
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = []
    text_del = []
    posi = -1  # from left to right, bottom to top
    for t in ax.collections[0].axes.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1])
        col = int(pos[0])
        posi += 1

        # set text
        txt_res = _write_cells(
            array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values
        )

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item["x"], item["y"], item["text"], **item["kw"])

    # titles and legends
    ax.set_title("Confusion matrix", fontweight="bold", fontsize=17)
    ax.set_xlabel(xlbl, fontweight="bold", fontsize=16)
    ax.set_ylabel(ylbl, fontweight="bold", fontsize=16)

    plt.savefig(out_image_path, bbox_inches="tight")

#* ENTRENAMIENTO DE LOS DATOS
def train_model_land_cover(land_cover_dataset: str, n_jobs: int = 2):
    """Trains a Random Forest model using a land cover dataset."""

    train_df = pd.read_csv(land_cover_dataset)
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.fillna(np.nan)
    train_df = train_df.dropna()

    y_train_data = train_df["class"]
    x_train_data = train_df.drop(
        [
            "class",
            "latitude",
            "longitude",
            "spring_product_name",
            "autumn_product_name",
            "summer_product_name",
        ],
        axis=1,
    )

    used_columns = _feature_reduction(x_train_data, y_train_data)
    reduced_x_train_data = train_df[used_columns]

    X_train, X_test, y_train, y_test = train_test_split(
        x_train_data, y_train_data, test_size=0.15
    )

    clf = RandomForestClassifier(n_jobs=n_jobs)

    param_grid = {
        'n_estimators' : [50, 100, 20],
        'max_depth' : [None, 10, 20],
        'min_sample_split' : [2, 5, 10],
        'min_sample_leaf' : [1, 2, 4],
        'max_features' : ['auto', 'sqrt', 'log2'],
        'bootstrap' : [True, False]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_parameters = float(-9999999)
    

    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        
        for n_estimators in param_grid['n_estimators']:
                    for max_depth in param_grid['max_depth']:
                        for min_samples_split in param_grid['min_samples_split']:
                            for min_samples_leaf in param_grid['min_samples_leaf']:
                                for max_features in param_grid['max_features']:
                                    for bootstrap in param_grid['bootstrap']:
                                        # Configurar el clasificador con los hiperparámetros actuales
                                        clf.set_params(
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            bootstrap=bootstrap
                                        )
                                        #X_train_fold = X_train.reindex(columns=used_columns)


                                        # Entrenar el modelo
                                        clf.fit(X_train_fold, y_train_fold)

                                        # Evaluar en el conjunto de validación
                                        y_val_pred = clf.predict(X_val_fold)
                                        current_metric = calcular_metrica(y_val_fold, y_val_pred)

                                        # Comparar con el mejor modelo anterior
                                        if current_metric > best_metric:
                                            best_metric = current_metric
                                            best_model = clf

    # Entrenar el mejor modelo en el conjunto completo de entrenamiento
    best_model.fit(X_train, y_train)

    # Evaluar en el conjunto de prueba
    y_test_pred = best_model.predict(X_test)
    test_metric = calcular_metrica(y_test, y_test_pred)

    X_train = X_train.reindex(columns=used_columns)
    print(X_train)
    clf.fit(X_train, y_train)
    print(clf.fit(X_train, y_train))
    print(y_train)
    print("------------------------------")
    y_true = clf.predict(X_test)
    print(y_true)
    print("------------------------------")
    labels = y_train_data.unique()

    print(X_test)

    confusion_image_filename = "confusion_matrix.png"
    out_image_path = "C:\\TFG_resources\\satelite_images_sigpac\\ml\\"

    compute_confusion_matrix(y_true, y_test, labels, out_image_path=out_image_path + confusion_image_filename)

    model_metadata = {
        "model": str(type(clf)),
        "n_jobs": n_jobs,
        "used_columns": list(used_columns),
        "classes": list(labels)
    }

    model_metadata_name = "metadata.json"
    model_metadata_path = "C:\\TFG_resources\\satelite_images_sigpac\\ml\\metadata.json"

    with open("C:\\TFG_resources\\satelite_images_sigpac\\ml\\metadata.json", "w") as f:
        json.dump(model_metadata, f)

train_model_land_cover("C:\\TFG_resources\\satelite_images_sigpac\\ml\\dataset_postprocessed.csv")
