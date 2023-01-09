import json
from os.path import join

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# from landcoverpy.config import settings
# from landcoverpy.minio import MinioConnection
# from landcoverpy.utilities.confusion_matrix import compute_confusion_matrix

def train_model_land_cover(land_cover_dataset: str, n_jobs: int = 2):
    """Trains a Random Forest model using a land cover dataset."""

    train_df = pd.read_csv(training_dataset_path)
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
        reduced_x_train_data, y_train_data, test_size=0.15
    )

    # Train model
    clf = RandomForestClassifier(n_jobs=n_jobs)
    X_train = X_train.reindex(columns=used_columns)
    print(X_train)
    clf.fit(X_train, y_train)
    y_true = clf.predict(X_test)

    labels = y_train_data.unique()

    confusion_image_filename = "confusion_matrix.png"
    out_image_path = join(settings.TMP_DIR, confusion_image_filename)
    compute_confusion_matrix(y_true, y_test, labels, out_image_path=out_image_path)

    model_metadata = {
        "model": str(type(clf)),
        "n_jobs": n_jobs,
        "used_columns": list(used_columns),
        "classes": list(labels)
    }

    model_metadata_name = "metadata.json"

    with open(model_metadata_name, "w") as f:
        json.dump(model_metadata, f)
