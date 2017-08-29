import pandas as pd
import numpy as np

def make_features_ready(data_frame):
    data_frame = data_frame.dropna(subset=["0"], how="all")
    data_frame = data_frame.dropna(subset=["1"], how="all")
    data_frame = data_frame.dropna(subset=["2"], how="all")

    print data_frame.isnull().any()
    print data_frame.info()

    # Set column names
    exp_data_col_names = list(data_frame)

    # Create Data & Label set
    exp_data_labels = data_frame["hotel_cluster"]

    # exp_data['orig_destination_distance'] = exp_data['orig_destination_distance'].fillna(-1)

    drop_labels = ["hotel_cluster", "date_time", "srch_ci", "srch_co", "event_date", "event_time"]

    for i in drop_labels:
        exp_data_features = data_frame.drop(drop_labels, axis=1)

    return exp_data_features, exp_data_labels




