#!/usr/bin/python
import sys
sys.path.append("./tools/")
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
from math import sqrt
import ml_metrics as metrics

#Load clean sample data
exp_data = pd.read_csv('../exp_data/sampled/exp_data_sample_balanced.csv', delimiter=',')


#Set column names
exp_data_col_names = list(exp_data)



#Create Data & Label set
exp_data_labels = exp_data["hotel_cluster"]
drop_labels = ["date_time","srch_ci","srch_co","orig_destination_distance"]

for i in drop_labels:
    exp_data_data = exp_data.drop(drop_labels, axis=1)


"""
#Handle missing values in Training Data Set
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(exp_data_data)
exp_data_data = imp.transform(exp_data_data)

"""

columns = exp_data_data.columns


# Create train test split
#features_train, features_test, labels_train, labels_test = train_test_split(exp_data_data, exp_data_labels, test_size=0.25, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from itertools import chain

all_probs = []
unique_clusters = exp_data_data["hotel_cluster"].unique()
for cluster in unique_clusters:
    exp_data_data["target"] = 1
    exp_data_data["target"][exp_data_data["hotel_cluster"] != cluster] = 0

    predictors = [col for col in exp_data_data if col not in ['hotel_cluster', "target"]]

    probs = []

    cv = KFold(len(exp_data_data["target"]), n_folds=2)
    clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)


    for i, (tr, te) in enumerate(cv):
        clf.fit(exp_data_data[predictors].iloc[tr], exp_data_data["target"].iloc[tr])
        preds = clf.predict_proba(exp_data_data[predictors].iloc[te])
        probs.append([p[1] for p in preds])
    full_probs = chain.from_iterable(probs)
    all_probs.append(list(full_probs))

prediction_frame = pd.DataFrame(all_probs).T
prediction_frame.columns = unique_clusters

def find_top_5(row):
    return list(row.nlargest(5).index)

preds = []
for index, row in prediction_frame.iterrows():
    preds.append(find_top_5(row))

print metrics.mapk([[l] for l in exp_data_data.iloc["hotel_cluster"]], preds, k=5)