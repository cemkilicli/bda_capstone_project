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
from ml_metrics import mapk

#Load clean sample data
exp_data = pd.read_csv("../exp_data/processed/clean_sample_train.csv", delimiter=',')

exp_data = exp_data.dropna(subset=["0"], how="all")
exp_data = exp_data.dropna(subset=["1"], how="all")
exp_data = exp_data.dropna(subset=["2"], how="all")



#Set column names
exp_data_col_names = list(exp_data)


#Create Data & Label set
exp_data_labels = exp_data["hotel_cluster"]

#exp_data['orig_destination_distance'] = exp_data['orig_destination_distance'].fillna(-1)

drop_labels = ["hotel_cluster", "date_time","srch_ci","srch_co","event_date","event_time"]

for i in drop_labels:
    exp_data_data = exp_data.drop(drop_labels, axis=1)

columns = exp_data_data.columns

# Create train test split
features_train, features_test, labels_train, labels_test = train_test_split(exp_data_data, exp_data_labels, test_size=0.25, random_state=42)

from sklearn.linear_model import LogisticRegression

for multi_class in ('multinomial', 'ovr'):
    clf = LogisticRegression(solver='sag', max_iter=100, random_state=42,
                             multi_class=multi_class).fit(features_train, labels_train)

    print("training score : %.3f (%s)" % (clf.score(features_train, labels_train), multi_class))

    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    pred_prob = clf.predict_proba(features_test)
    probs = pd.DataFrame(pred_prob)
    preds = pd.DataFrame([list([r.sort_values(ascending=False)[:5].index.values]) for i, r in probs.iterrows()])

    print"map@5:", mapk([[l] for l in labels_test], preds[0], 5)
    from sklearn.metrics import accuracy_score

    print "accuracy is", accuracy_score(labels_test, pred)


