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
exp_data_train = pd.read_csv("../exp_data/processed/clean_sample_train.csv", delimiter=',')
exp_data_test = pd.read_csv("../exp_data/processed/clean_sample_test.csv", delimiter=',')


exp_data_train = exp_data_train.sample(frac=0.7, random_state=42)


#Create Data & Label set
exp_data_train_labels = exp_data_train["hotel_cluster"]
exp_data_test_labels = exp_data_test["hotel_cluster"]

exp_data_train_features = exp_data_train.drop(["hotel_cluster","date_time","srch_ci","srch_co","event_date","event_time"], axis=1)
exp_data_test_features = exp_data_test.drop(["hotel_cluster","date_time","srch_ci","srch_co","event_date","event_time"], axis=1)

print "Train Feature shape:",exp_data_train_features.shape,
print "Train label shape:",exp_data_train_labels.shape
print "Test Feature shape:",exp_data_test_features.shape,
print "Train label shape:",exp_data_test_labels.shape



from sklearn.svm import SVC


clf_rand = SVC(class_weight = "balanced")
clf_rand.fit(exp_data_train_features,exp_data_train_labels)

pred = clf_rand.predict(exp_data_test_features)
print pred

pred_prob = clf_rand.predict_proba(exp_data_test_features)
print pred_prob

probs = pd.DataFrame(pred_prob)
probs.columns = np.unique(exp_data_test_labels.sort_values().values)

preds = pd.DataFrame([list([r.sort_values(ascending=False)[:5].index.values]) for i,r in probs.iterrows()])

print"map@5:", mapk([[l] for l in exp_data_test_labels], preds[0], 5)
from sklearn.metrics import accuracy_score
print "accuracy is", accuracy_score(exp_data_test_labels, pred)

class_names = exp_data_train_labels.unique()

# Compute confusion matrix
cnf_matrix = confusion_matrix(exp_data_test_labels, pred)
np.set_printoptions(precision=2)


plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Random Forest - Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Random Forest - Normalized confusion matrix')

plt.show()


"""
from sklearn.metrics import classification_report

print(classification_report(exp_data_test_labels, pred, target_names=class_names))
"""


"""
default
map@5: 0.42150084118
accuracy is 0.354903963831
"""

"""
n_jobs=-1, n_estimators=150, min_samples_leaf= 50,random_state=42
map@5: 0.309244218781
accuracy is 0.18502260567

"""

"""
n_jobs=-1, n_estimators=150, min_samples_leaf= 50,random_state=42, oob_score = True, max_features="sqrt"
map@5: 0.309244218781
accuracy is 0.18502260567
"""

"""
n_jobs=-1, n_estimators=150, min_samples_leaf= 150,random_state=42, max_features="sqrt"
map@5: 0.275173377567
accuracy is 0.161243665861
"""