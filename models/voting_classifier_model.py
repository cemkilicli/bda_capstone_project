#!/usr/bin/python
import sys
sys.path.append("./tools/")
import pandas as pd
import numpy as np
from ml_metrics import mapk
from sklearn.metrics import accuracy_score

#Load clean sample data
exp_data_train = pd.read_csv("../exp_data/processed/clean_sample_train.csv", delimiter=',')
exp_data_test = pd.read_csv("../exp_data/processed/clean_sample_test.csv", delimiter=',')


exp_data_train = exp_data_train.sample(frac=0.5, random_state=0)


#Create Data & Label set
exp_data_train_labels = exp_data_train["hotel_cluster"]
exp_data_test_labels = exp_data_test["hotel_cluster"]

exp_data_train_features = exp_data_train.drop(["hotel_cluster","date_time","srch_ci","srch_co","event_date","event_time"], axis=1)
exp_data_test_features = exp_data_test.drop(["hotel_cluster","date_time","srch_ci","srch_co","event_date","event_time"], axis=1)

print "Train Feature shape:",exp_data_train_features.shape,
print "Train label shape:",exp_data_train_labels.shape
print "Test Feature shape:",exp_data_test_features.shape,
print "Train label shape:",exp_data_test_labels.shape,


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

clf_rand = RandomForestClassifier(n_jobs=-1, class_weight = "balanced")
clf_tre = DecisionTreeClassifier()
clf_knn = KNeighborsClassifier(weights="distance", algorithm="kd_tree", n_jobs = -1)

voting_clf = VotingClassifier(
    estimators=[("rf", clf_rand), ("tree", clf_tre), ("knn", clf_knn)],
    voting= "hard",
)

voting_clf.fit(exp_data_train_features, exp_data_train_labels)


pred = voting_clf.predict(exp_data_test_features)
print pred

pred_prob = voting_clf.predict_proba(exp_data_test_features)
print pred_prob

probs = pd.DataFrame(pred_prob)
probs.columns = np.unique(exp_data_test_labels.sort_values().values)

preds = pd.DataFrame([list([r.sort_values(ascending=False)[:5].index.values]) for i,r in probs.iterrows()])

print"map@5:", mapk([[l] for l in exp_data_test_labels], preds[0], 5)
from sklearn.metrics import accuracy_score
print "accuracy is", accuracy_score(exp_data_test_labels, pred)




"""

"""
