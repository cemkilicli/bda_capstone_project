#!/usr/bin/python
import sys
sys.path.append("./tools/")
from sklearn.model_selection import train_test_split
import pandas as pd

#Load clean sample data
exp_data = pd.read_csv('../exp_data/sampled/exp_data_sample_balanced.csv', delimiter=',')


#Set column names
exp_data_col_names = list(exp_data)



#Create Data & Label set
exp_data_labels = exp_data["hotel_cluster"]
drop_labels = ["hotel_cluster", "date_time","srch_ci","srch_co","orig_destination_distance"]

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
features_train, features_test, labels_train, labels_test = train_test_split(exp_data_data, exp_data_labels, test_size=0.25, random_state=42)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf_rand = AdaBoostClassifier( DecisionTreeClassifier(max_depth=1), n_estimators=200,
                               algorithm="SAMME.R",
                               learning_rate=0.5)



clf_rand.fit(features_train,labels_train)

pred = clf_rand.predict(features_test)
pred_prob = clf_rand.predict_proba(features_test)

print pred
print pred_prob

from sklearn.metrics import accuracy_score
print "accuracy is", accuracy_score(labels_test, pred)
from sklearn.metrics import r2_score
print "R-squared Error",r2_score(pred,labels_test)
from sklearn.model_selection import cross_val_score
print "cross val score", cross_val_score(clf_rand, features_train,labels_train,scoring="accuracy")

from sklearn.metrics import confusion_matrix
conf_mx = confusion_matrix(labels_train, pred)

print conf_mx



importance = clf_rand.feature_importances_

print "Feature Importance"
for column in columns:
    print column, importance(column)

