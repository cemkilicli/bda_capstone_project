#!/usr/bin/python
import sys
sys.path.append("./tools/")
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_confusion_matrix import plot_confusion_matrix

#Load clean sample data
exp_data = pd.read_csv('../exp_data/clean_data/clean_sample.csv', delimiter=',')

#Print data frame information
print exp_data.info()

#Set column names
exp_data_col_names = list(exp_data)



#Create Data & Label set
exp_data_labels = exp_data["hotel_cluster"]
exp_data_data = exp_data.drop("hotel_cluster", axis=1)

# Create train test split
features_train, features_test, labels_train, labels_test = train_test_split(exp_data_data, exp_data_labels, test_size=0.25, random_state=42)


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


# Print Confusion Matrix
class_names = [ "Actual", "Predicted"]
cnf_matrix = confusion_matrix(labels_test, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization - Gusian NB')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix - Gusian NB')
plt.show()


from sklearn.metrics import accuracy_score
# Print Accuracy Score
print "Accuracy is", accuracy_score(pred,labels_test)
print "The number of correct predictions is", accuracy_score(pred,labels_test, normalize=False)
print "Total sample used is", len(pred)  # number of all of the predictions