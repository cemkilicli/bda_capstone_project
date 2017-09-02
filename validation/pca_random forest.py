print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#Load clean sample data
exp_data_train = pd.read_csv("../exp_data/processed/clean_sample_train.csv", delimiter=',')
exp_data_test = pd.read_csv("../exp_data/processed/clean_sample_test.csv", delimiter=',')


exp_data_train = exp_data_train.sample(frac=0.7, random_state=42)


#Create Data & Label set
exp_data_train_labels = exp_data_train["hotel_cluster"]
exp_data_test_labels = exp_data_test["hotel_cluster"]

exp_data_train_features = exp_data_train.drop(["hotel_cluster","date_time","srch_ci","srch_co","event_date","event_time"], axis=1)
exp_data_test_features = exp_data_test.drop(["hotel_cluster","date_time","srch_ci","srch_co","event_date","event_time"], axis=1)




from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

# Plot the PCA spectrum
pca.fit(exp_data_train_features)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

# Prediction
n_components = [20, 40, 61]
Cs = np.logspace(1, 1, 1)

estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))

estimator.fit(exp_data_train_features, exp_data_train_labels)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()