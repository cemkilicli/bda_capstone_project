import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load clean sample data
exp_data_train = pd.read_csv('../exp_data/sampled_data/sample_booked.csv', delimiter=',')
exp_data_destinations = pd.read_csv('../exp_data/raw_data/destinations.csv', delimiter=',')

### ---- Trainning Data Analysis ---- ###
#Print data frame information
print exp_data_train.info()


#Plot the SalePrice to understand the skewness of data
sns.countplot(x='hotel_cluster', data = exp_data_train, color = 'g')
plt.yticks(label='small')
plt.xticks(label='small')
plt.show()


#Find correlation
corrolations = exp_data_train.corr()['hotel_cluster'][:-1]
golden_feature_list = corrolations[abs(corrolations) > 0.5].sort_values(ascending = False)
print("There is {} strongly correlated values with hotel_cluster:\n{}".format(len(golden_feature_list), golden_feature_list))


#Select numerical data types
set(exp_data_train.dtypes.tolist())
dfnum = exp_data_train.select_dtypes(include = ['float64', 'int64'])
print dfnum.info()

#Create Heatmap of Correlated variables
dfnum_corr1 = dfnum.corr()
cols = dfnum_corr1.nlargest(10, 'hotel_cluster')['hotel_cluster'].index
cm = np.corrcoef(dfnum[cols].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.yticks(rotation=30, label='small')
plt.xticks(rotation=30, label='small')

plt.show()

### ---- Destinations Data Analysis ---- ###

#Print data frame information
print exp_data_destinations.info()

#Print data frame shape
print exp_data_destinations.shape

#Print data frame shape
print exp_data_destinations.head(5)

#Select numerical data types
set(exp_data_destinations.dtypes.tolist())
dfnum_dest = exp_data_destinations.select_dtypes(include = ['float64', 'int64'])
print dfnum_dest.info()




