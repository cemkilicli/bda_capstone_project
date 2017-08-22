import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load clean sample data
exp_data_train = pd.read_csv('../exp_data/sampled/sample_mini.csv', delimiter=',')
exp_data_destinations = pd.read_csv('../exp_data/original_data/destinations.csv', delimiter=',')

#print exp_data_train["hotel_cluster"].value_counts()

#groups_count = exp_data_train.groupby(["hotel_cluster"]).count()



### ---- Trainning Data Analysis ---- ###
#Print data frame information
print exp_data_train.info()
print exp_data_train.head(5)

#exp_data_train = exp_data_train.groupby(["hotel_cluster","srch_destination_id","srch_destination_type_id"], as_index=False).agg({"is_booking":"sum"})



sns.stripplot(x="srch_ci", y="hotel_cluster", hue="is_booking",
              data=exp_data_train, dodge=True, jitter=True,
              alpha=.25, zorder=1)
plt.show()

#print hotel_cluster

"""1.
#Plot the hotel cluster to understand the skewness of data
sns.countplot(x='hotel_cluster', data = exp_data_train)
plt.yticks(label='small')
plt.ylabel('Frequency')
plt.xticks(label='small', rotation=90)
plt.xlabel('Hotel Clusters')
plt.show()


#Plot the hotel cluster to understand the skewness of data
sns.countplot(x='is_booking', data = exp_data_train)
plt.yticks(label='small')
plt.ylabel('Frequency')
plt.xticks(label='small')
plt.xlabel('Bookings & Clicks')
plt.show()



#Find correlation
corrolations = exp_data_train.corr()['hotel_cluster'][:-1]
golden_feature_list = corrolations[abs(corrolations)].sort_values(ascending = False)
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
"""
