import pandas as pd
import numpy as  np
import seaborn as sns
import matplotlib.pyplot as plt

exp_data_train = pd.read_csv("../exp_data/original_data/train.csv", delimiter=',')


exp_data_train = exp_data_train[exp_data_train.orig_destination_distance.isnull() != True]
print exp_data_train.shape

exp_data_train["srch_ci"].replace("", np.nan, inplace=True)
exp_data_train["srch_co"].replace("", np.nan, inplace=True)
exp_data_train = exp_data_train.dropna(subset=["srch_ci"], how="all")
exp_data_train = exp_data_train.dropna(subset=["srch_co"], how="all")
print exp_data_train.shape

exp_data_train = exp_data_train.sample(frac=0.1)
print exp_data_train.shape

#Plot the hotel cluster to understand the skewness of data
sns.countplot(x='is_booking', data = exp_data_train)
plt.yticks(label='small')
plt.ylabel('Frequency')
plt.xticks(label='small')
plt.xlabel('Bookings & Clicks')
plt.show()


exp_data_train.to_csv("../exp_data/sampled/sample_train.csv", index=False, encoding='utf-8', index_label=True)


