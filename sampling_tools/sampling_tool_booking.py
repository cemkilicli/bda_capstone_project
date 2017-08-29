import pandas as pd
import numpy as  np
import seaborn as sns
import matplotlib.pyplot as plt

exp_data_train = pd.read_csv("../exp_data/original_data/train.csv", delimiter=',')


exp_data_train = exp_data_train[exp_data_train.orig_destination_distance.isnull() != True]

exp_data_train["srch_ci"].replace("", np.nan, inplace=True)
exp_data_train["srch_co"].replace("", np.nan, inplace=True)
exp_data_train = exp_data_train.dropna(subset=["srch_ci"], how="all")
exp_data_train = exp_data_train.dropna(subset=["srch_co"], how="all")

exp_data_sample_train = exp_data_train.sample(frac=0.2, random_state=42)
print exp_data_sample_train.shape

exp_data_test = exp_data_train[exp_data_train.is_booking == True]
exp_data_sample_test = exp_data_train.sample(frac=0.05, random_state=52)
print exp_data_sample_test.shape

exp_data_sample_train.to_csv("../exp_data/sampled/sample_train.csv", index=False, encoding='utf-8', index_label=True)
exp_data_sample_test.to_csv("../exp_data/sampled/sample_test.csv", index=False, encoding='utf-8', index_label=True)


