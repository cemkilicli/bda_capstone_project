from operator import index

import numpy as np
import pandas as pd


exp_raw_train_data = pd.read_csv('../exp_data/original_data/train.csv', delimiter=',')

print exp_raw_train_data.shape


exp_train_data_booked = exp_raw_train_data[exp_raw_train_data.is_booking == 0]
exp_train_data_not_booked = exp_raw_train_data[exp_raw_train_data.is_booking == 1]

exp_train_data_booked_sampled = exp_train_data_booked.sample(2000000)

exp_train_data_not_booked_sampled = exp_train_data_not_booked.sample(2000000)

print "exp_train_data_booked shape", exp_train_data_booked.shape
print "exp_train_data_booked head", exp_train_data_booked.head(5)


print "exp_train_data_not_booked_sampled shape", exp_train_data_not_booked_sampled.shape
print "exp_train_data_not_booked_sampled head",exp_train_data_not_booked_sampled.head(5)

frames = [exp_train_data_booked_sampled,exp_train_data_not_booked_sampled]

exp_data = pd.concat(frames)

print "exp_data shape", exp_data.shape
print "exp_data head", exp_data.head(5)


"""
exp_train_data_sample = exp_train_data.sample(frac=0.1)

print exp_train_data_sample.shape

"""


exp_data.to_csv("exp_data_sample_balanced.csv", index=False, encoding='utf-8', index_label=True,)