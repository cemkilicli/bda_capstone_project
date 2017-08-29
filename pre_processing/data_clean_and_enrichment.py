#!/usr/bin/python
import sys
sys.path.append("./tools/")
import pandas as pd

from pre_processing_tool import process_test
from pre_processing_tool import process_train


#from cross_validation import cross_validation_sk


exp_data_train = pd.read_csv('../exp_data/sampled/sample_train.csv', delimiter=',')
exp_data_test = pd.read_csv('../exp_data/sampled/sample_test.csv', delimiter=',')
exp_data_destinations = pd.read_csv('../exp_data/original_data/destinations.csv', delimiter=',')


"""
# Remove column stripped columns from data frame
df_drop_columns = ["date_time","srch_ci","srch_co","event_date","event_time"]
for i in df_drop_columns:
    exp_data_train = exp_data_train.drop(df_drop_columns, axis=1)
"""

process_train(exp_data_train,exp_data_destinations)
process_test(exp_data_test,exp_data_destinations)




