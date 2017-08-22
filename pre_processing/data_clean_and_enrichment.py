#!/usr/bin/python
import sys
sys.path.append("./tools/")

import numpy as np
import pandas as pd

from pre_processing_tool import strip
from pre_processing_tool import date_subtract
from pre_processing_tool import create_month_bins
from pre_processing_tool import weekend_check
from pre_processing_tool import check_time
from pre_processing_tool import check_season
from pre_processing_tool import strip_srch
from pre_processing_tool import weight_event

#from cross_validation import cross_validation_sk


exp_data_train = pd.read_csv('../exp_data/sampled/sample_train.csv', delimiter=',')
exp_data_destinations = pd.read_csv('../exp_data/original_data/destinations.csv', delimiter=',')

print exp_data_train.shape

exp_data_train["srch_ci"].replace("", np.nan, inplace=True)
exp_data_train["srch_co"].replace("", np.nan, inplace=True)
exp_data_train = exp_data_train.dropna(subset=["srch_ci"], how="all")
exp_data_train = exp_data_train.dropna(subset=["srch_co"], how="all")


#Creating vatiables
exp_data_train["event_date"] = exp_data_train.apply(lambda row: strip(row["date_time"], "date"), axis=1)
exp_data_train["event_time"] = exp_data_train.apply(lambda row: strip(row["date_time"], "time"), axis=1)
exp_data_train["event_month"] = exp_data_train.apply(lambda row: strip(row["date_time"], "month"), axis=1)
exp_data_train["event_day"] = exp_data_train.apply(lambda row: strip(row["date_time"], "day"), axis=1)
exp_data_train["event_year"] = exp_data_train.apply(lambda row: strip(row["date_time"], "year"), axis=1)

exp_data_train["srch_ci_year"] = exp_data_train.apply(lambda row: strip_srch(row["srch_ci"], "year"), axis=1)
exp_data_train["srch_ci_month"] = exp_data_train.apply(lambda row: strip_srch(row["srch_ci"], "month"), axis=1)
exp_data_train["srch_ci_day"] = exp_data_train.apply(lambda row: strip_srch(row["srch_ci"], "day"), axis=1)
exp_data_train["srch_ci_season_bin"] = exp_data_train.apply(lambda row: create_month_bins(row["srch_ci_month"]), axis=1)

exp_data_train["srch_co_year"] = exp_data_train.apply(lambda row: strip_srch(row["srch_co"], "year"), axis=1)
exp_data_train["srch_co_month"] = exp_data_train.apply(lambda row: strip_srch(row["srch_co"], "month"), axis=1)
exp_data_train["srch_co_day"] = exp_data_train.apply(lambda row: strip_srch(row["srch_co"], "day"), axis=1)
exp_data_train["srch_co_season_bin"] = exp_data_train.apply(lambda row: create_month_bins(row["srch_co_month"]), axis=1)

exp_data_train["srch_ci_is_winter"] = exp_data_train.apply(lambda row: check_season(row["srch_ci_season_bin"],"winter"), axis=1)
exp_data_train["srch_ci_is_spring"] = exp_data_train.apply(lambda row: check_season(row["srch_ci_season_bin"],"spring"), axis=1)
exp_data_train["srch_ci_is_summer"] = exp_data_train.apply(lambda row: check_season(row["srch_ci_season_bin"],"summer"), axis=1)
exp_data_train["srch_ci_is_fall"] = exp_data_train.apply(lambda row: check_season(row["srch_ci_season_bin"],"fall"), axis=1)

exp_data_train["srch_co_is_winter"] = exp_data_train.apply(lambda row: check_season(row["srch_co_season_bin"],"winter"), axis=1)
exp_data_train["srch_co_is_spring"] = exp_data_train.apply(lambda row: check_season(row["srch_co_season_bin"],"spring"), axis=1)
exp_data_train["srch_co_is_summer"] = exp_data_train.apply(lambda row: check_season(row["srch_co_season_bin"],"summer"), axis=1)
exp_data_train["srch_co_is_fall"] = exp_data_train.apply(lambda row: check_season(row["srch_co_season_bin"],"fall"), axis=1)

exp_data_train["night_of_stay"] = exp_data_train.apply(lambda row: date_subtract(row["srch_ci"], row["srch_co"]), axis=1)
exp_data_train["night_of_stay"] = exp_data_train["night_of_stay"] / np.timedelta64(1, 'D')
#exp_data_train["booking_window"] = exp_data_train["event_date"]- exp_data_train["srch_ci"]

exp_data_train["adult_per_room"] = (exp_data_train["srch_adults_cnt"]/exp_data_train["srch_rm_cnt"])

exp_data_train["children_per_room"] = (exp_data_train["srch_children_cnt"] / exp_data_train["srch_rm_cnt"])
exp_data_train["person_per_room"] = ((exp_data_train["srch_children_cnt"] + exp_data_train["srch_adults_cnt"])/exp_data_train["srch_rm_cnt"])
exp_data_train["event_season_bin"] = exp_data_train.apply(lambda row: create_month_bins(row["event_month"]), axis=1)

exp_data_train["adult_per_room"] =  exp_data_train["adult_per_room"].round(2)
exp_data_train["children_per_room"] = exp_data_train["children_per_room"].round(2)
exp_data_train["person_per_room"] = exp_data_train["person_per_room"].round(2)
exp_data_train["event_season_bin"] = exp_data_train["event_season_bin"].round(2)

exp_data_train["event_is_winter"] = exp_data_train.apply(lambda row: check_season(row["event_season_bin"],"winter"), axis=1)
exp_data_train["event_is_spring"] = exp_data_train.apply(lambda row: check_season(row["event_season_bin"],"spring"), axis=1)
exp_data_train["event_is_summer"] = exp_data_train.apply(lambda row: check_season(row["event_season_bin"],"summer"), axis=1)
exp_data_train["event_is_fall"] = exp_data_train.apply(lambda row: check_season(row["event_season_bin"],"fall"), axis=1)

#### This need to be check before final submittion ####
exp_data_train["event_weekday"] = exp_data_train["event_date"].apply(lambda x: x.weekday())
exp_data_train["weekend_event"] = exp_data_train.apply(lambda row: weekend_check(row["event_weekday"]), axis=1)

exp_data_train["evet_is_late_night"] = exp_data_train.apply(lambda row: check_time(row["event_time"],"late_night"), axis=1)
exp_data_train["evet_is_early_morning"] = exp_data_train.apply(lambda row: check_time(row["event_time"],"early_morning"), axis=1)
exp_data_train["evet_is_morning"] = exp_data_train.apply(lambda row: check_time(row["event_time"],"morning"), axis=1)
exp_data_train["evet_is_mid_day"] = exp_data_train.apply(lambda row: check_time(row["event_time"],"mid_day"), axis=1)
exp_data_train["evet_is_afternoon"] = exp_data_train.apply(lambda row: check_time(row["event_time"],"afternoon"), axis=1)
exp_data_train["evet_is_evening"] = exp_data_train.apply(lambda row: check_time(row["event_time"],"evening"), axis=1)
exp_data_train["evet_is_night"] = exp_data_train.apply(lambda row: check_time(row["event_time"],"night"), axis=1)

pd.to_numeric(exp_data_train["night_of_stay"])

exp_data_train = exp_data_train.dropna(subset=["adult_per_room"], how="all")
exp_data_train = exp_data_train.dropna(subset=["children_per_room"], how="all")
exp_data_train = exp_data_train.dropna(subset=["person_per_room"], how="all")

#exp_data_train['orig_destination_distance'] = exp_data_train['orig_destination_distance'].fillna(-1)

exp_data_train["event_weight"] = exp_data_train.apply(lambda row: weight_event(row["is_booking"]), axis=1 )

# Decopose destination data
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
exp_data_destinations_pca = pca.fit_transform(exp_data_destinations[["d{0}".format(i + 1) for i in range(149)]])
exp_data_destinations_pca = pd.DataFrame(exp_data_destinations_pca)
exp_data_destinations_pca["srch_destination_id"] = exp_data_destinations["srch_destination_id"]

print exp_data_destinations_pca

# Join Destinations and train set
#exp_data_train = exp_data_train.join(exp_data_destinations_pca, on="srch_destination_id", how='left', rsuffix="dest")
exp_data_train = pd.merge(exp_data_train,exp_data_destinations_pca , on="srch_destination_id", how='left')

print exp_data_train.info()

"""
# Remove column stripped columns from data frame
df_drop_columns = ["date_time","srch_ci","srch_co","event_date","event_time"]
for i in df_drop_columns:
    exp_data_train = exp_data_train.drop(df_drop_columns, axis=1)
"""

print exp_data_train.info()

# Export data to new csv
exp_data_train.to_csv("../exp_data/processed/clean_sample_train.csv", index=False, encoding='utf-8', index_label=True,)
