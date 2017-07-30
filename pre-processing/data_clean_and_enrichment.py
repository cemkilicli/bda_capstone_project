import numpy as np
import pandas as pd
from preprocess_tools import strip
from preprocess_tools import date_subtract
from preprocess_tools import create_month_bins
from preprocess_tools import weekend_check
from preprocess_tools import check_time
from preprocess_tools import check_season
from preprocess_tools import strip_srch
#from cross_validation import cross_validation_sk


exp_data = pd.read_csv('../exp_data/sample_mini.csv', delimiter=',')

print exp_data.shape

exp_data["srch_ci"].replace("", np.nan, inplace=True)
exp_data["srch_co"].replace("", np.nan, inplace=True)
exp_data = exp_data.dropna(subset=["srch_ci"], how="all")
exp_data = exp_data.dropna(subset=["srch_co"], how="all")



#Creating vatiables
exp_data["event_date"] = exp_data.apply(lambda row: strip(row["date_time"], "date"), axis=1)
exp_data["event_time"] = exp_data.apply(lambda row: strip(row["date_time"], "time"), axis=1)
exp_data["event_month"] = exp_data.apply(lambda row: strip(row["date_time"], "month"), axis=1)
exp_data["event_day"] = exp_data.apply(lambda row: strip(row["date_time"], "day"), axis=1)
exp_data["event_year"] = exp_data.apply(lambda row: strip(row["date_time"], "year"), axis=1)

exp_data["srch_ci_year"] = exp_data.apply(lambda row: strip_srch(row["srch_ci"], "year"), axis=1)
exp_data["srch_ci_month"] = exp_data.apply(lambda row: strip_srch(row["srch_ci"], "month"), axis=1)
exp_data["srch_ci_day"] = exp_data.apply(lambda row: strip_srch(row["srch_ci"], "day"), axis=1)
exp_data["srch_ci_season_bin"] = exp_data.apply(lambda row: create_month_bins(row["srch_ci_month"]), axis=1)

exp_data["srch_co_year"] = exp_data.apply(lambda row: strip_srch(row["srch_co"], "year"), axis=1)
exp_data["srch_co_month"] = exp_data.apply(lambda row: strip_srch(row["srch_co"], "month"), axis=1)
exp_data["srch_co_day"] = exp_data.apply(lambda row: strip_srch(row["srch_co"], "day"), axis=1)
exp_data["srch_co_season_bin"] = exp_data.apply(lambda row: create_month_bins(row["srch_co_month"]), axis=1)

exp_data["srch_ci_is_winter"] = exp_data.apply(lambda row: check_season(row["srch_ci_season_bin"],"winter"), axis=1)
exp_data["srch_ci_is_spring"] = exp_data.apply(lambda row: check_season(row["srch_ci_season_bin"],"spring"), axis=1)
exp_data["srch_ci_is_summer"] = exp_data.apply(lambda row: check_season(row["srch_ci_season_bin"],"summer"), axis=1)
exp_data["srch_ci_is_fall"] = exp_data.apply(lambda row: check_season(row["srch_ci_season_bin"],"fall"), axis=1)

exp_data["srch_co_is_winter"] = exp_data.apply(lambda row: check_season(row["srch_co_season_bin"],"winter"), axis=1)
exp_data["srch_co_is_spring"] = exp_data.apply(lambda row: check_season(row["srch_co_season_bin"],"spring"), axis=1)
exp_data["srch_co_is_summer"] = exp_data.apply(lambda row: check_season(row["srch_co_season_bin"],"summer"), axis=1)
exp_data["srch_co_is_fall"] = exp_data.apply(lambda row: check_season(row["srch_co_season_bin"],"fall"), axis=1)

exp_data["room_night"] = exp_data.apply(lambda row: date_subtract(row["srch_ci"], row["srch_co"]), axis=1)
exp_data["room_night"] = exp_data["room_night"]/ np.timedelta64(1, 'D')
exp_data["adult_per_room"] = (exp_data["srch_adults_cnt"]/exp_data["srch_rm_cnt"])

exp_data["children_per_room"] = (exp_data["srch_children_cnt"]/exp_data["srch_rm_cnt"])
exp_data["person_per_room"] = ((exp_data["srch_children_cnt"]+exp_data["srch_adults_cnt"])/exp_data["srch_rm_cnt"])
exp_data["event_season_bin"] = exp_data.apply(lambda row: create_month_bins(row["event_month"]), axis=1)

exp_data["adult_per_room"] =  exp_data["adult_per_room"].round(2)
exp_data["children_per_room"] = exp_data["children_per_room"].round(2)
exp_data["person_per_room"] = exp_data["person_per_room"].round(2)
exp_data["event_season_bin"] = exp_data["event_season_bin"].round(2)

exp_data["event_is_winter"] = exp_data.apply(lambda row: check_season(row["event_season_bin"],"winter"), axis=1)
exp_data["event_is_spring"] = exp_data.apply(lambda row: check_season(row["event_season_bin"],"spring"), axis=1)
exp_data["event_is_summer"] = exp_data.apply(lambda row: check_season(row["event_season_bin"],"summer"), axis=1)
exp_data["event_is_fall"] = exp_data.apply(lambda row: check_season(row["event_season_bin"],"fall"), axis=1)

exp_data["event_weekday"] = exp_data["event_date"].apply(lambda x: x.weekday())
exp_data["weekend_event"] = exp_data.apply(lambda row: weekend_check(row["event_weekday"]), axis=1)

exp_data["evet_is_late_night"] = exp_data.apply(lambda row: check_time(row["event_time"],"late_night"), axis=1)
exp_data["evet_is_late_early_morning"] = exp_data.apply(lambda row: check_time(row["event_time"],"early_morning"), axis=1)
exp_data["evet_is_morning"] = exp_data.apply(lambda row: check_time(row["event_time"],"morning"), axis=1)
exp_data["evet_is_mid_day"] = exp_data.apply(lambda row: check_time(row["event_time"],"mid_day"), axis=1)
exp_data["evet_is_afternoon"] = exp_data.apply(lambda row: check_time(row["event_time"],"afternoon"), axis=1)
exp_data["evet_is_evening"] = exp_data.apply(lambda row: check_time(row["event_time"],"evening"), axis=1)
exp_data["evet_is_night"] = exp_data.apply(lambda row: check_time(row["event_time"],"night"), axis=1)

pd.to_numeric(exp_data["room_night"])




print exp_data.info()
print exp_data.head()


# Create tabel variable to pass train_test_split
exp_data_labels = exp_data["is_booking"]

exp_data = exp_data.dropna(subset=["adult_per_room"], how="all")
exp_data = exp_data.dropna(subset=["children_per_room"], how="all")
exp_data = exp_data.dropna(subset=["person_per_room"], how="all")

# Remove column stripped columns from data frame
df_drop_columns = ["date_time","orig_destination_distance","srch_ci","srch_co","event_date","event_time",]

for i in df_drop_columns:
    data_train = exp_data.drop(df_drop_columns, axis=1)


print data_train

data_train.to_csv("clean_sample.csv", index=False, encoding='utf-8', index_label=True,)

print "train data", data_train.shape