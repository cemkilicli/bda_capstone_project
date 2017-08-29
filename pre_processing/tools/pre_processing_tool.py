from datetime import datetime
import pandas as pd
import numpy as np


def strip(exp_date, type):
        stripped_date = datetime.strptime(exp_date, "%Y-%m-%d %H:%M:%S")
        if type == "date":
            return stripped_date.date()
        elif type == "time":
            return stripped_date.time()
        elif type == "year":
            return stripped_date.year
        elif type == "month":
            return stripped_date.month
        elif type == "day":
            return stripped_date.day

def strip_srch(exp_date, type):
    stripped_date = datetime.strptime(exp_date, "%Y-%m-%d")
    if type == "date":
        return stripped_date.year
    elif type == "time":
        return stripped_date.time()
    elif type == "year":
        return stripped_date.year
    elif type == "month":
        return stripped_date.month
    elif type == "day":
        return stripped_date.day


def weight_event(is_booking):

    if is_booking == 1:
        weight = 1
    else:
        weight = 0.25

    return weight




def date_subtract(date1,date2):
    stripped_date1 = datetime.strptime(date1, "%Y-%m-%d")
    stripped_date2 = datetime.strptime(date2, "%Y-%m-%d")
    date_difference = stripped_date2 - stripped_date1
    return date_difference


def create_month_bins(month):
    if month <=3:
        return  1
    elif month > 3 and month <=6:
        return  2
    elif month >6 and month <=9:
        return 3
    elif month >9 and month <=12:
        return 4

def check_season(season_bin,season):

    if season == "winter":
        if season_bin == 1:
            return 1
        else:
            return 0
    elif season == "spring":
        if season_bin == 2:
            return 1
        else:
            return 0
    elif season == "summer":
        if season_bin == 3:
            return 1
        else:
            return 0
    elif season == "fall":
        if season_bin == 4:
            return 1
        else:
            return 0

def weekend_check(day):
    if day == 6 or day == 5:
        return 1
    else:
        return 0

def check_time(time, day_time):
    import datetime

    time_1 = datetime.time(0, 0, 0)
    time_2 = datetime.time(6, 0, 0)
    time_3 = datetime.time(8, 0, 0)
    time_4 = datetime.time(10, 0, 0)
    time_5 = datetime.time(14, 0, 0)
    time_6 = datetime.time(16, 0, 0)
    time_7 = datetime.time(20, 0, 0)

    if day_time == "late_night":
        if time >= time_1 and time < time_2:
            return 1
        else:
            return 0

    if day_time == "early_morning":
        if time >= time_2 and time < time_3:
            return 1
        else:
            return 0

    if day_time == "morning":
        if time >= time_3 and time < time_4:
            return 1
        else:
            return 0

    if day_time == "mid_day":
        if time >= time_4 and time < time_5:
            return 1
        else:
            return 0

    if day_time == "afternoon":
        if time >= time_5 and time < time_6:
            return 1
        else:
            return 0

    if day_time == "evening":
        if time >= time_6 and time < time_7:
            return 1
        else:
            return 0

    if day_time == "night":
        if time >= time_7:
            return 1
        else:
            return 0


def process_train(exp_data_train, exp_data_destinations):
    exp_data_train["srch_ci"].replace("", np.nan, inplace=True)
    exp_data_train["srch_co"].replace("", np.nan, inplace=True)
    exp_data_train = exp_data_train.dropna(subset=["srch_ci"], how="all")
    exp_data_train = exp_data_train.dropna(subset=["srch_co"], how="all")

    # Creating vatiables
    exp_data_train["event_date"] = exp_data_train.apply(lambda row: strip(row["date_time"], "date"), axis=1)
    exp_data_train["event_time"] = exp_data_train.apply(lambda row: strip(row["date_time"], "time"), axis=1)
    exp_data_train["event_month"] = exp_data_train.apply(lambda row: strip(row["date_time"], "month"), axis=1)
    exp_data_train["event_day"] = exp_data_train.apply(lambda row: strip(row["date_time"], "day"), axis=1)
    exp_data_train["event_year"] = exp_data_train.apply(lambda row: strip(row["date_time"], "year"), axis=1)

    exp_data_train["srch_ci_year"] = exp_data_train.apply(lambda row: strip_srch(row["srch_ci"], "year"), axis=1)
    exp_data_train["srch_ci_month"] = exp_data_train.apply(lambda row: strip_srch(row["srch_ci"], "month"), axis=1)
    exp_data_train["srch_ci_day"] = exp_data_train.apply(lambda row: strip_srch(row["srch_ci"], "day"), axis=1)
    exp_data_train["srch_ci_season_bin"] = exp_data_train.apply(lambda row: create_month_bins(row["srch_ci_month"]),
                                                                axis=1)

    exp_data_train["srch_co_year"] = exp_data_train.apply(lambda row: strip_srch(row["srch_co"], "year"), axis=1)
    exp_data_train["srch_co_month"] = exp_data_train.apply(lambda row: strip_srch(row["srch_co"], "month"), axis=1)
    exp_data_train["srch_co_day"] = exp_data_train.apply(lambda row: strip_srch(row["srch_co"], "day"), axis=1)
    exp_data_train["srch_co_season_bin"] = exp_data_train.apply(lambda row: create_month_bins(row["srch_co_month"]),
                                                                axis=1)

    exp_data_train["srch_ci_is_winter"] = exp_data_train.apply(
        lambda row: check_season(row["srch_ci_season_bin"], "winter"), axis=1)
    exp_data_train["srch_ci_is_spring"] = exp_data_train.apply(
        lambda row: check_season(row["srch_ci_season_bin"], "spring"), axis=1)
    exp_data_train["srch_ci_is_summer"] = exp_data_train.apply(
        lambda row: check_season(row["srch_ci_season_bin"], "summer"), axis=1)
    exp_data_train["srch_ci_is_fall"] = exp_data_train.apply(
        lambda row: check_season(row["srch_ci_season_bin"], "fall"), axis=1)

    exp_data_train["srch_co_is_winter"] = exp_data_train.apply(
        lambda row: check_season(row["srch_co_season_bin"], "winter"), axis=1)
    exp_data_train["srch_co_is_spring"] = exp_data_train.apply(
        lambda row: check_season(row["srch_co_season_bin"], "spring"), axis=1)
    exp_data_train["srch_co_is_summer"] = exp_data_train.apply(
        lambda row: check_season(row["srch_co_season_bin"], "summer"), axis=1)
    exp_data_train["srch_co_is_fall"] = exp_data_train.apply(
        lambda row: check_season(row["srch_co_season_bin"], "fall"), axis=1)

    exp_data_train["night_of_stay"] = exp_data_train.apply(lambda row: date_subtract(row["srch_ci"], row["srch_co"]),
                                                           axis=1)
    exp_data_train["night_of_stay"] = exp_data_train["night_of_stay"] / np.timedelta64(1, 'D')
    # exp_data_train["booking_window"] = exp_data_train["event_date"]- exp_data_train["srch_ci"]

    exp_data_train["adult_per_room"] = (exp_data_train["srch_adults_cnt"] / exp_data_train["srch_rm_cnt"])

    exp_data_train["children_per_room"] = (exp_data_train["srch_children_cnt"] / exp_data_train["srch_rm_cnt"])
    exp_data_train["person_per_room"] = (
    (exp_data_train["srch_children_cnt"] + exp_data_train["srch_adults_cnt"]) / exp_data_train["srch_rm_cnt"])
    exp_data_train["event_season_bin"] = exp_data_train.apply(lambda row: create_month_bins(row["event_month"]), axis=1)

    exp_data_train["adult_per_room"] = exp_data_train["adult_per_room"].round(2)
    exp_data_train["children_per_room"] = exp_data_train["children_per_room"].round(2)
    exp_data_train["person_per_room"] = exp_data_train["person_per_room"].round(2)
    exp_data_train["event_season_bin"] = exp_data_train["event_season_bin"].round(2)

    exp_data_train["event_is_winter"] = exp_data_train.apply(
        lambda row: check_season(row["event_season_bin"], "winter"), axis=1)
    exp_data_train["event_is_spring"] = exp_data_train.apply(
        lambda row: check_season(row["event_season_bin"], "spring"), axis=1)
    exp_data_train["event_is_summer"] = exp_data_train.apply(
        lambda row: check_season(row["event_season_bin"], "summer"), axis=1)
    exp_data_train["event_is_fall"] = exp_data_train.apply(lambda row: check_season(row["event_season_bin"], "fall"),
                                                           axis=1)

    #### This need to be check before final submittion ####
    exp_data_train["event_weekday"] = exp_data_train["event_date"].apply(lambda x: x.weekday())
    exp_data_train["weekend_event"] = exp_data_train.apply(lambda row: weekend_check(row["event_weekday"]), axis=1)

    exp_data_train["evet_is_late_night"] = exp_data_train.apply(lambda row: check_time(row["event_time"], "late_night"),
                                                                axis=1)
    exp_data_train["evet_is_early_morning"] = exp_data_train.apply(
        lambda row: check_time(row["event_time"], "early_morning"), axis=1)
    exp_data_train["evet_is_morning"] = exp_data_train.apply(lambda row: check_time(row["event_time"], "morning"),
                                                             axis=1)
    exp_data_train["evet_is_mid_day"] = exp_data_train.apply(lambda row: check_time(row["event_time"], "mid_day"),
                                                             axis=1)
    exp_data_train["evet_is_afternoon"] = exp_data_train.apply(lambda row: check_time(row["event_time"], "afternoon"),
                                                               axis=1)
    exp_data_train["evet_is_evening"] = exp_data_train.apply(lambda row: check_time(row["event_time"], "evening"),
                                                             axis=1)
    exp_data_train["evet_is_night"] = exp_data_train.apply(lambda row: check_time(row["event_time"], "night"), axis=1)

    pd.to_numeric(exp_data_train["night_of_stay"])

    exp_data_train = exp_data_train.dropna(subset=["adult_per_room"], how="all")
    exp_data_train = exp_data_train.dropna(subset=["children_per_room"], how="all")
    exp_data_train = exp_data_train.dropna(subset=["person_per_room"], how="all")

    # exp_data_train['orig_destination_distance'] = exp_data_train['orig_destination_distance'].fillna(-1)

    exp_data_train["event_weight"] = exp_data_train.apply(lambda row: weight_event(row["is_booking"]), axis=1)

    # Decopose destination data
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    exp_data_destinations_pca = pca.fit_transform(exp_data_destinations[["d{0}".format(i + 1) for i in range(149)]])
    exp_data_destinations_pca = pd.DataFrame(exp_data_destinations_pca)
    exp_data_destinations_pca["srch_destination_id"] = exp_data_destinations["srch_destination_id"]

    # Join Destinations and train set
    # exp_data_train = exp_data_train.join(exp_data_destinations_pca, on="srch_destination_id", how='left', rsuffix="dest")
    exp_data_train = pd.merge(exp_data_train, exp_data_destinations_pca, on="srch_destination_id", how='left')

    exp_data_train = exp_data_train.rename(columns={0: "dest_0"})
    exp_data_train = exp_data_train.rename(columns={1: "dest_1"})
    exp_data_train = exp_data_train.rename(columns={2: "dest_2"})
    exp_data_train["dest_0"].replace("", np.nan, inplace=True)
    exp_data_train["dest_1"].replace("", np.nan, inplace=True)
    exp_data_train["dest_2"].replace("", np.nan, inplace=True)
    exp_data_train = exp_data_train.dropna(subset=["dest_0"], how="all")
    exp_data_train = exp_data_train.dropna(subset=["dest_1"], how="all")
    exp_data_train = exp_data_train.dropna(subset=["dest_2"], how="all")


    exp_data_train.to_csv("../exp_data/processed/clean_sample_train.csv", index=False, encoding='utf-8',
                          index_label=True, )

    return exp_data_train




def process_test(exp_data_train, exp_data_destinations):

    exp_data_train = exp_data_train[exp_data_train.is_booking == 1]

    exp_data_train["srch_ci"].replace("", np.nan, inplace=True)
    exp_data_train["srch_co"].replace("", np.nan, inplace=True)
    exp_data_train = exp_data_train.dropna(subset=["srch_ci"], how="all")
    exp_data_train = exp_data_train.dropna(subset=["srch_co"], how="all")

    # Creating vatiables
    exp_data_train["event_date"] = exp_data_train.apply(lambda row: strip(row["date_time"], "date"), axis=1)
    exp_data_train["event_time"] = exp_data_train.apply(lambda row: strip(row["date_time"], "time"), axis=1)
    exp_data_train["event_month"] = exp_data_train.apply(lambda row: strip(row["date_time"], "month"), axis=1)
    exp_data_train["event_day"] = exp_data_train.apply(lambda row: strip(row["date_time"], "day"), axis=1)
    exp_data_train["event_year"] = exp_data_train.apply(lambda row: strip(row["date_time"], "year"), axis=1)

    exp_data_train["srch_ci_year"] = exp_data_train.apply(lambda row: strip_srch(row["srch_ci"], "year"), axis=1)
    exp_data_train["srch_ci_month"] = exp_data_train.apply(lambda row: strip_srch(row["srch_ci"], "month"), axis=1)
    exp_data_train["srch_ci_day"] = exp_data_train.apply(lambda row: strip_srch(row["srch_ci"], "day"), axis=1)
    exp_data_train["srch_ci_season_bin"] = exp_data_train.apply(lambda row: create_month_bins(row["srch_ci_month"]),
                                                                axis=1)

    exp_data_train["srch_co_year"] = exp_data_train.apply(lambda row: strip_srch(row["srch_co"], "year"), axis=1)
    exp_data_train["srch_co_month"] = exp_data_train.apply(lambda row: strip_srch(row["srch_co"], "month"), axis=1)
    exp_data_train["srch_co_day"] = exp_data_train.apply(lambda row: strip_srch(row["srch_co"], "day"), axis=1)
    exp_data_train["srch_co_season_bin"] = exp_data_train.apply(lambda row: create_month_bins(row["srch_co_month"]),
                                                                axis=1)

    exp_data_train["srch_ci_is_winter"] = exp_data_train.apply(
        lambda row: check_season(row["srch_ci_season_bin"], "winter"), axis=1)
    exp_data_train["srch_ci_is_spring"] = exp_data_train.apply(
        lambda row: check_season(row["srch_ci_season_bin"], "spring"), axis=1)
    exp_data_train["srch_ci_is_summer"] = exp_data_train.apply(
        lambda row: check_season(row["srch_ci_season_bin"], "summer"), axis=1)
    exp_data_train["srch_ci_is_fall"] = exp_data_train.apply(
        lambda row: check_season(row["srch_ci_season_bin"], "fall"), axis=1)

    exp_data_train["srch_co_is_winter"] = exp_data_train.apply(
        lambda row: check_season(row["srch_co_season_bin"], "winter"), axis=1)
    exp_data_train["srch_co_is_spring"] = exp_data_train.apply(
        lambda row: check_season(row["srch_co_season_bin"], "spring"), axis=1)
    exp_data_train["srch_co_is_summer"] = exp_data_train.apply(
        lambda row: check_season(row["srch_co_season_bin"], "summer"), axis=1)
    exp_data_train["srch_co_is_fall"] = exp_data_train.apply(
        lambda row: check_season(row["srch_co_season_bin"], "fall"), axis=1)

    exp_data_train["night_of_stay"] = exp_data_train.apply(lambda row: date_subtract(row["srch_ci"], row["srch_co"]),
                                                           axis=1)
    exp_data_train["night_of_stay"] = exp_data_train["night_of_stay"] / np.timedelta64(1, 'D')
    # exp_data_train["booking_window"] = exp_data_train["event_date"]- exp_data_train["srch_ci"]

    exp_data_train["adult_per_room"] = (exp_data_train["srch_adults_cnt"] / exp_data_train["srch_rm_cnt"])

    exp_data_train["children_per_room"] = (exp_data_train["srch_children_cnt"] / exp_data_train["srch_rm_cnt"])
    exp_data_train["person_per_room"] = (
    (exp_data_train["srch_children_cnt"] + exp_data_train["srch_adults_cnt"]) / exp_data_train["srch_rm_cnt"])
    exp_data_train["event_season_bin"] = exp_data_train.apply(lambda row: create_month_bins(row["event_month"]), axis=1)

    exp_data_train["adult_per_room"] = exp_data_train["adult_per_room"].round(2)
    exp_data_train["children_per_room"] = exp_data_train["children_per_room"].round(2)
    exp_data_train["person_per_room"] = exp_data_train["person_per_room"].round(2)
    exp_data_train["event_season_bin"] = exp_data_train["event_season_bin"].round(2)

    exp_data_train["event_is_winter"] = exp_data_train.apply(
        lambda row: check_season(row["event_season_bin"], "winter"), axis=1)
    exp_data_train["event_is_spring"] = exp_data_train.apply(
        lambda row: check_season(row["event_season_bin"], "spring"), axis=1)
    exp_data_train["event_is_summer"] = exp_data_train.apply(
        lambda row: check_season(row["event_season_bin"], "summer"), axis=1)
    exp_data_train["event_is_fall"] = exp_data_train.apply(lambda row: check_season(row["event_season_bin"], "fall"),
                                                           axis=1)

    #### This need to be check before final submittion ####
    exp_data_train["event_weekday"] = exp_data_train["event_date"].apply(lambda x: x.weekday())
    exp_data_train["weekend_event"] = exp_data_train.apply(lambda row: weekend_check(row["event_weekday"]), axis=1)

    exp_data_train["evet_is_late_night"] = exp_data_train.apply(lambda row: check_time(row["event_time"], "late_night"),
                                                                axis=1)
    exp_data_train["evet_is_early_morning"] = exp_data_train.apply(
        lambda row: check_time(row["event_time"], "early_morning"), axis=1)
    exp_data_train["evet_is_morning"] = exp_data_train.apply(lambda row: check_time(row["event_time"], "morning"),
                                                             axis=1)
    exp_data_train["evet_is_mid_day"] = exp_data_train.apply(lambda row: check_time(row["event_time"], "mid_day"),
                                                             axis=1)
    exp_data_train["evet_is_afternoon"] = exp_data_train.apply(lambda row: check_time(row["event_time"], "afternoon"),
                                                               axis=1)
    exp_data_train["evet_is_evening"] = exp_data_train.apply(lambda row: check_time(row["event_time"], "evening"),
                                                             axis=1)
    exp_data_train["evet_is_night"] = exp_data_train.apply(lambda row: check_time(row["event_time"], "night"), axis=1)

    pd.to_numeric(exp_data_train["night_of_stay"])

    exp_data_train = exp_data_train.dropna(subset=["adult_per_room"], how="all")
    exp_data_train = exp_data_train.dropna(subset=["children_per_room"], how="all")
    exp_data_train = exp_data_train.dropna(subset=["person_per_room"], how="all")

    # exp_data_train['orig_destination_distance'] = exp_data_train['orig_destination_distance'].fillna(-1)

    exp_data_train["event_weight"] = exp_data_train.apply(lambda row: weight_event(row["is_booking"]), axis=1)

    # Decopose destination data
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    exp_data_destinations_pca = pca.fit_transform(exp_data_destinations[["d{0}".format(i + 1) for i in range(149)]])
    exp_data_destinations_pca = pd.DataFrame(exp_data_destinations_pca)
    exp_data_destinations_pca["srch_destination_id"] = exp_data_destinations["srch_destination_id"]

    print exp_data_destinations_pca

    # Join Destinations and train set
    # exp_data_train = exp_data_train.join(exp_data_destinations_pca, on="srch_destination_id", how='left', rsuffix="dest")
    exp_data_test = pd.merge(exp_data_train, exp_data_destinations_pca, on="srch_destination_id", how='left')

    exp_data_test = exp_data_test.rename(columns={0: "dest_0"})
    exp_data_test = exp_data_test.rename(columns={1: "dest_1"})
    exp_data_test = exp_data_test.rename(columns={2: "dest_2"})

    exp_data_test["dest_0"].replace("", np.nan, inplace=True)
    exp_data_test["dest_1"].replace("", np.nan, inplace=True)
    exp_data_test["dest_2"].replace("", np.nan, inplace=True)
    exp_data_test = exp_data_test.dropna(subset=["dest_0"], how="all")
    exp_data_test = exp_data_test.dropna(subset=["dest_1"], how="all")
    exp_data_test = exp_data_test.dropna(subset=["dest_2"], how="all")


    exp_data_test.to_csv("../exp_data/processed/clean_sample_test.csv", index=False, encoding='utf-8',
                          index_label=True, )

    return exp_data_test