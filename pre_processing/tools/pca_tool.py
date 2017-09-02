
import pandas as pd
# Decopose destination data
from sklearn.decomposition import PCA

def pca_data(data):
    pca_label_list = ["srch_ci_year", "srch_ci_month", "srch_ci_day", "srch_ci_season_bin",
                      "srch_co_year", "srch_co_month", "srch_co_day", "srch_co_season_bin",
                      "srch_ci_is_winter", "srch_ci_is_spring", "srch_ci_is_summer", "srch_ci_is_fall",
                      "srch_co_is_winter", "srch_co_is_spring", "srch_co_is_summer", "srch_co_is_fall",
                      "event_season_bin", "event_is_winter", "event_is_spring", "event_is_summer",
                      "event_is_fall", "event_weekday", "weekend_event", "evet_is_late_night",
                      "evet_is_early_morning", "evet_is_morning", "evet_is_mid_day", "evet_is_afternoon",
                      "evet_is_evening"]

    exp_data_events = data[pca_label_list]

    ### PCA train data
    pca = PCA(n_components=4)
    exp_data_events_pca = pca.fit_transform(exp_data_events[pca_label_list])

    exp_data_events_pca = pd.DataFrame(exp_data_events_pca)
    exp_data_events_pca["srch_destination_id"] = data["srch_destination_id"]


    exp_data_train = pd.merge(data, exp_data_events_pca, on="srch_destination_id", how='left')
    print "train data info", exp_data_train.info()

    data.to_csv("../exp_data/processed/clean_sample_pca_train111.csv", index=False, encoding='utf-8',
                           index_label=True, )
