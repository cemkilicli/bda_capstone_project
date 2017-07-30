import pandas as pd

exp_data_train = pd.read_csv("train.csv", delimiter=',')


exp_data_train = exp_data_train[exp_data_train.is_booking != 0]

print exp_data_train["is_booking"]

exp_data_train.to_csv("sample_booked.csv", index=False, encoding='utf-8', index_label=True)


