# %%
import pandas as pd

# %%
train_data = pd.read_parquet("data/train_data.parquet")
print(train_data.isna().sum() / len(train_data))

# %%
submission_data = pd.read_parquet("data/submission_data.parquet")
print(submission_data.isna().sum() / len(train_data))
# %%
total_data = pd.concat([train_data, submission_data])
# %%
split_feats = ["brand", "country"]
total_data["code"] = [
    "_".join([str(brand), str(country)])
    for brand, country in zip(total_data["brand"], total_data["country"])
]
# %%
# codes_test = total_data.query('date>="01-01-2022"')["code"].unique().tolist()
# total_data = total_data.query("code in @codes_test")
# %%
from tqdm import tqdm

tqdm.pandas()


# %%
def add_basic_valid_lag_features_neighbour_rolling(
    df, n_lags_week, n_lags_day, n_lags_month
):
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for j in range(1, 4):
        for i in range(-n_lags_day, n_lags_day + 1):
            df[f"lag_phase_yr{j}_{i}_days_median"] = (
                df.phase.shift(365 * j + i, freq="D").rolling(3, center=True).median()
            )

        for i in range(-n_lags_month, n_lags_month + 1):
            df[f"lag_phase_yr{j}_{i}_month_median"] = (
                df.phase.shift(365 * j + 30 * i, freq="D")
                .rolling(5, center=True)
                .median()
            )

        for i in range(-n_lags_week, n_lags_week + 1):
            df[f"lag_phase_yr{j}_{i}_yr_exact"] = (
                df.phase.shift(365 * j + 7 * i, freq="D")
                .rolling(3, center=True)
                .median()
            )

    return df.reset_index()


# %%
total_data_with_lags = (
    total_data.groupby("code", as_index=False).progress_apply(
        lambda x: add_basic_valid_lag_features_neighbour_rolling(x, 5, 10, 3)
    )
).reset_index()
# %%
total_data_with_lags.query('date<"01-01-2022"').to_parquet(
    "data/205_feateng_train_data.parquet"
)
total_data_with_lags.query('date>="01-01-2022"').to_parquet(
    "data/205_feateng_test_data.parquet"
)


# %%
total_data_with_lags.query('date<"01-01-2022"').corr()["phase"].sort_values()
# %%
