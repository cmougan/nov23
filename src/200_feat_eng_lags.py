# %%
import pandas as pd
from utils.preprocessing import add_basic_lag_features

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
from tqdm import tqdm

tqdm.pandas()
# %%
total_data_with_lags = (
    total_data.groupby("code", as_index=False).progress_apply(
        lambda x: add_basic_lag_features(x, 14, 3, 1)
    )
).reset_index()
# %%
lags_features = [k for k in total_data_with_lags.keys() if "lag_" in k]
# fillna?
# %%
total_data_with_lags.query('date<"01-01-2022"').to_parquet(
    "data/200_feateng_train_data.parquet"
)
total_data_with_lags.query('date>="01-01-2022"').to_parquet(
    "data/200_feateng_test_data.parquet"
)


# %%
