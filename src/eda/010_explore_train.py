# %% 
import pandas as pd

pd.set_option("display.max_columns", 500)
# %%

train_data = pd.read_parquet("data/train_data.parquet")
submission_data = pd.read_parquet("data/submission_data.parquet")
# %%
train_data
# %%
train_data.dayweek.value_counts()
# %%
submission_data.dayweek.value_counts()
# %%
submission_data
# %%
train_data.country.nunique()
# %%
train_data.brand.nunique()
# %%
train_data["code"] = train_data["brand"].astype(str) + "_" + train_data["country"].astype(str)
# %%
train_data.code.nunique()
# %%
train_data.groupby("dayweek").agg({"phase": "mean", "code": "size"})
# %%
def groupby_agg(df, groupby, agg):
    return (
        df.assign(**{groupby: lambda x: x[groupby].astype(str).fillna("unknown")})
        .groupby(groupby)
        .agg({"phase": agg, "code": "size"})
    )
# %%
groupby_agg(train_data, "ther_area", "mean")
# %%
groupby_agg(train_data, "main_channel", "mean")
# %%
submission_data
# %%
