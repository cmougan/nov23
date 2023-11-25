# %% 
import pandas as pd

from src.utils.rolling import rolling_pd

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 300)
# %%

train_data = pd.read_parquet("data/train_data.parquet")
submission_data = pd.read_parquet("data/submission_data.parquet")
# %%
all_data = pd.concat([train_data, submission_data]).sort_values(["brand", "country", "date"])
# %%
sample_df = all_data.query("brand == 'ZVLFE' and country == 'Zamunda'")
sample_df

# %%
sample_df["rolling"] = rolling_pd(sample_df, groupby_cols=["brand", "country", "month", "wd"], column="phase", function="mean", rolling_periods=2, shift_periods=1)
sample_df["rolling"]
# %%
sample_df.head(300)
# %%
sample_df.query("month == 1 and wd == 3")
# %%
all_data["rolling"] = rolling_pd(all_data, groupby_cols=["brand", "country", "month", "wd"], column="phase", function="mean", rolling_periods=2, shift_periods=1)
# %%
all_data.query("month == 1 and wd == 3 and brand == 'ZVLFE' and country == 'Zamunda'")
# %%
