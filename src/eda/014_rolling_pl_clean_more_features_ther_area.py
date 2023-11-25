# %% 
import numpy as np
import polars as pl
from tqdm import tqdm

# %%
train_data = pl.read_parquet("data/train_data.parquet")
submission_data = pl.read_parquet("data/submission_data.parquet")

# %%
# %% create empty columns for monthly and phase
submission_data = submission_data.with_columns(pl.lit(np.nan).alias("monthly"))
submission_data = submission_data.with_columns(pl.lit(np.nan).alias("phase"))
# %%
submission_data
# %% sort columns to match train_data
submission_data = submission_data[train_data.columns]

all_data = (
    pl.concat([train_data, submission_data])
    .sort(["brand", "country", "date"])
    .with_columns(
        pl.col('date').apply(lambda x: x.strftime('%m%d')).alias('formatted_date'),
        # convert main channel and ther area to str and fill nan with unknown
        pl.col('main_channel').cast(pl.Categorical).cast(pl.Utf8).fill_null("unknown").alias('main_channel'),
        pl.col('ther_area').cast(pl.Categorical).cast(pl.Utf8).fill_null("unknown").alias('ther_area'),
    )
)

# %%
all_data

# %% 
def run_aggregations(data, group_columns, agg_column, window_period, window_closed="left", window_offset=None, fn="mean"):
    """
    Run rolling aggregations on the given data.

    Parameters:
    - data: polars.DataFrame, the input data frame.
    - group_columns: list of str, columns to group by.
    - agg_column: str, the column to aggregate.
    - window_period: str, the window period for rolling.
    - window_closed: str, the closed parameter for rolling window (default is "left").
    - window_offset: str, the offset parameter for rolling window (default is None).

    Returns:
    - polars.DataFrame, the result of the rolling aggregation.
    """
    if window_offset is None:
        aggs = data.rolling(
            index_column="date", by=group_columns, period=window_period, closed=window_closed
        )
    else:
        aggs = data.rolling(
            index_column="date", by=group_columns, period=window_period, closed=window_closed, offset=window_offset
        )

    if fn == "mean":
        aggs = aggs.agg(pl.col(agg_column).mean().alias(f"{agg_column}_mean"))
    elif fn == "median":
        aggs = aggs.agg(pl.col(agg_column).median().alias(f"{agg_column}_median"))

    result_aggs = data.join(aggs, on=["date"] + group_columns, how="left")

    return result_aggs[f"{agg_column}_{fn}"]


# %%
for n_years in tqdm(range(1, 6)):

    for fn in tqdm(["mean", "median"]):
        
        feature_mchannel = run_aggregations(
            all_data.sort(["main_channel", "date"]),
            group_columns=["main_channel", "month", "wd"],
            agg_column="phase",
            window_period=f"{n_years}y",
            window_closed="left",
            window_offset=None,
            fn=fn
        )

        feature_ther_area = run_aggregations(
            all_data.sort(["ther_area", "date"]),
            group_columns=["ther_area", "month", "wd"],
            agg_column="phase",
            window_period=f"{n_years}y",
            window_closed="left",
            window_offset=None,
            fn=fn
        )

        feature_b_dayweek = run_aggregations(
            all_data.sort(["brand", "date"]),
            group_columns=["brand", "dayweek"],
            agg_column="phase",
            window_period=f"{n_years}y",
            window_closed="left",
            window_offset=f"{-n_years - 1}y",
            fn=fn
        )

        feature_b_wd = run_aggregations(
            all_data.sort(["brand", "date"]),
            group_columns=["brand", "wd"],
            agg_column="phase",
            window_period=f"{n_years}y",
            window_closed="left",
            window_offset=None,
            fn=fn
        )

        feature_c_dayweek = run_aggregations(
            all_data.sort(["country", "date"]),
            group_columns=["country", "dayweek"],
            agg_column="phase",
            window_period=f"{n_years}y",
            window_closed="left",
            window_offset=f"{-n_years - 1}y",
            fn=fn
        )

        feature_c_wd = run_aggregations(
            all_data.sort(["country", "date"]),
            group_columns=["country", "wd"],
            agg_column="phase",
            window_period=f"{n_years}y",
            window_closed="left",
            window_offset=None,
            fn=fn
        )

        all_data = all_data.with_columns(
            feature_b_dayweek.alias(f"b_phase_{fn}_{n_years}y_dayweek"),
            feature_b_wd.alias(f"b_phase_{fn}_{n_years}y_wd"),
            feature_c_dayweek.alias(f"c_phase_{fn}_{n_years}y_dayweek"),
            feature_c_wd.alias(f"c_phase_{fn}_{n_years}y_wd"),
            feature_mchannel.alias(f"mchannel_phase_{fn}_{n_years}y"),
            feature_ther_area.alias(f"ther_area_phase_{fn}_{n_years}y"),
        )

        
# %%
new_features = set(all_data.columns).difference(set(submission_data.columns)).difference(set(["formatted_date"]))
print(new_features)

# %%
all_data[["date", "country", "brand"] + list(new_features)].write_parquet("data/rolling_features_less_aggs_ther_area.parquet")

# %%
all_data[["date", "country", "brand"] + list(new_features)]

# %%
# %%
