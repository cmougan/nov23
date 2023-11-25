# %% 
import numpy as np
import polars as pl

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
        pl.col('date').apply(lambda x: x.strftime('%m%d')).alias('formatted_date')
    )
)

# %% 
def run_aggregations(data, group_columns, agg_column, window_period, window_closed="left", window_offset=None):
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
        ).agg(pl.col(agg_column).mean().alias(f"{agg_column}_mean"))
    else:
        aggs = data.rolling(
            index_column="date", by=group_columns, period=window_period, closed=window_closed, offset=window_offset
        ).agg(pl.col(agg_column).mean().alias(f"{agg_column}_mean"))

    result_aggs = data.join(aggs, on=["date"] + group_columns, how="left")

    return result_aggs[f"{agg_column}_mean"]


# %%
for n_years in range(1, 6):
    feature_wd = run_aggregations(
        all_data,
        group_columns=["brand", "country", "month", "wd"],
        agg_column="phase",
        window_period=f"{n_years}y",
        window_closed="left",
        window_offset=None
    )

    feature_formatted_date = run_aggregations(
        all_data,
        group_columns=["brand", "country", "month", "formatted_date"],
        agg_column="phase",
        window_period=f"{n_years}y",
        window_closed="left",
        window_offset=None
    )

    feature_dayweek = run_aggregations(
        all_data,
        group_columns=["brand", "country", "month", "dayweek"],
        agg_column="phase",
        window_period=f"{n_years}y",
        window_closed="left",
        window_offset=f"{-n_years - 1}y",
    )

    all_data = all_data.with_columns(
        feature_wd.alias(f"phase_mean_{n_years}y_wd"),
        feature_formatted_date.alias(f"phase_mean_{n_years}y_formatted_date"),
        feature_dayweek.alias(f"phase_mean_{n_years}y_dayweek"),
    )

        
# %%
all_data.describe()
# all_data.corr()
# %%
aggs = all_data.rolling(
    index_column="date", by=["brand", "country", "month", "wd"], period="3y", closed="left"
).agg(pl.col("phase").mean())
# %%
all_data.join(aggs, on=["date"] + ["brand", "country", "month", "wd"], how="left")
# %%
new_features = set(all_data.columns).difference(set(submission_data.columns))
# %%
all_data[["date", "country", "brand"] + list(new_features)].write_csv("data/rolling_features.csv")

# %%
all_data[["date", "country", "brand"] + list(new_features)].write_parquet("data/rolling_features.parquet")

# %%
all_data[["date", "country", "brand"] + list(new_features)].tail(100).write_csv("data/rolling_features_100.csv")
# %%
