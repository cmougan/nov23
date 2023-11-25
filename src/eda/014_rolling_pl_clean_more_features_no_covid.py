# %% 
import datetime

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
    .filter(
        # remove covid months
        (pl.col("date") < datetime.datetime(2020, 2, 1)) | (pl.col("date") >= datetime.datetime(2020, 5, 1))
    )
)

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
for n_years in range(1, 6):

    for fn in ["mean", "median"]:
        feature_bcm_wd = run_aggregations(
            all_data,
            group_columns=["brand", "country", "month", "wd"],
            agg_column="phase",
            window_period=f"{n_years}y",
            window_closed="left",
            window_offset=None,
            fn=fn
        )

        feature_bcm_formatted_date = run_aggregations(
            all_data,
            group_columns=["brand", "country", "month", "formatted_date"],
            agg_column="phase",
            window_period=f"{n_years}y",
            window_closed="left",
            window_offset=None,
            fn=fn
        )

        feature_bc_dayweek = run_aggregations(
            all_data,
            group_columns=["brand", "country", "dayweek"],
            agg_column="phase",
            window_period=f"{n_years}y",
            window_closed="left",
            window_offset=f"{-n_years - 1}y",
            fn=fn
        )

        feature_bc_wd = run_aggregations(
            all_data,
            group_columns=["brand", "country", "wd"],
            agg_column="phase",
            window_period=f"{n_years}y",
            window_closed="left",
            window_offset=None,
            fn=fn
        )


        # feature_b_dayweek = run_aggregations(
        #     all_data,
        #     group_columns=["brand", "dayweek"],
        #     agg_column="phase",
        #     window_period=f"{n_years}y",
        #     window_closed="left",
        #     window_offset=f"{-n_years - 1}y",
        #     fn=fn
        # )

        # feature_b_wd = run_aggregations(
        #     all_data,
        #     group_columns=["brand", "wd"],
        #     agg_column="phase",
        #     window_period=f"{n_years}y",
        #     window_closed="left",
        #     window_offset=None,
        #     fn=fn
        # )

        # feature_c_dayweek = run_aggregations(
        #     all_data,
        #     group_columns=["country", "dayweek"],
        #     agg_column="phase",
        #     window_period=f"{n_years}y",
        #     window_closed="left",
        #     window_offset=f"{-n_years - 1}y",
        #     fn=fn
        # )

        # feature_c_wd = run_aggregations(
        #     all_data,
        #     group_columns=["country", "wd"],
        #     agg_column="phase",
        #     window_period=f"{n_years}y",
        #     window_closed="left",
        #     window_offset=None,
        #     fn=fn
        # )

        feature_bcm_dayweek = run_aggregations(
            all_data,
            group_columns=["brand", "country", "month", "dayweek"],
            agg_column="phase",
            window_period=f"{n_years}y",
            window_closed="left",
            window_offset=f"{-n_years - 1}y",
            fn=fn
        )

        all_data = all_data.with_columns(
            feature_bcm_wd.alias(f"bcm_phase_{fn}_{n_years}y_wd"),
            feature_bcm_formatted_date.alias(f"bcm_phase_{fn}_{n_years}y_formatted_date"),
            feature_bcm_dayweek.alias(f"bcm_phase_{fn}_{n_years}y_dayweek"),
            feature_bc_dayweek.alias(f"bc_phase_{fn}_{n_years}y_dayweek"),
            feature_bc_wd.alias(f"bc_phase_{fn}_{n_years}y_wd"),
            # feature_b_dayweek.alias(f"b_phase_{fn}_{n_years}y_dayweek"),
            # feature_b_wd.alias(f"b_phase_{fn}_{n_years}y_wd"),
            # feature_c_dayweek.alias(f"c_phase_{fn}_{n_years}y_dayweek"),
            # feature_c_wd.alias(f"c_phase_{fn}_{n_years}y_wd"),
        )

        
# %%
new_features = set(all_data.columns).difference(set(submission_data.columns))
new_features

# %%
all_data[["date", "country", "brand"] + list(new_features)].write_parquet("data/rolling_features_less_aggs_no_covid.parquet")

# %%
all_data[["date", "country", "brand"] + list(new_features)]

# %%
