# %% 
import numpy as np
import polars as pl

# %%

train_data = pl.read_parquet("data/train_data.parquet")
submission_data = pl.read_parquet("data/submission_data.parquet")
# %%
# all_data = pd.concat([train_data, submission_data]).sort_values(["brand", "country", "date"])
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
)

# %% add index
all_data_filter = all_data.filter(
    (pl.col("brand") == "ZVLFE") & (pl.col("country") == "Zamunda")
)
all_data_filter = all_data_filter.with_columns(
    # create a clumn that maps 2023-01-01 to 0101 and 2022-05-02 to 0502
    # Try disambiguating with `lit` or `col`.
        pl.col('date').apply(lambda x: x.strftime('%m%d')).alias('formatted_date')
)

all_data_filter


# %%
aggs = all_data_filter.rolling(
    index_column="date", by=["brand", "country", "month", "wd"], period="3y", closed="left", #offset="-1y"
).agg(pl.col("phase").mean().alias("phase_mean"))

aggs
# %%
all_data_filter.join(
    aggs, on=["date", "brand", "country", "month", "wd"], how="left"
).filter(
    # wd == 20
    (pl.col("month") == 12) & (pl.col("wd") == 10)
)[["phase", "month", "date", "wd", "phase_mean"]]



# %%
aggs = all_data_filter.rolling(
    index_column="date", by=["brand", "country", "month", "formatted_date"], period="3y", closed="left", #offset="-1y"
).agg(pl.col("phase").mean().alias("phase_mean"))

aggs
# %%
all_data_filter.join(
    aggs, on=["date", "brand", "country", "month", "formatted_date"], how="left"
).filter(
    (pl.col("formatted_date") == '0104')
)[["phase", "month", "date", "wd", "phase_mean"]]



# %%
aggs = all_data_filter.rolling(
    index_column="date", by=["brand", "country", "month", "dayweek"], period="3y", closed="left", offset="-4y"
).agg(pl.col("phase").mean().alias("phase_mean"))

aggs
# %%
all_data_filter.join(
    aggs, on=["date", "brand", "country", "month", "dayweek"], how="left"
).filter(
    # wd == 20
    (pl.col("month") == 12) & (pl.col("dayweek") == 3)
)[["phase", "month", "date", "dayweek", "phase_mean"]]


# %%
aggs = all_data_filter.rolling(
    index_column="date", by=["brand", "country", "month", "wd"], period="3y", closed="left", #offset="-1y"
).agg(pl.col("phase").mean().alias("phase_mean"))

aggs
# %%
all_data_filter.join(
    aggs, on=["date", "brand", "country", "month", "wd"], how="left"
).filter(
    # wd == 20
    (pl.col("month") == 12) & (pl.col("wd") == 10)
)[["phase", "month", "date", "wd", "phase_mean"]]


# %%
import numpy as np
import polars as pl


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
    aggs = data.rolling(
        index_column="date", by=group_columns, period=window_period, closed=window_closed, offset=window_offset
    ).agg(pl.col(agg_column).mean().alias(f"{agg_column}"))

    result = data.join(aggs, on=["date"] + group_columns, how="left")

    return result[f"{agg_column}"]


# Example usage:
result = run_aggregations(
    all_data,
    group_columns=["brand", "country", "month", "wd"],
    agg_column="phase",
    window_period="3y",
    window_closed="left",
    window_offset=None
)

print(result)

    
# %%
