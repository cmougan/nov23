# %% TODO: Make sure it works properly. There are tons of features, maybe dont make sense.
# What happens with the nans?
import numpy as np
import polars as pl

# %%

train_data = pl.read_parquet("data/train_data.parquet")
submission_data = pl.read_parquet("data/submission_data.parquet")
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
        pl.col("date").apply(lambda x: x.strftime("%m%d")).alias("formatted_date")
    )
    .with_columns(
        pl.col("week") << pl.col("date").dt.week(),
        #     pl.col("year") << pl.col("date").dt.year(),
        #     pl.col("quarter") << pl.col("date").dt.quarter(),
    )
)


# %%
def run_aggregations(
    data,
    group_columns,
    agg_column,
    window_period,
    window_closed="left",
    window_offset=None,
):
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
            index_column="date",
            by=group_columns,
            period=window_period,
            closed=window_closed,
        ).agg(pl.col(agg_column).mean().alias(f"{agg_column}_mean"))
    else:
        aggs = data.rolling(
            index_column="date",
            by=group_columns,
            period=window_period,
            closed=window_closed,
            offset=window_offset,
        ).agg(pl.col(agg_column).mean().alias(f"{agg_column}_mean"))

    result_aggs = data.join(aggs, on=["date"] + group_columns, how="left")

    return result_aggs[f"{agg_column}_mean"]


# %%
for n_years in range(1, 6):
    feature_wd_month = run_aggregations(
        all_data,
        group_columns=["brand", "country", "month", "wd"],
        agg_column="phase",
        window_period=f"{n_years}y",
        window_closed="left",
        window_offset=None,
    )

    feature_formatted_date_month = run_aggregations(
        all_data,
        group_columns=["brand", "country", "month", "formatted_date"],
        agg_column="phase",
        window_period=f"{n_years}y",
        window_closed="left",
        window_offset=None,
    )

    feature_dayweek_month = run_aggregations(
        all_data,
        group_columns=["brand", "country", "month", "dayweek"],
        agg_column="phase",
        window_period=f"{n_years}y",
        window_closed="left",
        window_offset=f"{-n_years - 1}y",
    )

    # feature_ther_area_month = run_aggregations(
    #     all_data,
    #     group_columns=["brand", "country", "month", "ther_area"],
    #     agg_column="phase",
    #     window_period=f"{n_years}y",
    #     window_closed="left",
    #     window_offset=f"{-n_years - 1}y",
    # )

    # feature_hospital_rate_month = run_aggregations(
    #     all_data,
    #     group_columns=["brand", "country", "month", "hospital_rate"],
    #     agg_column="phase",
    #     window_period=f"{n_years}y",
    #     window_closed="left",
    #     window_offset=f"{-n_years - 1}y",
    # )

    all_data = all_data.with_columns(
        feature_wd_month.alias(f"phase_mean_month_{n_years}y_wd"),
        feature_formatted_date_month.alias(
            f"phase_mean_month_{n_years}y_formatted_date"
        ),
        feature_dayweek_month.alias(f"phase_mean_month_{n_years}y_dayweek"),
        # feature_ther_area_month.alias(f"phase_mean_month_{n_years}y_ther_area"),
        # feature_hospital_rate_month.alias(f"phase_mean_month_{n_years}y_hospital_rate"),
    )


for month in range(1, 12):
    feature_wd = run_aggregations(
        all_data,
        group_columns=["brand", "country", "wd"],
        agg_column="phase",
        window_period=f"{month}m",
        window_closed="left",
        window_offset=None,
    )

    feature_formatted_date = run_aggregations(
        all_data,
        group_columns=["brand", "country", "formatted_date"],
        agg_column="phase",
        window_period=f"{month}m",
        window_closed="left",
        window_offset=None,
    )

    feature_dayweek = run_aggregations(
        all_data,
        group_columns=["brand", "country", "dayweek"],
        agg_column="phase",
        window_period=f"{month}m",
        window_closed="left",
        window_offset=f"{-month - 1}m",
    )

    # feature_ther_area = run_aggregations(
    #     all_data,
    #     group_columns=["brand", "country", "ther_area"],
    #     agg_column="phase",
    #     window_period=f"{n_years}y",
    #     window_closed="left",
    #     window_offset=f"{-n_years - 1}y",
    # )

    # feature_hospital_rate = run_aggregations(
    #     all_data,
    #     group_columns=["brand", "country", "hospital_rate"],
    #     agg_column="phase",
    #     window_period=f"{n_years}y",
    #     window_closed="left",
    #     window_offset=f"{-n_years - 1}y",
    # )

    all_data = all_data.with_columns(
        feature_wd.alias(f"phase_mean_year_{month}m_wd"),
        feature_formatted_date.alias(f"phase_mean_year_{month}m_formatted_date"),
        feature_dayweek.alias(f"phase_mean_year_{month}m_dayweek"),
        # feature_ther_area.alias(f"phase_mean_year_{n_years}y_ther_area"),
        # feature_hospital_rate.alias(f"phase_mean_year_{n_years}y_hospital_rate"),
    )

# %%
all_data = all_data.sort(["brand", "date"])
# %%
for n_years in range(1, 6):
    feature_wd_month_nocountry = run_aggregations(
        all_data,
        group_columns=["brand", "month", "wd"],
        agg_column="phase",
        window_period=f"{n_years}y",
        window_closed="left",
        window_offset=None,
    )

    feature_formatted_date_month_nocountry = run_aggregations(
        all_data,
        group_columns=["brand", "month", "formatted_date"],
        agg_column="phase",
        window_period=f"{n_years}y",
        window_closed="left",
        window_offset=None,
    )

    feature_dayweek_month_nocountry = run_aggregations(
        all_data,
        group_columns=["brand", "month", "dayweek"],
        agg_column="phase",
        window_period=f"{n_years}y",
        window_closed="left",
        window_offset=f"{-n_years - 1}y",
    )

    # feature_ther_area_month_nocountry = run_aggregations(
    #     all_data,
    #     group_columns=["brand", "month", "ther_area"],
    #     agg_column="phase",
    #     window_period=f"{n_years}y",
    #     window_closed="left",
    #     window_offset=f"{-n_years - 1}y",
    # )

    # feature_hospital_rate_month_nocountry = run_aggregations(
    #     all_data,
    #     group_columns=["brand", "month", "hospital_rate"],
    #     agg_column="phase",
    #     window_period=f"{n_years}y",
    #     window_closed="left",
    #     window_offset=f"{-n_years - 1}y",
    # )

    all_data = all_data.with_columns(
        feature_wd_month_nocountry.alias(f"phase_mean_month_nocountry_{n_years}y_wd"),
        feature_formatted_date_month_nocountry.alias(
            f"phase_mean_month_nocountry_{n_years}y_formatted_date"
        ),
        feature_dayweek_month_nocountry.alias(
            f"phase_mean_month_nocountry_{n_years}y_dayweek"
        ),
        # feature_ther_area_month_nocountry.alias(
        #     f"phase_mean_month_nocountry_{n_years}y_ther_area"
        # ),
        # feature_hospital_rate_month_nocountry.alias(
        #     f"phase_mean_month_nocountry_{n_years}y_hospital_rate"
        # ),
    )


for month in range(1, 12):
    feature_wd_nocountry = run_aggregations(
        all_data,
        group_columns=["brand", "wd"],
        agg_column="phase",
        window_period=f"{month}m",
        window_closed="left",
        window_offset=None,
    )

    feature_formatted_date_nocountry = run_aggregations(
        all_data,
        group_columns=["brand", "formatted_date"],
        agg_column="phase",
        window_period=f"{month}m",
        window_closed="left",
        window_offset=None,
    )

    feature_dayweek_nocountry = run_aggregations(
        all_data,
        group_columns=["brand", "dayweek"],
        agg_column="phase",
        window_period=f"{month}m",
        window_closed="left",
        window_offset=f"{-month - 1}m",
    )

    # feature_ther_area_nocountry = run_aggregations(
    #     all_data,
    #     group_columns=["brand", "ther_area"],
    #     agg_column="phase",
    #     window_period=f"{month}m",
    #     window_closed="left",
    #     window_offset=f"{-month - 1}m",
    # )

    # feature_hospital_rate_nocountry = run_aggregations(
    #     all_data,
    #     group_columns=["brand", "hospital_rate"],
    #     agg_column="phase",
    #     window_period=f"{month}m",
    #     window_closed="left",
    #     window_offset=f"{-month - 1}m",
    # )

    all_data = all_data.with_columns(
        feature_wd_nocountry.alias(f"phase_mean_year_nocountry_{month}m_wd"),
        feature_formatted_date_nocountry.alias(
            f"phase_mean_year_nocountry_{month}m_formatted_date"
        ),
        feature_dayweek_nocountry.alias(f"phase_mean_year_nocountry_{month}m_dayweek"),
        # feature_ther_area_nocountry.alias(
        #     f"phase_mean_year_nocountry_{month}m_ther_area"
        # ),
        # feature_hospital_rate_nocountry.alias(
        #     f"phase_mean_year_nocountry_{month}m_hospital_rate"
        # ),
    )


# %%
all_data.describe()
# all_data.corr()
# %%
aggs = all_data.rolling(
    index_column="date",
    by=["brand", "country", "month", "wd"],
    period="3y",
    closed="left",
).agg(pl.col("phase").mean())
# %%
all_data.join(aggs, on=["date"] + ["brand", "country", "month", "wd"], how="left")
# %%
new_features = set(all_data.columns).difference(set(submission_data.columns))
# %%
all_data[["date", "country", "brand"] + list(new_features)].write_csv(
    "data/rolling_features.csv"
)

# %%
all_data[["date", "country", "brand"] + list(new_features)].write_parquet(
    "data/rolling_features.parquet"
)

# %%
all_data[["date", "country", "brand"] + list(new_features)].tail(100).write_csv(
    "data/rolling_features_100.csv"
)
# %%
