from typing import List

import numpy as np
import pandas as pd


def initial_add_date_cols(df):
    """
    Convert date to datetime and add year, month, quarter and week columns
    """
    df["date"] = pd.to_datetime(df["date"])

    # create datetime columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week"] = df.date.dt.isocalendar().week
    df["quarter_w"] = np.where(
        df["quarter"] == 1,
        1,
        np.where(df["quarter"] == 2, 0.75, np.where(df["quarter"] == 3, 0.66, 0.5)),
    )
    df["quarter_wm"] = df["quarter_w"] * df["monthly"]

    return df



def add_date_cols(df, add_weights=True):
    """
    Convert date to datetime and add year, month, quarter and week columns
    """
    df["date"] = pd.to_datetime(df["date"])

    # create datetime columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["dayweek"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.day_of_year // 7  # df["date"].dt.isocalendar().week

    if "num_week_month" not in df.columns:
        df = get_week_inmonth(df)  # Add week in month

        df["Week_day"] = (
            df.num_week_month.astype(str) + "-" + df.date.dt.dayofweek.astype(str)
        )
    df["quarter_w"] = np.where(
        df["quarter"] == 1,
        1,
        np.where(df["quarter"] == 2, 0.75, np.where(df["quarter"] == 3, 0.66, 0.5)),
    )
    if add_weights:
        df["quarter_wm"] = df["quarter_w"] * df["monthly"]

    return df


def calculate_time_features(df, time_column, drop=False):
    """
    Calculate time features from the date. It applies cosine/sine transformations to the month and day of the week.
    """

    df[f"month"] = df[time_column].dt.month
    # df.drop(columns=[time_column], inplace=True)

    year_cycle = 12
    df[f"sin_year"] = np.sin(2 * np.pi * df[f"year"] / year_cycle)
    df[f"cos_year"] = np.cos(2 * np.pi * df[f"year"] / year_cycle)

    quarter_cycle = 4
    df[f"sin_quarter"] = np.sin(2 * np.pi * df[f"quarter"] / quarter_cycle)
    df[f"cos_quarter"] = np.cos(2 * np.pi * df[f"quarter"] / quarter_cycle)

    month_cycle = 12
    df[f"sin_month"] = np.sin(2 * np.pi * df[f"month"] / month_cycle)
    df[f"cos_month"] = np.cos(2 * np.pi * df[f"month"] / month_cycle)

    week_cycle = 53
    df[f"sin_week"] = np.sin(2 * np.pi * df[f"week"] / week_cycle)
    df[f"cos_week"] = np.cos(2 * np.pi * df[f"week"] / week_cycle)

    day_cycle = 7
    df[f"sin_day"] = np.sin(2 * np.pi * df[f"dayweek"] / day_cycle)
    df[f"cos_day"] = np.cos(2 * np.pi * df[f"dayweek"] / day_cycle)

    if drop:
        df.drop(columns=[f"dayweek"], inplace=True)
        df.drop(columns=[f"week"], inplace=True)
        df.drop(columns=[f"month"], inplace=True)
        df.drop(columns=[f"quarter"], inplace=True)
        df.drop(columns=[f"year"], inplace=True)

    return df


def add_basic_lag_features(df, n_lags_day, n_lags_month, n_lags_yr):
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for i in range(1, n_lags_day + 1):
        df[f"lag_phase_{i}_days"] = df.phase.shift(i, freq="D")

    for i in range(1, n_lags_month + 1):
        df[f"lag_phase_{i}_month"] = np.nanmean(
            [
                df.phase.shift(30 * i, freq="D"),
                df.phase.shift(30 * i + 1, freq="D"),
                df.phase.shift(30 * i - 1, freq="D"),
            ],
            axis=0,
        )

    for i in range(1, n_lags_yr + 1):
        df[f"lag_phase_{i}_yr"] = np.nanmean(
            [
                df.phase.shift(365 * i, freq="D"),
                df.phase.shift(365 * i + 1, freq="D"),
                df.phase.shift(365 * i - 1, freq="D"),
            ],
            axis=0,
        )

    return df.reset_index()


def add_basic_lag_features_week(df, n_lags_day, n_lags_month, n_lags_week):
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for i in range(1, n_lags_day + 1):
        df[f"lag_phase_{i}_days"] = df.phase.shift(i, freq="D")

    for i in range(1, n_lags_month + 1):
        df[f"lag_phase_{i}_month_exact"] = df.phase.shift(30 * i, freq="D")
        df[f"lag_phase_{i}_month_before"] = df.phase.shift(30 * i - 1, freq="D")
        df[f"lag_phase_{i}_month_after"] = df.phase.shift(30 * i + 1, freq="D")

    for i in range(1, n_lags_week + 1):
        df[f"lag_phase_{i}_week_exact"] = df.phase.shift(7 * i, freq="D")
        df[f"lag_phase_{i}_week_before"] = df.phase.shift(7 * i - 1, freq="D")
        df[f"lag_phase_{i}_week_after"] = df.phase.shift(7 * i + 1, freq="D")

    return df.reset_index()


def add_basic_valid_lag_features(df, n_lags_day, n_lags_month, n_lags_yr):
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for i in range(1, n_lags_day + 1):
        df[f"lag_phase_{i}_days"] = df.phase.shift(365 + i, freq="D")

    for i in range(1, n_lags_month + 1):
        df[f"lag_phase_{i}_month"] = (
            df.phase.shift(365 + 30 * i, freq="D")
            .fillna(df.phase.shift(365 + 30 * i + 1, freq="D"))
            .fillna(df.phase.shift(365 + 30 * i - 1, freq="D"))
        )

    for i in range(1, n_lags_yr + 1):
        df[f"lag_phase_{i}_yr"] = (
            df.phase.shift(365 + 365 * i, freq="D")
            .fillna(df.phase.shift(365 + 365 * i + 1, freq="D"))
            .fillna(df.phase.shift(365 + 365 * i - 1, freq="D"))
        )

    return df.reset_index()


def add_basic_valid_lag_features_v2(df, n_lags_day, n_lags_month, n_lags_yr):
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for i in range(1, n_lags_day + 1):
        df[f"lag_phase_{i}_days"] = df.phase.shift(365 + i, freq="D")

    for i in range(1, n_lags_month + 1):
        df[f"lag_phase_{i}_month_exact"] = df.phase.shift(365 + 30 * i, freq="D")
        df[f"lag_phase_{i}_month_before"] = df.phase.shift(365 + 30 * i - 1, freq="D")
        df[f"lag_phase_{i}_month_after"] = df.phase.shift(365 + 30 * i + 1, freq="D")

    for i in range(1, n_lags_yr + 1):
        df[f"lag_phase_{i}_yr_exact"] = df.phase.shift(365 + 365 * i, freq="D")
        df[f"lag_phase_{i}_yr_before"] = df.phase.shift(365 + 365 * i - 1, freq="D")
        df[f"lag_phase_{i}_yr_after"] = df.phase.shift(365 + 365 * i + 1, freq="D")

    return df.reset_index()


def add_basic_valid_lag_features_neighbour(df, n_lags_week, n_lags_day, n_lags_month):
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    for i in range(-n_lags_day, n_lags_day + 1):
        df[f"lag_phase_{i}_days"] = df.phase.shift(365 + i, freq="D")

    for i in range(-n_lags_month, n_lags_month + 1):
        df[f"lag_phase_{i}_month_exact"] = df.phase.shift(365 + 30 * i, freq="D")
        df[f"lag_phase_{i}_month_before"] = df.phase.shift(365 + 30 * i - 1, freq="D")
        df[f"lag_phase_{i}_month_after"] = df.phase.shift(365 + 30 * i + 1, freq="D")

    for i in range(-n_lags_week, n_lags_week + 1):
        df[f"lag_phase_{i}_yr_exact"] = df.phase.shift(365 + 7 * i, freq="D")
        df[f"lag_phase_{i}_yr_before"] = df.phase.shift(365 + 7 * i - 1, freq="D")
        df[f"lag_phase_{i}_yr_after"] = df.phase.shift(365 + 7 * i + 1, freq="D")

    return df.reset_index()


def impute_end_dates(data, threshold_date):
    if data < pd.Timestamp(threshold_date):
        return data
    else:
        return ""


def impute_start_dates(data, threshold_date):
    if data > pd.Timestamp(threshold_date):
        return data
    else:
        return ""


def get_days_sincestart_toend(
    df: pd.DataFrame, threshold_start="2013-02-01", threshold_end="2022-12-01"
):
    """
    Compute the number of days between the start_date and date
    Compute the number of days between the end_date and date

    It could give some sense of maturity on the brand. If we have no information on the date of start/end of the brand, we have a nan.

    Parameters:
    -----------
    df: pd.DataFrame
        dataframe with columns brand, country and date
    threshold_start: str
        date from which we consider we can ensure the brand has launch from that day (not inclusive)
    threshold_end: str
        date from which we consider we can ensure the brand has stopped selling before that day (not inclusive)

    Returns:
    --------
    df: pd.DataFrame
        dataframe with columns brand, country, date, days_since_start, days_until_end
    """

    start_end_dates = (
        df.copy()
        .groupby(["brand", "country"])
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        .reset_index()
    )

    start_end_dates = start_end_dates.assign(
        end_date=start_end_dates.end_date.apply(
            impute_end_dates, threshold_date=threshold_end
        ),
        start_date=start_end_dates.start_date.apply(
            impute_start_dates, threshold_date=threshold_start
        ),
    )
    # Convert to timestamps again
    start_end_dates = start_end_dates.assign(
        end_date=lambda x: pd.to_datetime(x.end_date),
        start_date=lambda x: pd.to_datetime(x.start_date),
    )

    df = df.merge(start_end_dates, on=["brand", "country"], how="left")

    # Compute the number of days between the start_date ad date
    df = df.assign(
        days_since_start=lambda x: (x.date - x.start_date).dt.days,
        days_until_end=lambda x: (x.end_date - x.date).dt.days,
    )
    df.drop(columns=["start_date", "end_date"], inplace=True)
    return df


def get_ther_areas_group(df, grouping: List):
    """
    Get the ther_areas that are in the group.
    It can be used with brand, brand and country, etc.
    """
    df = df.copy()
    name = "_".join(grouping)
    unique_ther_areas = df.groupby(grouping).apply(
        lambda x: pd.Series(
            {
                f"unique_areas_{name}": "-".join(
                    set(x["ther_area"].dropna().unique().tolist())
                )
            }
        )
    )

    df = df.merge(unique_ther_areas, on=grouping)

    return df


def get_ther_areas_data(df):
    """
    Get the ther_areas that are in the group.
    It can be used with brand, brand and country, etc.

    """
    df = df.copy()
    df = get_ther_areas_group(df, ["brand", "country"])
    df = get_ther_areas_group(df, ["brand"])
    df = get_ther_areas_group(df, ["country"])
    return df


def get_week_inmonth(df: pd.DataFrame):
    """
    Get the week in the month
    """
    df = df.copy()
    tmp = (
        df.groupby(["year", "month"])
        .agg(
            first_week=("week", "min"),
        )
        .reset_index()
    )

    df = df.merge(tmp, on=["year", "month"]).assign(
        num_week_month=lambda x: x.week - x.first_week
    )
    return df


def add_covid_exo(df, start_from="2020-03-01", end_at=None):
    df = df.copy()
    df["covid"] = 0
    df.loc[df.date >= start_from, "covid"] = 1
    if end_at is not None:
        df.loc[df.date >= end_at, "covid"] = 0

    df["covid_short"] = 0
    df.loc[(df.date >= start_from) & (df.date < "2020-06-01"), "covid_short"] = 1

    df["covid_year"] = 0
    df.loc[(df.date >= start_from) & (df.date < "2021-01-01"), "covid_short"] = 1

    return df
