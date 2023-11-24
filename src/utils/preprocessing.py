import numpy as np

import pandas as pd


def add_date_cols(df):
    """
    Convert date to datetime and add year, month, quarter and week columns
    """
    df["date"] = pd.to_datetime(df["date"])

    # create datetime columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week"] = df.date.dt.isocalendar().week
    return df


def calculate_time_features(df, time_column, out_name):
    """
    Calculate time features from the date. It applies cosine/sine transformations to the month and day of the week.
    """

    df[f"month_{out_name}"] = df[time_column].dt.month
    df[f"day_{out_name}"] = df[time_column].dt.dayofweek
    # df.drop(columns=[time_column], inplace=True)

    month_cycle = 12
    df[f"sin_month_{out_name}"] = np.sin(
        2 * np.pi * df[f"month_{out_name}"] / month_cycle
    )
    df[f"cos_month_{out_name}"] = np.cos(
        2 * np.pi * df[f"month_{out_name}"] / month_cycle
    )
    df.drop(columns=[f"month_{out_name}"], inplace=True)

    day_cycle = 7
    df[f"sin_day_{out_name}"] = np.sin(2 * np.pi * df[f"day_{out_name}"] / day_cycle)
    df[f"cos_day_{out_name}"] = np.cos(2 * np.pi * df[f"day_{out_name}"] / day_cycle)
    df.drop(columns=[f"day_{out_name}"], inplace=True)

    df[f"sin_day_{out_name}"] = np.sin(2 * np.pi * df[f"day_{out_name}"] / day_cycle)
    df[f"cos_day_{out_name}"] = np.cos(2 * np.pi * df[f"day_{out_name}"] / day_cycle)
    df.drop(columns=[f"day_{out_name}"], inplace=True)

    return df


def add_basic_lag_features(df, n_lags_day, n_lags_month, n_lags_yr):
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    for i in range(1, n_lags_day + 1):
        df[f"lag_phase_{i}_days"] = df.phase.shift(freq=f"{i}D")

    for i in range(n_lags_month):
        df[f"lag_phase_{i}_month"] = df.phase.shift(freq=f"{i}M")

    for i in range(n_lags_yr):
        df[f"lag_phase_{i}_yr"] = df.phase.shift(freq=f"{i}Y")

    return df
