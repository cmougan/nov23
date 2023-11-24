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


def calculate_time_features(df, time_column):
    """
    Calculate time features from the date. It applies cosine/sine transformations to the month and day of the week.
    """

    df[f"month"] = df[time_column].dt.month
    # df.drop(columns=[time_column], inplace=True)

    year_cycle = 12
    df[f"sin_year"] = np.sin(2 * np.pi * df[f"year"] / year_cycle)
    df[f"cos_year"] = np.cos(2 * np.pi * df[f"year"] / year_cycle)
    df.drop(columns=[f"year"], inplace=True)

    quarter_cycle = 4
    df[f"sin_quarter"] = np.sin(2 * np.pi * df[f"quarter"] / quarter_cycle)
    df[f"cos_quarter"] = np.cos(2 * np.pi * df[f"quarter"] / quarter_cycle)
    df.drop(columns=[f"quarter"], inplace=True)

    month_cycle = 12
    df[f"sin_month"] = np.sin(2 * np.pi * df[f"month"] / month_cycle)
    df[f"cos_month"] = np.cos(2 * np.pi * df[f"month"] / month_cycle)
    df.drop(columns=[f"month"], inplace=True)

    week_cycle = 53
    df[f"sin_week"] = np.sin(2 * np.pi * df[f"week"] / week_cycle)
    df[f"cos_week"] = np.cos(2 * np.pi * df[f"week"] / week_cycle)
    df.drop(columns=[f"week"], inplace=True)

    day_cycle = 7
    df[f"sin_day"] = np.sin(2 * np.pi * df[f"dayweek"] / day_cycle)
    df[f"cos_day"] = np.cos(2 * np.pi * df[f"dayweek"] / day_cycle)
    df.drop(columns=[f"dayweek"], inplace=True)

    return df
