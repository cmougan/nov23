
from dataclasses import dataclass
from typing import Iterable, Union

import pandas as pd
from tqdm import tqdm


@dataclass
class RollingConfig:
    metric: str
    rolling_periods: int
    function: str
    groupby_cols: Union[str, list] = "brand"
    suffix: Union[str, None] = None

    def __repr__(self):
        if self.suffix is not None:
            return f"rolling_{self.metric}_{self.function}_{self.rolling_periods}_{self.suffix}"
        else:
            return f"rolling_{self.metric}_{self.function}_{self.rolling_periods}"

def rolling_fn(
    dataf: pd.DataFrame,
    groupby_cols: Union[str, list],
    column: str,
    function: str = "mean",
    rolling_periods: int = 1,
    shift_periods: int = 1,
    *args,
    **kwargs,
) -> pd.Series:
    return dataf.groupby(groupby_cols)[column].transform(
        lambda d: (
            d.shift(shift_periods)
            .rolling(rolling_periods, min_periods=1)
            .agg(function, *args, **kwargs)
        )
    )

def iterate_rolling_configs(metric="phase", n_periods=5) -> Iterable[RollingConfig]:
    """Return configuration for rolling aggregations, one at a time
    """
    # Features regarding wd
    for rolling_periods in range(1, n_periods + 1):
        for function in ["mean", "median"]:
            yield RollingConfig(
                metric=metric,
                rolling_periods=rolling_periods,
                function=function,
                groupby_cols=["brand", "country", "month", "wd"],
                suffix="bcm_wd",
            )

    # Features regarding dayweek
    for rolling_periods in range(1, n_periods + 1):
        for function in ["mean", "median"]:
            yield RollingConfig(
                metric=metric,
                rolling_periods=rolling_periods,
                function=function,
                groupby_cols=["brand", "country", "month", "dayweek"],
                suffix="bcm_dayweek",
            )

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling features to the dataframe.
    """
    df = df.copy().set_index("date")
    for rolling_config in tqdm(iterate_rolling_configs()):
        df = df.assign(
            **{
                f"{rolling_config}": lambda d: rolling_fn(
                    d,
                    column=rolling_config.metric,  # noqa: B023
                    groupby_cols=rolling_config.groupby_cols,  # noqa: B023
                    function=rolling_config.function,  # noqa: B023
                    rolling_periods=rolling_config.rolling_periods,  # noqa: B023
                )
            }
        )
    return df.reset_index()