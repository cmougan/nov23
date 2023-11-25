from typing import Union

import polars as pl


def rolling_pl(
    data: pl.DataFrame,
    groupby_cols: Union[str, list],
    column: str,
    function: str = "mean",
    rolling_periods: int = 1,
    shift_periods: int = 1,
    *args,
    **kwargs,
) -> pl.DataFrame:
    return (
        data
        .with_columns(pl.int_range(0,pl.count()).cast(pl.Int64).alias('index'))
        .rolling(
            index_column="index",
            period=f"{rolling_periods}i",
            closed="both",
            by=groupby_cols
        )
        .agg(
            getattr(pl.col(column).shift(shift_periods), function)(*args, **kwargs)
        )
        .drop("index")
    )