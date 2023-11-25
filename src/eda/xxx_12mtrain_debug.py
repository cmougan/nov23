# %% 
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from src.helper.helper import check_assert_sum_1, metric, scale_prediction
from src.model_pipelines.dummy import DummyModelPipeline
from src.model_pipelines.lgbm import LGBMModelPipeline
from src.utils.preprocessing import add_date_cols
from src.utils.rolling import rolling_pl
from src.utils.validation import initial_train_test_split_temporal

pipelines = {
    "lgbm": LGBMModelPipeline(),
    "dummy": DummyModelPipeline(),
}

rolling_file_name = "rolling_features_less_aggs.parquet"



# %%
model_pipeline = LGBMModelPipeline()

# %%

# Load data
data_path = Path("data")
submission_df_raw = pd.read_parquet(data_path / "submission_data.parquet")
train_df = pd.read_parquet(data_path / "train_data.parquet").sample(100_000)

all_df = pd.concat([train_df, submission_df_raw])

all_df = all_df.assign(
    main_channel=lambda x: x["main_channel"].astype(str).fillna("unknown"),
    ther_area=lambda x: x["ther_area"].astype(str).fillna("unknown"),
).pipe(add_date_cols).sort_values(["brand", "country", "date"])

rolling_df = pd.read_parquet(data_path / rolling_file_name)

all_df = all_df.merge(rolling_df, on=["date", "brand", "country"], how="left")

block_df = all_df.copy()

for month in tqdm(range(1, 13)):
    # compute lags for the month
    all_df_cp = all_df.copy()
    all_df_pl = pl.from_pandas(all_df_cp)

    for rolling_period in [1, 2, 3]:
        rolling_feature = rolling_pl(
            all_df_pl,
            groupby_cols=["brand", "country"],
            column="phase",
            rolling_periods=1,
            # todo: check if this is right
            shift_periods=month + rolling_period - 1,
        )["phase"]

        all_df_pl.with_columns(
            rolling_feature.alias(f"phase_lag_{month}_rolling_{rolling_period}")
        )
    
    all_df_cp = all_df_pl.to_pandas()

    # split train, test and submission data
    df = all_df_cp.query("date < '2022-01-01'")
    submission_df = all_df_cp.query("date >= '2022-01-01'")
    y = df.phase
    X_raw = df.drop(columns=["phase"])

    # Prepare X_train, X_test, y_train and y_test for ML
    X_train_raw, X_test_raw, y_train, y_test = initial_train_test_split_temporal(
        X_raw, y, date_col="date"
    )
    X_train = X_train_raw.drop(columns=["formatted_date", "date", "monthly", "quarter_wm"])
    X_test = X_test_raw.drop(columns=["formatted_date", "date", "monthly", "quarter_wm"])
    X = X_raw.drop(columns=["formatted_date", "date", "monthly", "quarter_wm"])
    X_subm = submission_df.drop(columns=["formatted_date", "date", "monthly", "phase", "quarter_wm"])

    # fit model with lags
    model_pipe = model_pipeline.get_pipeline()
    fit_kwargs = model_pipeline.get_fit_kwargs(X_train_raw)
    model_pipe.fit(X_train, y_train, **fit_kwargs)

    # predict for validation data
    df["prediction"] = model_pipe.predict(X).clip(0, None)
    # is this going to work? careful with indexes and so on
    block_df.loc[
        (block_df.date.dt.month == month) & (block_df.date.dt.year < 2022),
    "prediction"] = df.loc[
        df.date.dt.month == month, "prediction"
    ]

    # retrain with 2021
    fit_kwargs = model_pipeline.get_fit_kwargs(X_raw)
    model_pipe.fit(X, y, **fit_kwargs)

    # predict for submission data
    submission_df["prediction"] = model_pipe.predict(X_subm).clip(0, None)

    block_df.loc[
        (block_df.date.dt.month == month) & (block_df.date.dt.year == 2022),
    "prediction"] = submission_df.loc[
        submission_df.date.dt.month == month, "prediction"
    ]
    print("Ratio nulls in block_df:", block_df.prediction.isnull().mean())
    print("Ratio nulls in block_df in 2022:", block_df.query("date.dt.year == 2022").prediction.isnull().mean())
    print("Ratio nulls in block_df in 2021:", block_df.query("date.dt.year == 2021").prediction.isnull().mean())
    print("Ratio nulls in block_df before 2021:", block_df.query("date.dt.year < 2021").prediction.isnull().mean())

# %%
block_df

# %%
# save submission
block_df = scale_prediction(block_df)
check_assert_sum_1(block_df)

# %%
metric_train = metric(block_df.query("date.dt.year < 2021"))
print(f"Train metric for {model_pipeline.model_name}: {metric_train}")

metric_test = metric(block_df.query("date.dt.year == 2021"))
print(f"Test metric for {model_pipeline.model_name}: {metric_test}")


# %%
PATH = Path("data")
submission = pd.read_csv(PATH / "submission_template.csv")
submission_df = block_df.query("date >= '2022-01-01'")
submission_df = submission_df[submission.columns]

# %%
SAVE_PATH = Path("submissions")
SAVE_PATH.mkdir(exist_ok=True)
# %%

submission.to_csv(SAVE_PATH / f"submission_{submission_timestamp}_{message}.csv", index=False)