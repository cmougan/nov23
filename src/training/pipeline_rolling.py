from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.helper.helper import check_assert_sum_1, metric, scale_prediction
from src.model_pipelines.dummy import DummyModelPipeline
from src.model_pipelines.hist_gbm import HistGradientBoostingModelPipeline
from src.model_pipelines.lgbm import LGBMModelPipeline
from src.utils.preprocessing import initial_add_date_cols
from src.model_pipelines.catboost import CatBoostModelPipeline
from src.model_pipelines.xgboost import XGBModelPipeline
from src.utils.preprocessing import add_date_cols
from src.utils.validation import initial_train_test_split_temporal

import shap
import matplotlib.pyplot as plt

pipelines = {
    "lgbm": LGBMModelPipeline(),
    "dummy": DummyModelPipeline(),
    "hist_gbm": HistGradientBoostingModelPipeline(),
    "catboost": CatBoostModelPipeline(),
    "xgboost": XGBModelPipeline(),
}


def main(model_pipeline, submission_timestamp, message, rolling_file_name):
    # Load data
    data_path = Path("data")
    submission_df_raw = pd.read_parquet(data_path / "submission_data.parquet")
    train_df = pd.read_parquet(data_path / "train_data.parquet")

    all_df = pd.concat([train_df, submission_df_raw])

    all_df = all_df.assign(
        main_channel=lambda x: x["main_channel"].astype(str).fillna("unknown"),
        ther_area=lambda x: x["ther_area"].astype(str).fillna("unknown"),
    ).pipe(initial_add_date_cols)

    rolling_df = pd.read_parquet(data_path / rolling_file_name)

    all_df = all_df.merge(rolling_df, on=["date", "brand", "country"], how="left")

    submission_df_raw["country_brand"] = (
        submission_df_raw["country"] + submission_df_raw["brand"]
    )
    all_df["country_brand"] = all_df["country"] + all_df["brand"]
    all_df = all_df[all_df.country_brand.isin(submission_df_raw.country_brand.unique())]
    all_df = all_df.drop(columns=["country_brand"])
    submission_df_raw = submission_df_raw.drop(columns=["country_brand"])

    df = all_df.query("(date < '2022-01-01')")  # & (date >= '2020-01-01')
    submission_df = all_df.query("date >= '2022-01-01'")

    y = df.phase
    X_raw = df.drop(columns=["phase"])

    if (
        (model_pipeline.model_name == "lgbm")
        or (model_pipeline.model_name == "catboost")
        or (model_pipeline.model_name == "xgboost")
    ):
        for col in ["country", "brand", "main_channel", "ther_area"]:
            X_raw[col] = X_raw[col].astype("category")
            submission_df[col] = submission_df[col].astype("category")

    # Prepare X_train, X_test, y_train and y_test for ML
    X_train_raw, X_test_raw, y_train, y_test = initial_train_test_split_temporal(
        X_raw, y, date_col="date"
    )
    X_train = X_train_raw.drop(
        columns=["formatted_date", "date", "monthly", "quarter_wm"]
    )
    X_test = X_test_raw.drop(
        columns=["formatted_date", "date", "monthly", "quarter_wm"]
    )
    X = X_raw.drop(columns=["formatted_date", "date", "monthly", "quarter_wm"])
    X_subm = submission_df.drop(
        columns=["formatted_date", "date", "monthly", "phase", "quarter_wm"]
    )

    # Get model and grid
    model_pipe = model_pipeline.get_pipeline()
    fit_kwargs = model_pipeline.get_fit_kwargs(X_train_raw)

    # train pipeline (use monthly as weights)
    model_pipe.fit(X_train, y_train, **fit_kwargs)

    X_train_raw["prediction"] = model_pipe.predict(X_train).clip(0, None)
    mse = mean_squared_error(X_train_raw["prediction"], y_train)
    print(f"Train MSE for {model_pipeline.model_name}: {mse}")
    X_train_pred = scale_prediction(X_train_raw)
    X_train_pred["phase"] = y_train
    check_assert_sum_1(X_train_pred)
    metric_train = metric(X_train_pred)

    print(f"Train metric for {model_pipeline.model_name}: {metric_train}")

    X_test_raw["prediction"] = model_pipe.predict(X_test)
    mse = mean_squared_error(X_test_raw["prediction"], y_test)
    print(f"Test MSE for {model_pipeline.model_name}: {mse}")
    X_test_pred = scale_prediction(X_test_raw)
    X_test_pred["phase"] = y_test
    check_assert_sum_1(X_test_pred)
    metric_test = metric(X_test_pred)

    print(f"Test metric for {model_pipeline.model_name}: {metric_test}")

    # Train model with best params
    fit_kwargs = model_pipeline.get_fit_kwargs(X_raw)
    model_pipe.fit(X, y, **fit_kwargs)

    PATH = Path("data")
    submission_df["prediction"] = model_pipe.predict(X_subm).clip(0, None)
    submission_df = scale_prediction(submission_df)
    check_assert_sum_1(submission_df)
    submission = pd.read_csv(PATH / "submission_template.csv")
    submission = submission_df[submission.columns]

    SAVE_PATH = Path("submissions")
    SAVE_PATH.mkdir(exist_ok=True)
    submission.to_csv(
        SAVE_PATH / f"submission_{submission_timestamp}_{message}.csv", index=False
    )

    # Explain model
    explainer = shap.TreeExplainer(model_pipe.named_steps["model"])
    # Pipe transform data
    X_xai = X_test.sample(1000)
    shap_values = explainer.shap_values(X_xai)
    # Save image
    shap.summary_plot(shap_values, X_xai, plot_type="bar", show=False)
    plt.savefig(f"shap_summary_plot_{model_pipeline.model_name}_{message}.png")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        help="Model to train",
        choices=pipelines.keys(),
    )
    parser.add_argument(
        "--message", type=str, default="", help="Message to add to submission file name"
    )
    parser.add_argument(
        "--rolling-file-name",
        type=str,
        default="rolling_features_less_aggs.parquet",
        help="File name of rolling features to use",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    now = datetime.now()
    message = args.message
    rolling_file_name = args.rolling_file_name
    main(
        pipelines[args.model],
        now.strftime("%Y-%m-%d_%H-%M-%S"),
        message,
        rolling_file_name,
    )
