from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.helper.helper import check_assert_sum_1, metric, scale_prediction
from src.model_pipelines.dummy import DummyModelPipeline
from src.model_pipelines.lgbm import LGBMModelPipeline
from src.utils.preprocessing import add_date_cols
from src.utils.validation import train_test_split_temporal

pipelines = {
    "lgbm": LGBMModelPipeline(),
    "dummy": DummyModelPipeline(),
}



def main(model_pipeline):

    # Load data
    data_path = Path("data")
    df = pd.read_parquet(data_path / "train_data.parquet").assign(
        main_channel=lambda x: x["main_channel"].astype(str).fillna("unknown"),
        ther_area=lambda x: x["ther_area"].astype(str).fillna("unknown"),
    ).pipe(add_date_cols)

    rolling_df = pd.read_parquet(data_path / "rolling_features.parquet")

    df = df.merge(rolling_df, on=["date", "brand", "country"], how="left")

    y = df.phase
    X_raw = df.drop(columns=["phase"])

    # Prepare X_train, X_test, y_train and y_test for ML
    X_train_raw, X_test_raw, y_train, y_test = train_test_split_temporal(
        X_raw, y, date_col="date", date_split="2019-01-01"
    )
    X_train = X_train_raw.drop(columns=["date", "monthly"])
    X_test = X_test_raw.drop(columns=["date", "monthly"])
    X = X_raw.drop(columns=["date", "monthly"])

    # Get model and grid
    model_pipe = model_pipeline.get_pipeline()
    pipeline_grid = model_pipeline.get_grid()
    fit_kwargs = model_pipeline.get_fit_kwargs(X_train_raw)

    # Define cv pipeline
    model_cv = GridSearchCV(
        model_pipe,
        param_grid=pipeline_grid,
        cv=3,
        scoring="neg_mean_squared_error",
    )

    # train pipeline (use monthly as weights)
    model_cv.fit(X_train, y_train, **fit_kwargs)

    print(f"Best params for {model_pipeline.model_name}: {model_cv.best_params_}")
    print(f"CV MSE for {model_pipeline.model_name}: {model_cv.best_score_}")

    X_train_raw["prediction"] = model_cv.predict(X_train)
    mse = mean_squared_error(X_train_raw["prediction"], y_train)
    print(f"Train MSE for {model_pipeline.model_name}: {mse}")
    X_train_pred = scale_prediction(X_train_raw)
    X_train_pred["phase"] = y_train
    check_assert_sum_1(X_train_pred)
    metric_train = metric(X_train_pred)

    print(f"Train metric for {model_pipeline.model_name}: {metric_train}")

    X_test_raw["prediction"] = model_cv.predict(X_test)
    mse = mean_squared_error(X_test_raw["prediction"], y_test)
    print(f"Test MSE for {model_pipeline.model_name}: {mse}")
    X_test_pred = scale_prediction(X_test_raw)
    X_test_pred["phase"] = y_test
    check_assert_sum_1(X_test_pred)
    metric_test = metric(X_test_pred)

    print(f"Test metric for {model_pipeline.model_name}: {metric_test}")


    # Train model with best params
    fit_kwargs = model_pipeline.get_fit_kwargs(X_raw)
    model_pipe.set_params(**model_cv.best_params_)
    model_pipe.fit(X, y, **fit_kwargs)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="lgbm", help="Model to train", choices=pipelines.keys()
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(pipelines[args.model])