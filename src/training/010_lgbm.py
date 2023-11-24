from pathlib import Path

import pandas as pd
from category_encoders import OrdinalEncoder, TargetEncoder
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.utils.validation import train_test_split_temporal


class ModelPipeline:
    def get_pipeline(self):
        return Pipeline(
            [
                ("encoder", OrdinalEncoder(cols=["brand", "country", "main_channel", "ther_area"])),
                ("model", LGBMRegressor(random_state=42, n_jobs=-1, verbose=0)),
            ]
        )

    def get_grid(self):
        return {
            "model__n_estimators": [50, 100, 200],
        }

def main():

    # Load data
    data_path = Path("data")
    df = pd.read_parquet(data_path / "train_data.parquet").assign(
        main_channel=lambda x: x["main_channel"].astype(str).fillna("unknown"),
        ther_area=lambda x: x["ther_area"].astype(str).fillna("unknown"),
    )

    X = df.drop(columns=["phase"])
    y = df.phase
    weights = df.monthly

    # Prepare X_train, X_test, y_train and y_test for ML
    X_train, X_test, y_train, y_test = train_test_split_temporal(
        X, y, date_col="date", date_split="2019-01-01"
    )
    weights_train = X_train["monthly"]
    weights_test = X_test["monthly"]
    X_train = X_train.drop(columns=["date", "monthly"])
    X_test = X_test.drop(columns=["date", "monthly"])
    X = X.drop(columns=["date", "monthly"])

    # Train lightgbm
    lgb_pipe = ModelPipeline().get_pipeline()

    # We use 3 different values of n_estimators and choose the best
    # one based on cv score
    pipeline_grid = ModelPipeline().get_grid()

    lgb_cv = GridSearchCV(
        lgb_pipe,
        param_grid=pipeline_grid,
        cv=3,
        scoring="neg_mean_squared_error",
    )

    lgb_cv.fit(X_train, y_train)

    print(f"Best params for lgbm: {lgb_cv.best_params_}")
    print(f"CV MSE for lgbm: {lgb_cv.best_score_}")

    mse = mean_squared_error(lgb_cv.predict(X_train), y_train)
    print(f"Train MSE for lgbm: {mse}")

    mse = mean_squared_error(lgb_cv.predict(X_test), y_test)
    print(f"Test MSE for lgbm: {mse}")

    # Train model with best params
    lgb_pipe.set_params(**lgb_cv.best_params_)
    lgb_pipe.fit(X, y)


if __name__ == "__main__":
    main()