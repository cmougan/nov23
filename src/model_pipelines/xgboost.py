from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline


class XGBModelPipeline:
    model_name: str = "xgboost"

    def get_pipeline(self):
        return Pipeline(
            [
                (
                    "model",
                    XGBRegressor(
                        random_state=42,
                        n_estimators=200,
                        n_jobs=-1,
                        enable_categorical=True,
                    ),
                ),
            ]
        )

    def get_grid(self):
        return {
            "model__num_trees": [500],
        }

    def get_fit_kwargs(self, X_train):
        # TODO: this needs to change to actual weights
        return {"model__sample_weight": X_train["quarter_wm"]}
