from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline


class CatBoostModelPipeline:
    model_name: str = "catboost"

    def get_pipeline(self):
        return Pipeline(
            [
                (
                    "model",
                    CatBoostRegressor(
                        random_state=42,
                        num_trees=500,
                        thread_count=-1,
                        cat_features=[
                            "brand",
                            "country",
                            "Week_day",
                            "main_channel",
                            "ther_area",
                        ],
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
