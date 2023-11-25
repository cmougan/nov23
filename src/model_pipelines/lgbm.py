from category_encoders import OrdinalEncoder, TargetEncoder
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline


class LGBMModelPipeline:
    model_name: str = "lgbm"
    def get_pipeline(self):
        return Pipeline(
            [
                ("encoder", TargetEncoder(cols=["brand", "country", "main_channel", "ther_area"])),
                ("model", LGBMRegressor(random_state=42, n_jobs=-1, verbose=0)),
            ]
        )

    def get_grid(self):
        return {
            "model__n_estimators": [50],
        }

    def get_fit_kwargs(self, X_train):
        return {"model__sample_weight": X_train["monthly"]}