from category_encoders import OrdinalEncoder, TargetEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


class HistGradientBoostingModelPipeline:
    model_name: str = "hist_gradient_boosting"
    
    def get_pipeline(self):
        return Pipeline(
            [
                ("encoder", TargetEncoder(cols=["country", "brand", "main_channel", "ther_area"])),
                ("model", HistGradientBoostingRegressor(random_state=42, verbose=0, max_iter=300)),
            ]
        )

    def get_grid(self):
        return {
            "model__max_iter": [500],
        }

    def get_fit_kwargs(self, X_train):
        return {
            "model__sample_weight": X_train["quarter_wm"],
        }
