from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline


class TargetEncoderRegressor():

    def __init__(self, cols, agg_col):
        self.cols = cols
        self.agg_col = agg_col
        self.model = TargetEncoder(
            cols=[self.agg_col],
        )
    def fit(self, X, y):
        X = X.copy()
        X[self.agg_col] = X[self.cols].astype(str).apply(lambda x: "_".join(x), axis=1) 
        return self.model.fit(X, y)
    
    def predict(self, X):
        X = X.copy()
        X[self.agg_col] = X[self.cols].astype(str).apply(lambda x: "_".join(x), axis=1) 
        return self.model.transform(X)[self.agg_col]




class DummyModelPipeline:
    """Performs historical aggregation based on brand, country and wd"""
    model_name: str = "dummy"
    def get_pipeline(self):
        return Pipeline(
            [
                ("model", TargetEncoderRegressor(cols=["brand", "country", "wd"], agg_col="concat"))
            ]
        )

    def get_grid(self):
        return {}
    
    def get_fit_kwargs(self, X_train):
        return {}