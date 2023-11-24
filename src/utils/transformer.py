# Sklearn compatible transformer for feature engineering
from sklearn.base import BaseEstimator, TransformerMixin


class DropCols(BaseEstimator, TransformerMixin):
    """
    Drops columns from a dataframe
    If column is not present, it does nothing
    """

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.cols:
            try:
                X = X.drop(col, axis=1)
            except KeyError:
                pass
        return X


class GetNumerical(BaseEstimator, TransformerMixin):
    """
    Get numerical columns from a dataframe
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X._get_numeric_data()
