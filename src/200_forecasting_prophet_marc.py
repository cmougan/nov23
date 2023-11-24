# %%
import pandas as pd

# %%
train_data = pd.read_parquet("data/train_data.parquet")
print(train_data.isna().sum() / len(train_data))

# %%
submission_data = pd.read_parquet("data/submission_data.parquet")
print(submission_data.isna().sum() / len(train_data))

# %%
split_feats = ["brand", "country"]
train_data["code"] = [
    "_".join([str(brand), str(country)])
    for brand, country in zip(train_data["brand"], train_data["country"])
]


print(f'Unique brands: {train_data["brand"].nunique()}')
print(f'Unique countries: {train_data["country"].nunique()}')
print(f'Unique tuples: {train_data["code"].nunique()}')

# %%

submission_data["code"] = [
    "_".join([str(brand), str(country)])
    for brand, country in zip(submission_data["brand"], submission_data["country"])
]


print(f'Unique brands in submission: {submission_data["brand"].nunique()}')
print(f'Unique countries in submission: {submission_data["country"].nunique()}')
print(f'Unique tuples in submission: {submission_data["code"].nunique()}')
# %%

from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import numpy as np
from helper.helper import add_date_cols


# %%
categorical_feat = [
    "year",
    "quarter",
    "month",
    "ther_area",
    "main_channel",
    "brand",
    "country",
]


def transform_data(data, categorical_feat):
    data = add_date_cols(data)
    X = data.drop(columns=["code", "phase", "date", "monthly"])
    X[categorical_feat] = X[categorical_feat].astype("category")
    return X


y = train_data["phase"]
X = transform_data(train_data, categorical_feat)
# %%


# for cat in categorical_feat:
# isna = X.isna().iloc[:,0]
# le = LabelEncoder()
# X[cat] = le.fit_transform(X[cat])
# X[cat][isna] = np.nan

# %%

# %%
estimator = LGBMRegressor(categorical_feature=categorical_feat)
cv = GridSearchCV(
    estimator,
    {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.001, 0.1],
        "max_depth": [1, 2, 3, 5],
        "num_iterations": [1000],
    },
    cv=3,
)
cv_random = RandomizedSearchCV(
    estimator,
    param_distributions={
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.001, 0.1],
        "max_depth": [1, 2, 3, 5],
        "num_iterations": [1000],
    },
)
# %%
from sklearn.model_selection import cross_val_score

score = cross_val_score(estimator, X, y, cv=3)

# %%
from helper.helper import metric

estimator.fit(X, y)
y_pred = estimator.predict(X)
# %%
train_data["prediction"] = y_pred
metric(train_data)
# %%
X_subm = transform_data(submission_data, categorical_feat)
y_pred_sum = estimator.predict(X)
# %%
cv_random.fit(X, y)
print(cv.best_score_, cv.best_params_)
# %%


# import lightgbm
# from sklearn.model_selection import GridSearchCV

# models = {}
# for code,code_df in train_data.groupby('code'):
#     X = code_df.drop(columns = ['code','brand','country'])
#     estimator = lightgbm()
#     cv = GridSearchCV()
