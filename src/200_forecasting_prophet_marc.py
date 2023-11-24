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
from sklearn.model_selection import GridSearchCV

estimator = LGBMRegressor()
cv = GridSearchCV(
    estimator,
    {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.001, 0.1],
        "max_depth": [1, 2, 3, 5],
    },
    cv=5,
)
X = train_data.drop(columns=["code", "phase"])
y = train_data["phase"]

cv.fit(X, y)
print(cv.best_score(), cv.best_params_)


# %%


# import lightgbm
# from sklearn.model_selection import GridSearchCV

# models = {}
# for code,code_df in train_data.groupby('code'):
#     X = code_df.drop(columns = ['code','brand','country'])
#     estimator = lightgbm()
#     cv = GridSearchCV()
