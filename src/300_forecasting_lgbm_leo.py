# %%
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from src.utils.preprocessing import add_date_cols, calculate_time_features
from src.helper.helper import metric, scale_prediction, check_assert_sum_1
from src.utils.validation import train_test_split_temporal
from pathlib import Path

# %%
train_data = pd.read_parquet("data/train_data.parquet")
print(train_data.isna().sum() / len(train_data))

# train_data = train_data.query("date >= '2019-01-01'")


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
# %% Add features

train_data = add_date_cols(train_data)
train_data = calculate_time_features(train_data, "date")

submission_data = add_date_cols(submission_data)
submission_data = calculate_time_features(submission_data, "date")


# %%
X = train_data.drop(columns=["code", "phase"])
y = train_data["phase"]
X_subm = submission_data.drop(columns=["code"])


# %% Train Test Split
X_train, X_val, y_train, y_val = train_test_split_temporal(
    X, y, date_split="2021-01-01"
)

X_train
X_val

# %%


categorical_feat = [
    "ther_area",
    "main_channel",
    "brand",
    "country",
]

X[categorical_feat] = X[categorical_feat].astype("category")
X_train[categorical_feat] = X_train[categorical_feat].astype("category")
X_val[categorical_feat] = X_val[categorical_feat].astype("category")
X_subm[categorical_feat] = X_subm[categorical_feat].astype("category")


# %%

model = LGBMRegressor(categorical_feature=categorical_feat)
# cv = GridSearchCV(
#     model,
#     {
#         # "num_iterations": [100],
#         "n_estimators": [50, 100, 200, 500],
#         "learning_rate": [0.01, 0.1],
#         "max_depth": [3],
#         "num_leaves": [2, 5, 10],
#     },
#     cv=5,
# )
model.fit(X_train.drop(columns=["date", "monthly"]), y_train)
# print(cv.best_score_, cv.best_params_)

# %% #use best model
# model = LGBMRegressor(**cv.best_params_)
# model.fit(X_train.drop(columns=["date", "monthly"]), y_train)

# model = cv.best_estimator_
# %% Check performance
## Train
X_train["prediction"] = model.predict(X_train.drop(columns=["date", "monthly"]))
X_train = add_date_cols(X_train)
X_train = scale_prediction(X_train)
X_train["phase"] = y_train
print("Train Performance:", metric(X_train))

## CV
X_val["prediction"] = model.predict(X_val.drop(columns=["date", "monthly"]))
X_val = add_date_cols(X_val)
X_val = scale_prediction(X_val)
X_val["phase"] = y_val
print("CV Performance:", metric(X_val))

# %%
# Assert phase sums 1 for each year-month-brand-country
check_assert_sum_1(X_train)
check_assert_sum_1(X_val)


# %% Train with all data
model = LGBMRegressor(categorical_feature=categorical_feat)
model.fit(X.drop(columns=["date", "monthly"]), y)

# %% PREPARE SUBMISSION
X_subm["prediction"] = model.predict(X_subm.drop(columns=["date"]))
X_subm = add_date_cols(X_subm)
X_subm = scale_prediction(X_subm)
check_assert_sum_1(X_subm)

# %%
PATH = Path("data")
submission = pd.read_csv(PATH / "submission_template.csv")
submission = X_subm[submission.columns]

# %%
SAVE_PATH = Path("submissions")
submission.to_csv(SAVE_PATH / "submission.csv", index=False)

# %%
