# %%
import pandas as pd

# %%
train_data = pd.read_parquet("data/205_feateng_train_data.parquet").drop(
    columns=["level_0", "level_1"]
)
print(train_data.isna().sum() / len(train_data))

# %%
submission_data = pd.read_parquet("data/205_feateng_test_data.parquet").drop(
    columns=["level_0", "level_1"]
)
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

unique_codes_test = submission_data.code.unique().tolist()
print(f'Unique brands in submission: {submission_data["brand"].nunique()}')
print(f'Unique countries in submission: {submission_data["country"].nunique()}')
print(f'Unique tuples in submission: {submission_data["code"].nunique()}')
# %%
train_data = train_data.query('code in @unique_codes_test and date>"2017-01-01"')
# %%
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import numpy as np
from src.helper.helper import add_date_cols
from category_encoders import TargetEncoder, OrdinalEncoder

# %%
categorical_feat = [
    "ther_area",
    "main_channel",
    "brand",
    "country",
    "Week_day",
]


def transform_data(data, categorical_feat):
    data = add_date_cols(data, add_weights=False)
    X = data.drop(columns=["code", "phase"])
    X[categorical_feat] = X[categorical_feat].astype("category")
    return X


y = train_data["phase"]
X = transform_data(train_data, categorical_feat)
# %%

lag_feats = [k for k in X.keys() if "lag_" in k]
X[lag_feats] = X[lag_feats].astype(float)
# %%
X.index = train_data.index

# %%
X[["ther_area", "Week_day", "main_channel"]] = TargetEncoder().fit_transform(
    X[["ther_area", "Week_day", "main_channel"]].astype(str).fillna("unknown"), y
)

X[["brand", "country"]] = OrdinalEncoder().fit_transform(X[["brand", "country"]])
# %%
from src.utils.preprocessing import calculate_time_features

X = calculate_time_features(X, "date")
# %%
estimator = LGBMRegressor(n_jobs=1, random_state=42)
# %%
# from sklearn.model_selection import cross_val_score

# score = cross_val_score(estimator, X, y, cv=3)
# print(score)

# %%

# %%
from src.utils.validation import (
    train_test_split_temporal,
    initial_train_test_split_temporal,
)

X_tr, X_te, y_tr, y_te = initial_train_test_split_temporal(X, y)
# %%
estimator.fit(X_tr.drop(columns=["date", "monthly"]), y_tr)
# %%
y_pred = estimator.predict(X_tr.drop(columns=["date", "monthly"]))
y_te_pred = estimator.predict(X_te.drop(columns=["date", "monthly"]))

# %%
from src.helper.helper import metric, scale_prediction

X_tr["prediction"] = y_pred
X_te["prediction"] = y_te_pred


X_tr = scale_prediction(X_tr)
X_te = scale_prediction(X_te)
# %%
X_tr["phase"] = y_tr
metric(X_tr)
# %%
X_te["phase"] = y_te
metric(X_te)

# %%

estimator.fit(X.drop(columns=["date", "monthly"]), y)
# %%
y_pred = estimator.predict(X.drop(columns=["date", "monthly"]))
# %%
train_data["prediction"] = y_pred
train_data = scale_prediction(train_data)
metric(train_data)
# %%
X_subm = transform_data(submission_data, categorical_feat)
X_subm[lag_feats] = X_subm[lag_feats].fillna(method="ffill").astype(float)
y_pred_sum = estimator.predict(X_subm.drop(columns=["date", "monthly"]))
# %%
y_pred_sum = np.clip(y_pred_sum, 0, np.inf)
# %%
submission_data["prediction"] = y_pred_sum
submission_data = scale_prediction(submission_data)

# %%
submission_template = pd.read_csv("data/submission_template.csv")
submission_data = submission_data[submission_template.keys()]
# %%
# Save Submission
sub_number = "_lags_neighbour_rolling"
sub_name = "submission/submission{}.csv".format(sub_number)
submission_data.to_csv(sub_name, index=False)

# from sklearn.model_selection import GridSearchCV


# %%
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# %%
X_non_cat = X.drop(
    columns=[
        "ther_area",
        "main_channel",
        "brand",
        "country",
        "hospital_rate",
        "date",
        "monthly",
    ]
)
X_non_cat = X_non_cat.fillna(method="ffill").astype(float).fillna(0)
# %%
X_non_cat.isna().sum().sum()

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_non_cat)
# %%
lasso = Lasso(alpha=0.001)
lasso.fit(X_scaled, y)
# %%
for i in range(len(lasso.coef_)):
    if lasso.coef_[i] != 0:
        print(lasso.coef_[i], X_non_cat.columns[i])
# %%
