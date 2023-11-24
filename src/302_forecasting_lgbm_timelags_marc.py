# %%
import pandas as pd

# %%
train_data = pd.read_parquet("data/201_feateng_train_data.parquet").drop(columns = ['level_0','level_1'])
print(train_data.isna().sum() / len(train_data))

# %%
submission_data = pd.read_parquet("data/201_feateng_test_data.parquet").drop(columns = ['level_0','level_1'])
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
from src.helper.helper import add_date_cols


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

lag_feats = [k for k in X.keys() if 'lag_' in k]
X[lag_feats] = X[lag_feats].fillna(method = 'ffill').astype(float)


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
print(score)
# %%
from src.helper.helper import metric, scale_prediction

estimator.fit(X, y)
y_pred = estimator.predict(X)
# %%
train_data["prediction"] = y_pred
train_data = scale_prediction(train_data)
metric(train_data)
# %%
X_subm = transform_data(submission_data, categorical_feat)
X_subm[lag_feats] = X_subm[lag_feats].fillna(method = 'ffill').astype(float)
y_pred_sum = estimator.predict(X_subm)
#%%
y_pred_sum = np.clip(y_pred_sum,0,np.inf)
# %%
submission_data['prediction'] = y_pred_sum
submission_data = scale_prediction(submission_data)

# %%
submission_template = pd.read_csv("data/submission_template.csv")
submission_data = submission_data[submission_template.keys()]
# %%
# Save Submission
sub_number = '_lags_v2'
sub_name = "submission/submission{}.csv".format(sub_number)
submission_data.to_csv(sub_name, index=False)

# import lightgbm
# from sklearn.model_selection import GridSearchCV

# models = {}
# for code,code_df in train_data.groupby('code'):
#     X = code_df.drop(columns = ['code','brand','country'])
#     estimator = lightgbm()
#     cv = GridSearchCV()

#%%
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
# %%
X_non_cat = X.drop(columns = ["ther_area",
    "main_channel",
    "brand",
    "country",
    "hospital_rate"])
X_non_cat = X_non_cat.fillna(method = 'ffill').astype(float).fillna(0)
#%%
X_non_cat.isna().sum().sum()

#%%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_non_cat)
# %%
lasso = Lasso(alpha = 0.001)
lasso.fit(X_scaled,y)
# %%
print(lasso.coef_,X_non_cat.columns)
# %%
