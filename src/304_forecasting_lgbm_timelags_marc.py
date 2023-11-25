# %%
import pandas as pd

# %%
train_data = pd.read_parquet("data/204_feateng_train_data.parquet").drop(
    columns=["level_0", "level_1"]
)
print(train_data.isna().sum() / len(train_data))

# %%
submission_data = pd.read_parquet("data/204_feateng_test_data.parquet").drop(
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
estimator = LGBMRegressor(categorical_feature=categorical_feat)
# %%
# from sklearn.model_selection import cross_val_score

# score = cross_val_score(estimator, X, y, cv=3)
# print(score)

# %%
from src.utils.validation import train_test_split_temporal

X_tr, X_te, y_tr, y_te = train_test_split_temporal(X, y)
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
from src.utils.preprocessing import add_basic_lag_features_week
from tqdm import tqdm
import warnings

warnings.simplefilter(action="ignore")
tqdm.pandas()


def rolling_prediction(X, X_tr, y_tr, estimator, use_last_index=False):
    total_data_train = X_tr.copy()
    total_data_train["phase"] = y_tr
    total_data_train["code"] = [
        "_".join([str(brand), str(country)])
        for brand, country in zip(
            total_data_train["brand"], total_data_train["country"]
        )
    ]
    train_end = str(total_data_train.date.max().strftime("%Y-%m-%d"))
    total_data_test = X.assign(phase=np.nan)
    total_data_test["code"] = [
        "_".join([str(brand), str(country)])
        for brand, country in zip(total_data_test["brand"], total_data_test["country"])
    ]
    codes_test |= total_data_test.code.unique().tolist()
    min_date = (total_data_test.date.min() - pd.Timedelta(365, "D")).strftime(
        "%Y-%m-%d"
    )

    print(total_data_train.shape)
    total_data_train = total_data_train.query(f'date>="{min_date}"')
    print(total_data_train.shape)
    total_data = pd.concat([total_data_train, total_data_test])
    total_data = total_data.query("code in @codes_test")

    total_data_with_lags = (
        total_data.groupby("code", as_index=False).progress_apply(
            lambda x: add_basic_lag_features_week(x, 20, 5, 5)
        )
    ).reset_index()

    X = transform_data(
        total_data_with_lags.query(f'date>"{train_end}"'), categorical_feat
    )

    if use_last_index:
        lag_feats_day = [k for k in X.keys() if "lag_" in k and "_day" in k]
        X[lag_feats_day].to_csv("debug.csv")
        last_index = min(np.where(X[lag_feats_day].isna().sum(axis=1) > 15)[0])
    else:
        last_index = len(X) + 1

    for i in tqdm(range(3)):
        print(i, last_index)
        pred = estimator.predict(
            X.drop(columns=["date", "level_0", "level_1"]).iloc[:last_index]
        )
        diff = np.sum(np.abs(total_data_test.phase.iloc[:last_index] - pred))
        pred_series = pd.Series(pred, index=total_data_test.iloc[:last_index].index)
        total_data_test.phase.iloc[:last_index].fillna(pred_series, inplace=True)

        print(i, diff, X[lag_feats].isna().sum().sum(), X.shape)

        total_data = pd.concat([total_data_train, total_data_test])
        total_data = total_data.query("code in @codes_test")
        total_data_with_lags = (
            total_data.groupby("code", as_index=False).progress_apply(
                lambda x: add_basic_lag_features_week(x, 20, 5, 5)
            )
        ).reset_index()

        X = transform_data(
            total_data_with_lags.query(f'date>"{train_end}"'), categorical_feat
        )
        X[lag_feats] = X[lag_feats].fillna(method="ffill").astype(float)
        if use_last_index:
            last_index = min(np.where(X[lag_feats].isna().sum(axis=1) > 20)[0])
        else:
            last_index = len(X) + 1

    return pred


y_pred_rec = rolling_prediction(
    X_te.drop(columns=["prediction", "sum_pred", "monthly"]),
    X_tr.drop(columns=["prediction", "sum_pred", "monthly"]),
    y_tr,
    estimator,
    use_last_index=True,
)
# %%

X_te["prediction"] = y_pred_rec
X_te = scale_prediction(X_te)
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
y_pred_sum = rolling_prediction(
    X_subm.drop(columns=["monthly"]),
    X.drop(columns=["prediction", "sum_pred", "monthly"]),
    y,
    estimator,
)
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
sub_number = "_lags_neighbour"
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
argsort = np.argsort(abs(lasso.coef_))[::-1]
# %%
for i in argsort:
    if lasso.coef_[i] != 0:
        print(lasso.coef_[i], X_non_cat.columns[i])
# %%
