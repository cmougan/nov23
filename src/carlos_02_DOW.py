# %%
import pandas as pd
from helper.helper import metric, add_date_cols, scale_prediction, check_assert_sum_1
from utils.validation import train_test_split_temporal
import warnings

warnings.filterwarnings("ignore")
# %%
# Read files
train_data = pd.read_parquet("data/train_data.parquet")
## Transform phase based on monthly sales
train_data["phase"] = train_data["phase"] * train_data["monthly"]

# Load submission data
submission_data = pd.read_parquet("data/submission_data.parquet")
## Merge train and submission data
# TODO: Discuss if we want to follow this engineering approach
train_data["train"] = 1
submission_data["train"] = 0
df = pd.concat([train_data, submission_data], axis=0)
# %%
# Feature Engineering
## TODO: Add feature engineering

# Add date colummns
df = add_date_cols(df)

# Concatenate brand country and month
df["brand_country_month"] = (
    df["brand"] + "_" + df["country"] + "_" + df["month"].astype(str)
)

# Map dayweek with aggregated phase sum of groupby of country brand month
df["dayweek"] = df["dayweek"].astype(str)
df["dayweek"] = df["dayweek"].map(
    df.groupby(["country", "brand", "month"])["phase"].sum()
)


# %%
# Train Test Split
X_subm = df[df.train == 0]
X = df[df.train == 1].drop(["phase"], axis=1)
y = df[df.train == 1].phase
# Temporal split
X_tr, X_te, y_tr, y_te = train_test_split_temporal(
    X, y, date_col="date", date_split="2021-01-01"
)

# %%
# Model
from sklearn.pipeline import Pipeline
from category_encoders import OneHotEncoder, TargetEncoder
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from utils.transformer import DropCols, GetNumerical
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor

model = Pipeline(
    [
        ("drop_cols", DropCols(cols=["date", "train", "year", "sum_pred", "phase"])),
        (
            "ohe",
            TargetEncoder(cols=["country", "brand", "month", "brand_country_month"]),
        ),
        ("get_numerical", GetNumerical()),  # TODO: Remove this
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        # ("model", XGBRegressor(max_depth=5, n_estimators=20, n_jobs=-1)),
        ("model", Lasso(alpha=0.1)),
    ]
)

model.fit(X_tr, y_tr)
# %%
# Is the model learning?
# If linear regression, we can check the coefficients
if isinstance(model.named_steps["model"], Lasso):
    print(model.named_steps["model"].coef_)
# If tree based, we can check the feature importance
else:
    print(model.named_steps["model"].feature_importances_)
# %%
# Check performance
## Train
X_tr["prediction"] = model.predict(X_tr)
X_tr = scale_prediction(X_tr)
X_tr["phase"] = y_tr
print("Train Performance:", metric(X_tr))

## Test
X_te["prediction"] = model.predict(X_te)
X_te = scale_prediction(X_te)
X_te["phase"] = y_te
print("Test Performance:", metric(X_te))
# %%
# Assert phase sums 1 for each year-month-brand-country
check_assert_sum_1(X_tr)
check_assert_sum_1(X_te)
# %%
## PREPARE SUBMISSION
# %%
# Train in full pipeline
model.fit(X, y)
# Scale predictions
X_subm["prediction"] = model.predict(X_subm)
X_subm = scale_prediction(X_subm)

# %%
# Prepare file
submission = pd.read_csv("data/submission_template.csv").drop(["prediction"], axis=1)
submission = add_date_cols(submission)  ## TODO does this work?
# %%
# Merge submission template with predictions on "country", "brand", "date"
submission = submission.merge(
    X_subm[["country", "brand", "date", "prediction"]], on=["country", "brand", "date"]
)
submission = submission[["country", "brand", "date", "prediction"]]
# %%
# Check if submission has missing values
print(submission.shape)
submission.isna().sum()
# %%
# Are there negative values?
submission[submission.prediction < 0]
# %%
# Save Submission
sub_number = "xgb_01"
sub_name = "submissions/submission{}.csv".format(sub_number)
submission.to_csv(sub_name, index=False)

# %%
