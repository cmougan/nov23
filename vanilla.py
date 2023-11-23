# %%
import pandas as pd
from helper.helper import metric, add_date_cols, scale_prediction, check_assert_sum_1
from sklearn.model_selection import train_test_split

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

# %%
# Train Test Split
X_subm = df[df.train == 0]
X = df[df.train == 1].drop(["phase"], axis=1)
y = df[df.train == 1].phase

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
# %%

# %%
# Model
from sklearn.dummy import DummyRegressor

model = DummyRegressor(strategy="mean")
model.fit(X_tr, y_tr)


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
submission = pd.read_csv("data/submission_template.csv")
submission = add_date_cols(submission)

# Merge submission template with predictions
submission.merge(
    X_subm[["country", "brand", "month", "prediction"]],
    on=["country", "brand", "month"],
    how="left",
)
submission = submission[["country", "brand", "date", "prediction"]]
# %%
# Save Submission
sub_number = 1
sub_name = "submissions/submission{}.csv".format(sub_number)
submission.to_csv(sub_name, index=False)
