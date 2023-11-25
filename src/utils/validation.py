# %%
import os

import pandas as pd


def initial_train_test_split_temporal(
    X, y, date_col: str = "date", date_split: str = "2021-01-01"
):
    X_tr = X[X[date_col] < date_split]
    X_te = X[X[date_col] >= date_split]
    y_tr = y[X[date_col] < date_split]
    y_te = y[X[date_col] >= date_split]
    return X_tr, X_te, y_tr, y_te


def train_test_split_temporal(
    X,
    y,
    date_col: str = "date",
    date_split: str = "2021-01-01",
    filter: bool = False,
    also_train: bool = False,
):
    if also_train and filter == False:
        raise ValueError("If also_train is True, filter must be True")

    X["y"] = y
    X["country_brand"] = X["country"].astype(str) + X["brand"].astype(str)

    if filter:
        X_ = X[X.country_brand.isin(unique_test_country_brand())]
    if also_train:
        X_tr = X_[X_[date_col] < date_split].drop(["y", "country_brand"], axis=1)
        y_tr = X_[X_[date_col] < date_split]["y"]
    else:
        X_tr = X[X[date_col] < date_split].drop(["y", "country_brand"], axis=1)
        y_tr = X[X[date_col] < date_split]["y"]
    # If filter remove from test the country_brand that are not in submission_data

    if filter:
        X_te = X_[X_[date_col] >= date_split].drop(["y", "country_brand"], axis=1)
        y_te = X_[X_[date_col] >= date_split]["y"]
    else:
        X_te = X[X[date_col] >= date_split].drop(["y", "country_brand"], axis=1)
        y_te = X[X[date_col] >= date_split]["y"]
    return X_tr, X_te, y_tr, y_te


def unique_test_country_brand():
    # Check if file exists
    if os.path.exists("data/test_country_brand.txt"):
        return pd.read_csv("data/test_country_brand.txt", header=None)[0].values
    else:
        train_data = pd.read_parquet("data/train_data.parquet")
        submission_data = pd.read_parquet("data/submission_data.parquet")

        train_data = train_data[["country", "brand"]]
        train_data["country_brand"] = train_data["country"] + train_data["brand"]
        submission_data = submission_data[["country", "brand"]]
        submission_data["country_brand"] = (
            submission_data["country"] + submission_data["brand"]
        )

        # Remove from train_data the country_brand that are not in submission_data
        train_data = train_data[
            train_data.country_brand.isin(submission_data.country_brand.unique())
        ]

        # Save as .txt file
        vals = submission_data.country_brand.unique()
        with open("data/test_country_brand.txt", "w") as f:
            for val in vals:
                f.write(val + "\n")

            return pd.read_csv("data/test_country_brand.txt", header=None)[0].values


# %%
"""Testing
# Read files
train_data = pd.read_parquet("data/train_data.parquet")

submission_data = pd.read_parquet("data/submission_data.parquet")

train_data["train"] = 1
submission_data["train"] = 0
df = pd.concat([train_data, submission_data], axis=0)
# Feature Engineering
## TODO: Add feature engineering

# %%
# Train Test Split
X_subm = df[df.train == 0]
X = df[df.train == 1].drop(["phase"], axis=1)
y = df[df.train == 1].phase
# Temporal split
X_tr, X_te, y_tr, y_te = train_test_split_temporal(
    X, y, date_col="date", date_split="2021-01-01", 
)
print('Filter=False','AlsoTrain=False')
print('X_tr.shape',X_tr.shape)
print('X_te.shape',X_te.shape)
# Temporal split
X_tr, X_te, y_tr, y_te = train_test_split_temporal(
    X, y, date_col="date", date_split="2021-01-01", filter=True
)
print('Filter=True','AlsoTrain=False')
print('X_tr.shape',X_tr.shape)
print('X_te.shape',X_te.shape)
# Temporal split
X_tr, X_te, y_tr, y_te = train_test_split_temporal(
    X, y, date_col="date", date_split="2021-01-01", filter=True,also_train=True
)
print('Filter=True','AlsoTrain=Train')
print('X_tr.shape',X_tr.shape)
print('X_te.shape',X_te.shape)

# %%
# Should raise an error
X_tr, X_te, y_tr, y_te = train_test_split_temporal(
    X, y, date_col="date", date_split="2021-01-01", filter=False,also_train=True
)
print('Filter=False','AlsoTrain=False')
print('X_tr.shape',X_tr.shape)
print('X_te.shape',X_te.shape)

# %%
"""
