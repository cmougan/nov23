# %%
import os
import pandas as pd


def train_test_split_temporal(
    X, y, date_col: str = "date", date_split: str = "2019-01-01", filter: bool = False
):
    X_tr = X[X[date_col] < date_split]
    y_tr = y[X[date_col] < date_split]
    # If filter remove from test the country_brand that are not in submission_data
    if filter:
        X["y"] = y
        X["country_brand"] = X["country"] + X["brand"]
        X = X[X.country_brand.isin(unique_test_country_brand())]

    X_te = X[X[date_col] >= date_split].drop(["y"], axis=1)
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


# Train Test Split
X_subm = df[df.train == 0]
X = df[df.train == 1].drop(["phase"], axis=1)
y = df[df.train == 1].phase
# Temporal split
X_tr, X_te, y_tr, y_te = train_test_split_temporal(
    X, y, date_col="date", date_split="2021-01-01", filter=True
)


# %%
X['country_brand'] = X['country'] + X['brand']
# %%
X = X[X.country_brand.isin(unique_test_country_brand())]
# %%
y_te = y[X['date'] >= '2019-01-01']
# %%
unique_test_country_brand()
# %%
"""
