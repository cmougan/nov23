# %%
from pathlib import Path

import pandas as pd

from src.helper.helper import check_assert_sum_1

submission_path = Path("submissions")


files = {
    "submission_2023-11-25_21-16-40_ordinal-500-trees-quarter-wm.csv":  0.00995544,
    "submission_2023-11-25_21-05-52_ordinal-more-trees.csv":  0.00996189,
    "submission_2023-11-25_14-11-27_fix-aggs.csv": 0.01001395,
    "submission_lags_v3.csv": 0.0102128,
    "submission_lags_neighbour.csv":  0.01013355,
}

dfs = {}


for file, score in files.items():
    dfs[file] = pd.read_csv(submission_path / file)
    dfs[file]["error"] = score

# %%
ensemble_df = pd.concat(dfs.values())
ensemble_df["inverse_error"] = 0.01 + (ensemble_df["error"].max() / ensemble_df["error"]) - 1

total_sum_weights = ensemble_df["inverse_error"].unique().sum()
ensemble_df["inverse_error"] = ensemble_df["inverse_error"] / total_sum_weights

ensemble_df["inverse_error"].value_counts()


# %%
ensemble_df["prediction_score"] = ensemble_df["prediction"] * ensemble_df["inverse_error"]

ensemble_df = ensemble_df.groupby(["country", "brand", "date"], as_index=False).agg(
    prediction=("prediction_score", "sum"),
    inverse_error=("inverse_error", "sum"),
)


# %%
ensemble_df.assign(
    month=lambda x: pd.to_datetime(x["date"]).dt.month,
).groupby(["country", "brand", "month"], as_index=False).agg(
    prediction_score=("prediction", "sum"),
    inverse_error=("inverse_error", "sum"),
).describe()

# %%
ensemble_df.drop(columns=["inverse_error"]).to_csv(submission_path / "submission_ensemble_101.csv", index=False)
# %%
