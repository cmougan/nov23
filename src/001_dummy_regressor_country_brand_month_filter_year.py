# %%
import pandas as pd
from src.helper.helper import metric, check_assert_sum_1
from sklearn.model_selection import train_test_split

# %%
# Read parquet file
train_data = pd.read_parquet("data/train_data.parquet").query('date>"01-01-2019"')
# %%
## Transform Phase based on monthly sales
train_data["phase"] = train_data["phase"] * train_data["monthly"]


# %%
prediction_keys_map = (
    train_data.groupby(["country", "brand", "dayweek", "month"], as_index=False)
    .phase.mean()
    .rename(columns={"phase": "prediction"})
)

prediction_backup_map = (
    train_data.groupby(["country", "brand", "dayweek"], as_index=False)
    .phase.mean()
    .rename(columns={"phase": "prediction_backup"})
)


# %%
def get_dummy_prediction(df, prediction_keys_map, prediction_backup):
    df = add_date_cols(df)

    # add predictions to dataset
    df = df.merge(
        prediction_keys_map, on=["country", "brand", "dayweek", "month"], how="left"
    ).merge(prediction_backup, on=["country", "brand", "dayweek"])

    df.prediction.fillna(df.prediction_backup, inplace=True)
    print(df.prediction.isna().sum())
    assert df.prediction.isna().sum() == 0

    # Calcular los factores de normalizacion
    df_norm_factor = (
        df.groupby(["year", "month", "brand", "country"], as_index=False)
        .prediction.sum()
        .rename(columns={"prediction": "norm_factor"})
    )
    df = df.merge(df_norm_factor, on=["year", "month", "brand", "country"])

    # normalize
    df["prediction"] /= df["norm_factor"]
    return df


# %%

train_data_with_prediction = get_dummy_prediction(
    train_data, prediction_keys_map, prediction_backup_map
)
# %%
# Check performance
print("Performance train:", metric(train_data_with_prediction))


# %%
# Prepare submission
submission_data = pd.read_parquet("data/submission_data.parquet")
submission = pd.read_csv("data/submission_template.csv")
# %%
submission_data_with_prediction = get_dummy_prediction(
    submission_data, prediction_keys_map, prediction_backup_map
)[submission.keys()]

# %%
check_assert_sum_1(submission_data_with_prediction)
# %%
# Save Submission
sub_number = 5
sub_name = "submission/submission{}.csv".format(sub_number)
submission_data_with_prediction.to_csv(sub_name, index=False)

# %%
