# %%
import pandas as pd
from helper.helper import metric

# %%
# Read parquet file
train_data = pd.read_parquet("data/train_data.parquet")

# %%
train_data["prediction"] = 0
# %%
# Check performance
print("Performance:", metric(train_data))


# %%
# Prepare submission
submission_data = pd.read_parquet("data/submission_data.parquet")
submission = pd.read_csv("data/submission_template.csv")
# %%
# Save Submission
sub_number = 1
sub_name = "submission/submission{}.csv".format(sub_number)
submission.to_csv(sub_name, index=False)

# %%
