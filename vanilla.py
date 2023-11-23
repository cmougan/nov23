# %%
import pandas as pd
from helper.helper import metric
from sklearn.model_selection import train_test_split

# %%
# Read parquet file
train_data = pd.read_parquet("data/train_data.parquet")

# %%
# Train Test Split
y = train_data["phase"]
X = train_data.drop(["phase"], axis=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# Dummy Regressor
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy="mean")
dummy.fit(X, y)

train_data["prediction"] = dummy.predict(X)
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
