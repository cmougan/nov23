# %%
import pandas as pd
from helper.helper import metric, add_date_cols, scale_prediction, check_assert_sum_1
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# Read files
train_data = pd.read_parquet("data/train_data.parquet")
## Transform phase based on monthly sales
train_data["phase"] = train_data["phase"] * train_data["monthly"]

# Load submission data
submission_data = pd.read_parquet("data/submission_data.parquet")
# %%
## Get numeric data
train_data_numeric = train_data.select_dtypes(include=["float64", "int64"])
for col in train_data_numeric.columns:
    try:
        pval = ks_2samp(train_data_numeric[col], submission_data[col]).pvalue
        plt.figure()
        plt.title(col + " pval: " + str(pval))
        sns.kdeplot(train_data[col], label="train")
        sns.kdeplot(submission_data[col], label="test")
        plt.legend()
        plt.show()

    except Exception as e:
        print(e)
        print("error", col)

# %%


# %%
