# %%
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.helper.helper import check_assert_sum_1, metric, scale_prediction
from src.model_pipelines.dummy import DummyModelPipeline
from src.model_pipelines.lgbm import LGBMModelPipeline
from src.utils.preprocessing import add_date_cols
from src.utils.validation import initial_train_test_split_temporal

pipelines = {
    "lgbm": LGBMModelPipeline(),
    "dummy": DummyModelPipeline(),
}

# %%

# Load data
data_path = Path("data")
submission_df_raw = pd.read_parquet(data_path / "submission_data.parquet")
train_df = pd.read_parquet(data_path / "train_data.parquet")

all_df = pd.concat([train_df, submission_df_raw])

all_df = all_df.assign(
    main_channel=lambda x: x["main_channel"].astype(str).fillna("unknown"),
    ther_area=lambda x: x["ther_area"].astype(str).fillna("unknown"),
).pipe(add_date_cols)

rolling_df = pd.read_parquet(data_path / "rolling_features_less_aggs.parquet")

all_df = all_df.merge(rolling_df, on=["date", "brand", "country"], how="left")
all_df = all_df.query("date > '2020-01-01'")
df = all_df.query("date < '2022-01-01'")
# add random variable
df["random"] = np.random.random(size=len(df))
submission_df = all_df.query("date >= '2022-01-01'")
# %%
y = df.phase
X_raw = df.drop(columns=["phase"])

# Prepare X_train, X_test, y_train and y_test for ML
X_train_raw, X_test_raw, y_train, y_test = initial_train_test_split_temporal(
    X_raw, y, date_col="date"
)
X_train = X_train_raw.drop(columns=["formatted_date", "date", "monthly", "quarter_wm"])
X_test = X_test_raw.drop(columns=["formatted_date", "date", "monthly", "quarter_wm"])
X = X_raw.drop(columns=["formatted_date", "date", "monthly", "quarter_wm"])
X_subm = submission_df.drop(
    columns=["formatted_date", "date", "monthly", "phase", "quarter_wm"]
)
# %%
# Get model and grid
model_pipeline = pipelines["lgbm"]
model_pipe = model_pipeline.get_pipeline()
pipeline_grid = model_pipeline.get_grid()
fit_kwargs = model_pipeline.get_fit_kwargs(X_train_raw)

# Define cv pipeline
model_cv = GridSearchCV(
    model_pipe,
    param_grid=pipeline_grid,
    cv=3,
    scoring="neg_mean_squared_error",
)
# %%
# train pipeline (use monthly as weights)
model_pipe.fit(X_train, y_train)
# %%
import shap
import matplotlib.pyplot as plt

# Explain model
explainer = shap.TreeExplainer(model_pipe.named_steps["model"])
# %%
# Pipe transform data
X_xai = model_pipe.named_steps["encoder"].transform(X_test.sample(1000))
# %%
shap_values = explainer.shap_values(X_xai)

# %%
# Save image
shap.summary_plot(shap_values, X_xai, plot_type="bar", show=False)
plt.savefig("shap_summary_plot.png")
# %%

# %%
print(f"Best params for {model_pipeline.model_name}: {model_cv.best_params_}")
print(f"CV MSE for {model_pipeline.model_name}: {model_cv.best_score_}")
# %%
X_train_raw["prediction"] = model_cv.predict(X_train).clip(0, None)
mse = mean_squared_error(X_train_raw["prediction"], y_train)
print(f"Train MSE for {model_pipeline.model_name}: {mse}")
X_train_pred = scale_prediction(X_train_raw)
X_train_pred["phase"] = y_train
check_assert_sum_1(X_train_pred)
metric_train = metric(X_train_pred)

print(f"Train metric for {model_pipeline.model_name}: {metric_train}")
# %%
X_test_raw["prediction"] = model_pipe.predict(X_test)
mse = mean_squared_error(X_test_raw["prediction"], y_test)
print(f"Test MSE for : {mse}")
X_test_pred = scale_prediction(X_test_raw)
X_test_pred["phase"] = y_test
check_assert_sum_1(X_test_pred)
metric_test = metric(X_test_pred)

print(f"Test metric for: {metric_test}")
#%%

# Train model with best params
fit_kwargs = model_pipeline.get_fit_kwargs(X_raw)
model_pipe.set_params(**model_cv.best_params_)
model_pipe.fit(X, y, **fit_kwargs)

PATH = Path("data")
submission_df["prediction"] = model_pipe.predict(X_subm).clip(0, None)
submission_df = scale_prediction(submission_df)
check_assert_sum_1(submission_df)
submission = pd.read_csv(PATH / "submission_template.csv")
submission = submission_df[submission.columns]

SAVE_PATH = Path("submissions")
SAVE_PATH.mkdir(exist_ok=True)
submission.to_csv(SAVE_PATH / f"submission_{submission_timestamp}.csv", index=False)

# %%
