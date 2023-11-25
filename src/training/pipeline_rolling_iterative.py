# %%
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from src.helper.helper import check_assert_sum_1, metric, scale_prediction
from src.utils.preprocessing import add_date_cols
from src.utils.validation import initial_train_test_split_temporal
import shap
import matplotlib.pyplot as plt


cols_keep = [
    "bc_phase_mean_1y_wd",
    "bc_phase_mean_4y_dayweek",
    "bc_phase_mean_5y_dayweek",
    "bcm_phase_median_2y_formatted_date",
    "brand",
    "country",
    "date",
    "formatted_date",
    "main_channel",
    "month",
    "monthly",
    "n_nwd_aft",
    "n_weekday_3",
    "phase",
    "quarter_wm",
    "ther_area",
    "wd",
    "wd_left",
    "week",
    "year",
]

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

all_df = all_df.merge(rolling_df, on=["date", "brand", "country"], how="left")#[cols_keep]
#all_df = all_df.query("date > '2020-01-01'")
df = all_df.query("date < '2022-01-01'")

submission_df = all_df.query("date >= '2022-01-01'")[cols_keep]

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
from category_encoders import TargetEncoder
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

# Get model and grid

model1 = Pipeline(
    [
        (
            "encoder",
            TargetEncoder(cols=["brand", "country", "main_channel", "ther_area", "Week_day"]),
        ),
        (
            "model",
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=0, n_estimators=100),
        ),
    ]
)
model2 = Pipeline(
    [
        (
            "encoder",
            TargetEncoder(cols=["brand", "country", "main_channel", "ther_area", "Week_day"]),
        ),
        (
            "model",
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=0, n_estimators=100),
        ),
    ]
)
model3 = Pipeline(
    [
        (
            "encoder",
            TargetEncoder(cols=["brand", "country", "main_channel", "ther_area", "Week_day"]),
        ),
        (
            "model",
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=0, n_estimators=100),
        ),
    ]
)
model4 = Pipeline(
    [
        (
            "encoder",
            TargetEncoder(cols=["brand", "country", "main_channel", "ther_area", "Week_day"]),
        ),
        (
            "model",
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=0, n_estimators=100),
        ),
    ]
)
model5 = Pipeline(
    [
        (
            "encoder",
            TargetEncoder(cols=["brand", "country", "main_channel", "ther_area", "Week_day"]),
        ),
        (
            "model",
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=0, n_estimators=100),
        ),
    ]
)
model6 = Pipeline(
    [
        (
            "encoder",
            TargetEncoder(cols=["brand", "country", "main_channel", "ther_area", "Week_day"]),
        ),
        (
            "model",
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=0, n_estimators=100),
        ),
    ]
)
model7 = Pipeline(
    [
        (
            "encoder",
            TargetEncoder(cols=["brand", "country", "main_channel", "ther_area", "Week_day"]),
        ),
        (
            "model",
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=0, n_estimators=100),
        ),
    ]
)
model8 = Pipeline(
    [
        (
            "encoder",
            TargetEncoder(cols=["brand", "country", "main_channel", "ther_area", "Week_day"]),
        ),
        (
            "model",
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=0, n_estimators=100),
        ),
    ]
)
model9 = Pipeline(
    [
        (
            "encoder",
            TargetEncoder(cols=["brand", "country", "main_channel", "ther_area", "Week_day"]),
        ),
        (
            "model",
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=0, n_estimators=100),
        ),
    ]
)
model10 = Pipeline(
    [
        (
            "encoder",
            TargetEncoder(cols=["brand", "country", "main_channel", "ther_area", "Week_day"]),
        ),
        (
            "model",
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=0, n_estimators=100),
        ),
    ]
)
model11 = Pipeline(
    [
        (
            "encoder",
            TargetEncoder(cols=["brand", "country", "main_channel", "ther_area", "Week_day"]),
        ),
        (
            "model",
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=0, n_estimators=100),
        ),
    ]
)
model12 = Pipeline(
    [
        (
            "encoder",
            TargetEncoder(cols=["brand", "country", "main_channel", "ther_area", "Week_day"]),
        ),
        (
            "model",
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=0, n_estimators=100),
        ),
    ]
)


# %%
# Month 1
X_tr1 = X_train.query("month == 1")
y_tr1 = y_train.loc[X_tr1.index]
model1.fit(X_tr1, y_tr1)

# Preds month 1
X_te1 = X_test.query("month == 1")
y_te1 = y_test.loc[X_te1.index]
preds1 = model1.predict(X_te1)
# Month 2
X_tr2 = X_train.query("month == 2")
y_tr2 = y_train.loc[X_tr2.index]
X_tr2["preds1"] = model1.predict(X_tr2)
X_te2 = X_test.query("month == 2")
y_te2 = y_test.loc[X_te2.index]
model2.fit(X_tr2, y_tr2)

# Preds month 2
X_te2 = X_test.query("month == 2")
y_te2 = y_test.loc[X_te2.index]
X_te2["preds1"] = model1.predict(X_te2)
preds2 = model2.predict(X_te2)

# Month 3
X_tr3 = X_train.query("month == 3")
y_tr3 = y_train.loc[X_tr3.index]
X_tr3["preds1"] = model1.predict(X_tr3)
X_tr3["preds2"] = model2.predict(X_tr3)
X_te3 = X_test.query("month == 3")
y_te3 = y_test.loc[X_te3.index]
model3.fit(X_tr3, y_tr3)

# Preds month 3
X_te3 = X_test.query("month == 3")
y_te3 = y_test.loc[X_te3.index]
X_te3["preds1"] = model1.predict(X_te3)
X_te3["preds2"] = model2.predict(X_te3)
preds3 = model3.predict(X_te3)

# Month 4
X_tr4 = X_train.query("month == 4")
y_tr4 = y_train.loc[X_tr4.index]
X_tr4["preds1"] = model1.predict(X_tr4)
X_tr4["preds2"] = model2.predict(X_tr4)
X_tr4["preds3"] = model3.predict(X_tr4)
X_te4 = X_test.query("month == 4")
y_te4 = y_test.loc[X_te4.index]
model4.fit(X_tr4, y_tr4)

# Preds month 4
X_te4 = X_test.query("month == 4")
y_te4 = y_test.loc[X_te4.index]
X_te4["preds1"] = model1.predict(X_te4)
X_te4["preds2"] = model2.predict(X_te4)
X_te4["preds3"] = model3.predict(X_te4)
preds4 = model4.predict(X_te4)

# Month 5
X_tr5 = X_train.query("month == 5")
y_tr5 = y_train.loc[X_tr5.index]
X_tr5["preds1"] = model1.predict(X_tr5)
X_tr5["preds2"] = model2.predict(X_tr5)
X_tr5["preds3"] = model3.predict(X_tr5)
X_tr5["preds4"] = model4.predict(X_tr5)
X_te5 = X_test.query("month == 5")
y_te5 = y_test.loc[X_te5.index]
model5.fit(X_tr5, y_tr5)

# Preds month 5
X_te5 = X_test.query("month == 5")
y_te5 = y_test.loc[X_te5.index]
X_te5["preds1"] = model1.predict(X_te5)
X_te5["preds2"] = model2.predict(X_te5)
X_te5["preds3"] = model3.predict(X_te5)
X_te5["preds4"] = model4.predict(X_te5)
preds5 = model5.predict(X_te5)

# Month 6
X_tr6 = X_train.query("month == 6")
y_tr6 = y_train.loc[X_tr6.index]
X_tr6["preds1"] = model1.predict(X_tr6)
X_tr6["preds2"] = model2.predict(X_tr6)
X_tr6["preds3"] = model3.predict(X_tr6)
X_tr6["preds4"] = model4.predict(X_tr6)
X_tr6["preds5"] = model5.predict(X_tr6)
X_te6 = X_test.query("month == 6")
y_te6 = y_test.loc[X_te6.index]
model6.fit(X_tr6, y_tr6)

# Preds month 6
X_te6 = X_test.query("month == 6")
y_te6 = y_test.loc[X_te6.index]
X_te6["preds1"] = model1.predict(X_te6)
X_te6["preds2"] = model2.predict(X_te6)
X_te6["preds3"] = model3.predict(X_te6)
X_te6["preds4"] = model4.predict(X_te6)
X_te6["preds5"] = model5.predict(X_te6)
preds6 = model6.predict(X_te6)

# Month 7
X_tr7 = X_train.query("month == 7")
y_tr7 = y_train.loc[X_tr7.index]
X_tr7["preds1"] = model1.predict(X_tr7)
X_tr7["preds2"] = model2.predict(X_tr7)
X_tr7["preds3"] = model3.predict(X_tr7)
X_tr7["preds4"] = model4.predict(X_tr7)
X_tr7["preds5"] = model5.predict(X_tr7)
X_tr7["preds6"] = model6.predict(X_tr7)
X_te7 = X_test.query("month == 7")
y_te7 = y_test.loc[X_te7.index]
model7.fit(X_tr7, y_tr7)

# Preds month 7
X_te7 = X_test.query("month == 7")
y_te7 = y_test.loc[X_te7.index]
X_te7["preds1"] = model1.predict(X_te7)
X_te7["preds2"] = model2.predict(X_te7)
X_te7["preds3"] = model3.predict(X_te7)
X_te7["preds4"] = model4.predict(X_te7)
X_te7["preds5"] = model5.predict(X_te7)
X_te7["preds6"] = model6.predict(X_te7)
preds7 = model7.predict(X_te7)

# Month 8
X_tr8 = X_train.query("month == 8")
y_tr8 = y_train.loc[X_tr8.index]
X_tr8["preds1"] = model1.predict(X_tr8)
X_tr8["preds2"] = model2.predict(X_tr8)
X_tr8["preds3"] = model3.predict(X_tr8)
X_tr8["preds4"] = model4.predict(X_tr8)
X_tr8["preds5"] = model5.predict(X_tr8)
X_tr8["preds6"] = model6.predict(X_tr8)
X_tr8["preds7"] = model7.predict(X_tr8)
X_te8 = X_test.query("month == 8")
y_te8 = y_test.loc[X_te8.index]
model8.fit(X_tr8, y_tr8)

# Preds month 8
X_te8 = X_test.query("month == 8")
y_te8 = y_test.loc[X_te8.index]
X_te8["preds1"] = model1.predict(X_te8)
X_te8["preds2"] = model2.predict(X_te8)
X_te8["preds3"] = model3.predict(X_te8)
X_te8["preds4"] = model4.predict(X_te8)
X_te8["preds5"] = model5.predict(X_te8)
X_te8["preds6"] = model6.predict(X_te8)
X_te8["preds7"] = model7.predict(X_te8)
preds8 = model8.predict(X_te8)

# Month 9
X_tr9 = X_train.query("month == 9")
y_tr9 = y_train.loc[X_tr9.index]
X_tr9["preds1"] = model1.predict(X_tr9)
X_tr9["preds2"] = model2.predict(X_tr9)
X_tr9["preds3"] = model3.predict(X_tr9)
X_tr9["preds4"] = model4.predict(X_tr9)
X_tr9["preds5"] = model5.predict(X_tr9)
X_tr9["preds6"] = model6.predict(X_tr9)
X_tr9["preds7"] = model7.predict(X_tr9)
X_tr9["preds8"] = model8.predict(X_tr9)
X_te9 = X_test.query("month == 9")
y_te9 = y_test.loc[X_te9.index]
model9.fit(X_tr9, y_tr9)

# Preds month 9
X_te9 = X_test.query("month == 9")
y_te9 = y_test.loc[X_te9.index]
X_te9["preds1"] = model1.predict(X_te9)
X_te9["preds2"] = model2.predict(X_te9)
X_te9["preds3"] = model3.predict(X_te9)
X_te9["preds4"] = model4.predict(X_te9)
X_te9["preds5"] = model5.predict(X_te9)
X_te9["preds6"] = model6.predict(X_te9)
X_te9["preds7"] = model7.predict(X_te9)
X_te9["preds8"] = model8.predict(X_te9)
preds9 = model9.predict(X_te9)

# Month 10
X_tr10 = X_train.query("month == 10")
y_tr10 = y_train.loc[X_tr10.index]
X_tr10["preds1"] = model1.predict(X_tr10)
X_tr10["preds2"] = model2.predict(X_tr10)
X_tr10["preds3"] = model3.predict(X_tr10)
X_tr10["preds4"] = model4.predict(X_tr10)
X_tr10["preds5"] = model5.predict(X_tr10)
X_tr10["preds6"] = model6.predict(X_tr10)
X_tr10["preds7"] = model7.predict(X_tr10)
X_tr10["preds8"] = model8.predict(X_tr10)
X_tr10["preds9"] = model9.predict(X_tr10)
X_te10 = X_test.query("month == 10")
y_te10 = y_test.loc[X_te10.index]
model10.fit(X_tr10, y_tr10)

# Preds month 10
X_te10 = X_test.query("month == 10")
y_te10 = y_test.loc[X_te10.index]
X_te10["preds1"] = model1.predict(X_te10)
X_te10["preds2"] = model2.predict(X_te10)
X_te10["preds3"] = model3.predict(X_te10)
X_te10["preds4"] = model4.predict(X_te10)
X_te10["preds5"] = model5.predict(X_te10)
X_te10["preds6"] = model6.predict(X_te10)
X_te10["preds7"] = model7.predict(X_te10)
X_te10["preds8"] = model8.predict(X_te10)
X_te10["preds9"] = model9.predict(X_te10)
preds10 = model10.predict(X_te10)

# Month 11
X_tr11 = X_train.query("month == 11")
y_tr11 = y_train.loc[X_tr11.index]
X_tr11["preds1"] = model1.predict(X_tr11)
X_tr11["preds2"] = model2.predict(X_tr11)
X_tr11["preds3"] = model3.predict(X_tr11)
X_tr11["preds4"] = model4.predict(X_tr11)
X_tr11["preds5"] = model5.predict(X_tr11)
X_tr11["preds6"] = model6.predict(X_tr11)
X_tr11["preds7"] = model7.predict(X_tr11)
X_tr11["preds8"] = model8.predict(X_tr11)
X_tr11["preds9"] = model9.predict(X_tr11)
X_tr11["preds10"] = model10.predict(X_tr11)
X_te11 = X_test.query("month == 11")
y_te11 = y_test.loc[X_te11.index]
model11.fit(X_tr11, y_tr11)

# Preds month 11
X_te11 = X_test.query("month == 11")
y_te11 = y_test.loc[X_te11.index]
X_te11["preds1"] = model1.predict(X_te11)
X_te11["preds2"] = model2.predict(X_te11)
X_te11["preds3"] = model3.predict(X_te11)
X_te11["preds4"] = model4.predict(X_te11)
X_te11["preds5"] = model5.predict(X_te11)
X_te11["preds6"] = model6.predict(X_te11)
X_te11["preds7"] = model7.predict(X_te11)
X_te11["preds8"] = model8.predict(X_te11)
X_te11["preds9"] = model9.predict(X_te11)
X_te11["preds10"] = model10.predict(X_te11)
preds11 = model11.predict(X_te11)

# Month 12
X_tr12 = X_train.query("month == 12")
y_tr12 = y_train.loc[X_tr12.index]
X_tr12["preds1"] = model1.predict(X_tr12)
X_tr12["preds2"] = model2.predict(X_tr12)
X_tr12["preds3"] = model3.predict(X_tr12)
X_tr12["preds4"] = model4.predict(X_tr12)
X_tr12["preds5"] = model5.predict(X_tr12)
X_tr12["preds6"] = model6.predict(X_tr12)
X_tr12["preds7"] = model7.predict(X_tr12)
X_tr12["preds8"] = model8.predict(X_tr12)
X_tr12["preds9"] = model9.predict(X_tr12)
X_tr12["preds10"] = model10.predict(X_tr12)
X_tr12["preds11"] = model11.predict(X_tr12)
X_te12 = X_test.query("month == 12")
y_te12 = y_test.loc[X_te12.index]
model12.fit(X_tr12, y_tr12)

# Preds month 12
X_te12 = X_test.query("month == 12")
y_te12 = y_test.loc[X_te12.index]
X_te12["preds1"] = model1.predict(X_te12)
X_te12["preds2"] = model2.predict(X_te12)
X_te12["preds3"] = model3.predict(X_te12)
X_te12["preds4"] = model4.predict(X_te12)
X_te12["preds5"] = model5.predict(X_te12)
X_te12["preds6"] = model6.predict(X_te12)
X_te12["preds7"] = model7.predict(X_te12)
X_te12["preds8"] = model8.predict(X_te12)
X_te12["preds9"] = model9.predict(X_te12)
X_te12["preds10"] = model10.predict(X_te12)
X_te12["preds11"] = model11.predict(X_te12)
preds12 = model12.predict(X_te12)
# %%
# Store predictions in X_test_raw
X_test_raw.loc[X_te1.index, "prediction"] = preds1
X_test_raw.loc[X_te2.index, "prediction"] = preds2
X_test_raw.loc[X_te3.index, "prediction"] = preds3
X_test_raw.loc[X_te4.index, "prediction"] = preds4
X_test_raw.loc[X_te5.index, "prediction"] = preds5
X_test_raw.loc[X_te6.index, "prediction"] = preds6
X_test_raw.loc[X_te7.index, "prediction"] = preds7
X_test_raw.loc[X_te8.index, "prediction"] = preds8
X_test_raw.loc[X_te9.index, "prediction"] = preds9
X_test_raw.loc[X_te10.index, "prediction"] = preds10
X_test_raw.loc[X_te11.index, "prediction"] = preds11
X_test_raw.loc[X_te12.index, "prediction"] = preds12

# %%
# Explain model
explainer = shap.TreeExplainer(model12.named_steps["model"])
# Pipe transform data
X_xai = model12.named_steps["encoder"].transform(X_te12.sample(1000))
shap_values = explainer.shap_values(X_xai)
# Save image
shap.summary_plot(shap_values, X_xai, plot_type="bar", show=False)
plt.savefig(f"shap_summary_plot{3}.png")

# %%
# X_test_raw["prediction"] = preds
mse = mean_squared_error(X_test_raw["prediction"], y_test)
print(f"Test MSE  {mse}")
X_test_pred = scale_prediction(X_test_raw)
X_test_pred["phase"] = y_test
check_assert_sum_1(X_test_pred)
metric_test = metric(X_test_pred)

print(f"Test metric : {metric_test}")

# %%
