# %%
import numpy as np
import pandas as pd


data = pd.concat(
    [pd.read_csv("Eda_forecasting.csv"), pd.read_csv("Eda_forecasting_val.csv")]
)
data["date"] = pd.to_datetime(data["date"])
# %%
country = [s.split("_")[1] for s in data["code"]]
brand = [s.split("_")[0] for s in data["code"]]
data["country"] = country
data["brand"] = brand


# %%
def get_metrics(x):
    outliers = x.iloc[np.where(x["phase"] > 0.05)[0]]
    out_per = len(outliers) / len(x)
    outliers_mae = np.sum(abs(outliers["phase"] - outliers["prediction"])) / np.sum(
        abs(x["phase"] - x["prediction"])
    )
    mae = np.mean(abs(x["phase"] - x["prediction"]))
    coverage = (x["phase"] > x["prediction"]).sum() / len(x) * 100
    mape = np.sqrt(np.mean((x["phase"] - x["prediction"]) ** 2))
    l = (x.date.max() - x.date.min()).days
    zeros = 1 - len(x) / l

    return pd.Series(
        {
            "mae": mae,
            "coverage": coverage,
            "mape": mape,
            "quarters": l / 120,
            "zeros": zeros,
            "outliers_percentage": out_per,
            "outliers_contr": outliers_mae * 100,
        }
    )


errors_per_code = data.groupby(["country", "brand", "code"], as_index=False).apply(
    lambda x: get_metrics(x)
)

# %%
import seaborn as sns
import matplotlib.pylab as plt

# %%
sns.kdeplot(errors_per_code.mae)
plt.show()

sns.kdeplot(errors_per_code.mape)
plt.show()

sns.kdeplot(errors_per_code.coverage)
plt.show()

# %%
from matplotlib import cm

mae_col = np.clip(
    (errors_per_code.mae - np.min(errors_per_code.mae)) * 4 * 255, 0, 20
).astype(int)
colors = cm.get_cmap("Reds", 20)
plt.scatter(errors_per_code.quarters, errors_per_code.zeros, c=colors(mae_col))
plt.xlabel("Quarters of data since 2017")
plt.ylabel("% missing days")
plt.title("Error (in red) for each country-brand")
plt.show()
# %%
plt.scatter(errors_per_code.outliers_contr, errors_per_code.mae, c=colors(mae_col))
plt.xlabel("outliers contribution to error")
plt.ylabel("Error")
plt.title("Outliers contribution to error")
# %%
errors_per_code = errors_per_code.sort_values(by="mae")
# %%
for i in range(3):
    code = errors_per_code.iloc[i]["code"]
    code_data = data.query(f'code=="{code}"')
    plt.scatter(code_data["date"], code_data["phase"], alpha=0.6, label=code)
plt.title("Best Country-brand to predict")
plt.ylim(-0.01, 0.4)
plt.xlim(pd.to_datetime("01-01-2017"), pd.to_datetime("01-01-2022"))
plt.legend(bbox_to_anchor=(1.01, 1))
plt.show()

# %%

for i in range(3):
    code = errors_per_code.iloc[-i]["code"]
    code_data = data.query(f'code=="{code}"')
    plt.scatter(code_data["date"], code_data["phase"], alpha=0.6, label=code)
plt.title("Worst Country-brand to predict")
plt.ylim(-0.01, 0.4)
plt.xlim(pd.to_datetime("01-01-2017"), pd.to_datetime("01-01-2022"))
plt.legend(bbox_to_anchor=(1.01, 1))
plt.show()

# %%

errors_per_country = errors_per_code.groupby("country", as_index=False).apply(
    lambda x: x[["mae", "mape"]].mean()
)
# %%
errors_per_country["mae"] = (
    errors_per_country["mae"] / np.min(errors_per_country["mae"])
) - 1
errors_per_country = errors_per_country.sort_values(by="mae")

# %%
plt.figure(figsize=(10, 8))
plt.bar(errors_per_country.country, errors_per_country.mae * 100)
plt.ylabel("Error increase vs best country")
plt.title("Normlized error per country")
plt.xticks(rotation=45)
# %%
errors_per_brand = errors_per_code.groupby("brand", as_index=False).apply(
    lambda x: x[["mae", "mape"]].mean()
)
# %%
errors_per_brand["mae"] = (
    errors_per_brand["mae"] / np.min(errors_per_brand["mae"])
) - 1
errors_per_brand = errors_per_brand.sort_values(by="mae")

# %%
plt.figure(figsize=(10, 8))
plt.bar(errors_per_brand.brand.iloc[-10:], errors_per_brand.mae.iloc[-10:] * 100)
plt.ylabel("Error increase vs best brand")
plt.title("Normlized error per 10 WORST brand")
plt.xticks(rotation=45)
# %%
plt.figure(figsize=(10, 8))
plt.bar(np.arange(len(errors_per_brand)), errors_per_brand.mae * 100)
plt.ylabel("Error increase vs best brand")
plt.xlabel("brand_num")
plt.title("Normlized error per brand")
plt.xticks(rotation=45)

# %%

from statsmodels.tsa.stattools import acf
from scipy.signal import medfilt

acf_data = data.groupby("date").phase.mean()
plt.plot(np.array(acf_data)[-30:], label="freq = 1D (Current)")
acf_data_smooth = medfilt(acf_data, 3)
plt.plot(acf_data_smooth[-30:], label="freq = 2D")
acf_data_smooth = medfilt(acf_data, 7)
plt.plot(acf_data_smooth[-30:], label="freq = 1W")
plt.ylabel("Phase")
plt.axis("off")
plt.legend(bbox_to_anchor=(1.01, 1))
plt.title("Phase evolution vs forecasting frequency")
# %%
acf_res = acf(acf_data, nlags=60)
# %%
plt.bar(np.arange(len(acf_res)), acf_res)
plt.xlim(
    1,
)
# %%

from statsmodels.tsa.seasonal import seasonal_decompose

acf_data = data.groupby("date").phase.mean()
acf_data.sort_index(inplace=True)
# %%
result = seasonal_decompose(acf_data.iloc[-120:], model="additive", period=30)
result.plot()
# %%
