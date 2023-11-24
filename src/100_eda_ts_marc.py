# -*- coding: utf-8 -*-
"""

"""
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

# %%
train_data = pd.read_parquet("data/train_data.parquet")
print(train_data.isna().sum() / len(train_data))

# %%
submission_data = pd.read_parquet("data/submission_data.parquet")
print(submission_data.isna().sum() / len(train_data))

# %%
split_feats = ["brand", "country"]
train_data["code"] = [
    "_".join([str(brand), str(country)])
    for brand, country in zip(train_data["brand"], train_data["country"])
]


print(f'Unique brands: {train_data["brand"].nunique()}')
print(f'Unique countries: {train_data["country"].nunique()}')
print(f'Unique tuples: {train_data["code"].nunique()}')

# %%

submission_data["code"] = [
    "_".join([str(brand), str(country)])
    for brand, country in zip(submission_data["brand"], submission_data["country"])
]


print(f'Unique brands in submission: {submission_data["brand"].nunique()}')
print(f'Unique countries in submission: {submission_data["country"].nunique()}')
print(f'Unique tuples in submission: {submission_data["code"].nunique()}')
# %%
# Numero de brands per country
n_brands_per_country = train_data.groupby("country", as_index=False).brand.nunique()
plt.hist(n_brands_per_country.brand)
plt.show()

# %%
# number of info per day
train_data["date"] = pd.to_datetime(train_data["date"])
daily_info = train_data.groupby(
    "date", as_index=False
).size()  # not  all days contain all the info.
plt.scatter(daily_info.date, daily_info["size"])  #
plt.show()


# %%
# Evolucion general de las ventas
monthly_accum_sales = train_data.groupby(
    "date", as_index=False
).monthly.mean()  # not sum since not all days contain all the info.
plt.scatter(monthly_accum_sales.date, monthly_accum_sales.monthly)  #

# %%

from helper.helper import add_date_cols
from matplotlib import cm

train_data = add_date_cols(train_data)
yrs = train_data["year"].unique()
colors = cm.get_cmap("viridis", len(yrs))
year_to_col = {y: colors(y - min(yrs)) for y in yrs}
evolution_phase_per_year = train_data.groupby(
    ["year", "wd"], as_index=False
).phase.mean()
for year, yr_phase in evolution_phase_per_year.groupby("year"):
    plt.scatter(
        yr_phase.wd, yr_phase.phase, c=[year_to_col[year]] * len(yr_phase), label=year
    )

plt.legend()
plt.legend(bbox_to_anchor=(1.01, 1))
plt.show()

print(f'days in submission: {submission_data["wd"].describe()}')


# %%

train_data = add_date_cols(train_data)
yrs = train_data["month"].unique()
colors = cm.get_cmap("viridis", len(yrs))
year_to_col = {y: colors(y - min(yrs)) for y in yrs}
evolution_phase_per_year = train_data.groupby(
    ["month", "wd"], as_index=False
).phase.mean()
for year, yr_phase in evolution_phase_per_year.groupby("month"):
    plt.scatter(
        yr_phase.wd, yr_phase.phase, c=[year_to_col[year]] * len(yr_phase), label=year
    )

plt.legend(bbox_to_anchor=(1.01, 1))
plt.show()

# %%


train_data = add_date_cols(train_data)
yrs = train_data["quarter"].unique()
colors = cm.get_cmap("viridis", len(yrs))
year_to_col = {y: colors(y - min(yrs)) for y in yrs}
evolution_phase_per_year = train_data.groupby(
    ["quarter", "wd"], as_index=False
).phase.mean()
for year, yr_phase in evolution_phase_per_year.groupby("quarter"):
    plt.scatter(
        yr_phase.wd, yr_phase.phase, c=[year_to_col[year]] * len(yr_phase), label=year
    )

plt.legend(bbox_to_anchor=(1.01, 1))
plt.show()

# %%
