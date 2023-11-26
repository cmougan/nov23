# %% 
import pandas as pd
import plotnine as pn

pd.set_option("display.max_columns", 500)

pn.theme_set(pn.theme_minimal())

# %%
train_data = pd.read_parquet("data/train_data.parquet")
# %%
country_dw = (
    train_data
    .groupby(["country", "dayweek"], as_index=False)
    .agg({"phase": "mean", "brand": "size"})
    .rename(columns={"brand": "count"})
    # transform dayweek to names
    .assign(dayweek=lambda x: x["dayweek"].map({0: "0 Monday", 1: "1 Tuesday", 2: "2 Wednesday", 3: "3 Thursday", 4: "4 Friday", 5: "5 Saturday", 6: "6 Sunday"}))
)

country_wd = (
    train_data
    .groupby(["country", "wd"], as_index=False)
    .agg({"phase": "mean", "brand": "size"})
    .rename(columns={"brand": "count"})
)

brand_wd = (
    train_data
    .groupby(["brand", "wd"], as_index=False)
    .agg({"phase": "mean", "country": "size"})
    .rename(columns={"country": "count"})
)

month_wd = (
    train_data
    .groupby(["month", "wd"], as_index=False)
    .agg({"phase": "mean", "country": "size"})
    .rename(columns={"country": "count"})
)

top_brands = (
    train_data
    .groupby("brand", as_index=False)
    .agg({"phase": "mean", "monthly": "sum"})
    .rename(columns={"monthly": "count"})
    .sort_values("count", ascending=False)
    .head(12)
)

country_month_wd = (
    train_data
    .groupby(["country", "month", "wd"], as_index=False)
    .agg({"phase": "mean", "brand": "size"})
    .rename(columns={"brand": "count"})
)


# %%
ther_area_wd = (
    train_data
    .assign(
        ther_area=lambda x: x["ther_area"].astype(str).fillna("Unknown")
    )
    .groupby(["ther_area", "wd"], as_index=False)
    .agg({"phase": "mean", "brand": "size"})
    .rename(columns={"brand": "count"})
)

main_channel_wd = (
    train_data
    .assign(
        main_channel=lambda x: x["main_channel"].astype(str).fillna("Unknown")
    )
    .groupby(["main_channel", "wd"], as_index=False)
    .agg({"phase": "mean", "brand": "size"})
    .rename(columns={"brand": "count"})
)

main_channel_dw = (
    train_data
    .assign(
        main_channel=lambda x: x["main_channel"].astype(str).fillna("Unknown")
    )
    .groupby(["main_channel", "dayweek"], as_index=False)
    .agg({"phase": "mean", "brand": "size"})
    .rename(columns={"brand": "count"})
    .assign(dayweek=lambda x: x["dayweek"].map({0: "0 Monday", 1: "1 Tuesday", 2: "2 Wednesday", 3: "3 Thursday", 4: "4 Friday", 5: "5 Saturday", 6: "6 Sunday"}))
)


# %%
(

    pn.ggplot(
        country_dw.query("country.isin(['Zamunda', 'Themyscira', 'Qarth', 'Atlantis'])"),
        pn.aes(x="dayweek", y="phase", size="count")) + 
    pn.geom_point() +
    pn.geom_smooth(method="lm") +
    pn.facet_wrap("country")
    # rotate x axis labels
    + pn.theme(axis_text_x=pn.element_text(angle=90))
    # remove legend title
    + pn.theme(legend_title=pn.element_blank())
    + pn.labs(x="Day of the week", y="Phase")
    
)
# %%

(
    pn.ggplot(
        country_wd, #.query("country.isin(['Zamunda', 'Themyscira', 'Qarth', 'Atlantis'])"),
        pn.aes(x="wd", y="phase", size="count")) + 
    pn.geom_smooth(method="loess") +
    # pn.geom_point() +
    pn.facet_wrap("country")
    # rotate x axis labels
    + pn.theme(axis_text_x=pn.element_text(angle=90))
    # remove legend title
    + pn.theme(legend_title=pn.element_blank())
    + pn.labs(x="Working day", y="Phase")
    
)
# %%
top_brands_wd = brand_wd.loc[brand_wd["brand"].isin(top_brands["brand"])]

# %%
(
    pn.ggplot(
        top_brands_wd, #.query("country.isin(['Zamunda', 'Themyscira', 'Qarth', 'Atlantis'])"),
        pn.aes(x="wd", y="phase", size="count")) + 
    pn.geom_smooth(method="loess") +
    # pn.geom_point() +
    pn.facet_wrap("brand")
    # rotate x axis labels
    + pn.theme(axis_text_x=pn.element_text(angle=90))
    # remove legend title
    + pn.theme(legend_title=pn.element_blank())
    + pn.labs(x="Working day", y="Phase")
    
)
# %%
(
    pn.ggplot(
        month_wd.assign(
            month=lambda x: x["month"].map({1: "01 January", 2: "02 February", 3: "03 March", 4: "04 April", 5: "05 May", 6: "06 June", 7: "07 July", 8: "08 August", 9: "09 September", 10: "10 October", 11: "11 November", 12: "12 December"})
        ), #.query("country.isin(['Zamunda', 'Themyscira', 'Qarth', 'Atlantis'])"),
        pn.aes(x="wd", y="phase", size="count")) +
    pn.geom_smooth(method="loess") +
    # pn.geom_point() +
    pn.facet_wrap("month")
    # rotate x axis labels
    + pn.theme(axis_text_x=pn.element_text(angle=90))
    # remove legend title
    + pn.theme(legend_title=pn.element_blank())
    + pn.labs(x="Working day", y="Phase")
)
# %%
(
    pn.ggplot(
        country_month_wd.assign(
            month=lambda x: x["month"].map({1: "01 January", 2: "02 February", 3: "03 March", 4: "04 April", 5: "05 May", 6: "06 June", 7: "07 July", 8: "08 August", 9: "09 September", 10: "10 October", 11: "11 November", 12: "12 December"})
        ).query("month == '01 January'"), #.query("country.isin(['Zamunda', 'Themyscira', 'Qarth', 'Atlantis'])"),
        pn.aes(x="wd", y="phase", size="count")) +
    pn.geom_smooth(method="loess") +
    # pn.geom_point() +
    pn.facet_wrap("country")
    # rotate x axis labels
    + pn.theme(axis_text_x=pn.element_text(angle=90))
    # remove legend title
    + pn.theme(legend_title=pn.element_blank())
    + pn.labs(x="Working day", y="Phase")
    # add title
    + pn.ggtitle("January for different countries")
)
# %%
(
    pn.ggplot(
        country_month_wd.assign(
            month=lambda x: x["month"].map({1: "01 January", 2: "02 February", 3: "03 March", 4: "04 April", 5: "05 May", 6: "06 June", 7: "07 July", 8: "08 August", 9: "09 September", 10: "10 October", 11: "11 November", 12: "12 December"})
        ).query("month == '08 August'"), #.query("country.isin(['Zamunda', 'Themyscira', 'Qarth', 'Atlantis'])"),
        pn.aes(x="wd", y="phase", size="count")) +
    pn.geom_smooth(method="loess") +
    # pn.geom_point() +
    pn.facet_wrap("country")
    # rotate x axis labels
    + pn.theme(axis_text_x=pn.element_text(angle=90))
    # remove legend title
    + pn.theme(legend_title=pn.element_blank())
    + pn.labs(x="Working day", y="Phase")
    # add title
    + pn.ggtitle("August for different countries")
)
# %%
ther_area_wd
# %%
(
    pn.ggplot(
        ther_area_wd, #.query("country.isin(['Zamunda', 'Themyscira', 'Qarth', 'Atlantis'])"),
        pn.aes(x="wd", y="phase", size="count")) +
    pn.geom_smooth(method="loess") +
    # pn.geom_point() +
    pn.facet_wrap("ther_area")
    # rotate x axis labels
    + pn.theme(axis_text_x=pn.element_text(angle=90))
    # remove legend title
    + pn.theme(legend_title=pn.element_blank())
    + pn.labs(x="Working day", y="Phase")

)
# %%
(
    pn.ggplot(
        main_channel_wd.query("main_channel != 'nan'"), #.query("country.isin(['Zamunda', 'Themyscira', 'Qarth', 'Atlantis'])"),
        pn.aes(x="wd", y="phase", size="count")) +
    pn.geom_smooth(method="loess") +
    # pn.geom_point() +
    pn.facet_wrap("main_channel")
    # rotate x axis labels
    + pn.theme(axis_text_x=pn.element_text(angle=90))
    # remove legend title
    + pn.theme(legend_title=pn.element_blank())
    + pn.labs(x="Working day", y="Phase")

)
# %%
(
    pn.ggplot(
        main_channel_dw.query("main_channel != 'nan'"), #.query("country.isin(['Zamunda', 'Themyscira', 'Qarth', 'Atlantis'])"),
        pn.aes(x="dayweek", y="phase", size="count")) +
    # pn.geom_smooth(method="loess") +
    pn.geom_point() +
    pn.facet_wrap("main_channel")
    # rotate x axis labels
    + pn.theme(axis_text_x=pn.element_text(angle=90))
    # remove legend title
    + pn.theme(legend_title=pn.element_blank())
    + pn.labs(x="Day of the week", y="Phase")
)

# %%
main_channel_dw
# %%
