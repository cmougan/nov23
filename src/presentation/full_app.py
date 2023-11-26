from datetime import datetime

import pandas as pd
import plotnine as pn
import streamlit as st

# Sample data (replace this with your actual data)
df = pd.read_csv("submissions/submission_ensemble_104.csv")
df["month"] = df["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").month)

train_data = pd.read_parquet("data/train_data.parquet")

# Streamlit app
st.title('Prediction Insights')

# Tab selection
selected_tab = st.radio('Select Tab:', ['Data Exploration', 'Insights'])

# Data Exploration Tab
if selected_tab == 'Data Exploration':
    st.subheader('Select your brand, country and month to see the predictions')

    # Sidebar for user input
    selected_country = st.sidebar.selectbox('Select Country', df['country'].unique())
    selected_brand = st.sidebar.selectbox('Select Brand', df['brand'].unique())
    selected_month = st.sidebar.selectbox('Select Month', df['month'].unique())

    # Filter data based on user selection
    filtered_df = df[(df['country'] == selected_country) & (df['brand'] == selected_brand) & (df['month'] == selected_month)]

    # Display the filtered data
    if not filtered_df.empty:

        # Create ggplot figure
        ggplot_fig = (
            pn.ggplot(
                data=filtered_df,
                mapping=pn.aes(x="date", y="prediction")
            ) 
            + pn.geom_point()
            + pn.theme_minimal()
            + pn.theme(axis_text_x=pn.element_text(angle=90)) 
            + pn.labs(x="Date", y="Prediction")
        )

        st.pyplot(ggplot_fig.draw())
    else:
        st.warning('No data available for the selected criteria.')

# Insights Tab
elif selected_tab == 'Insights':
    st.subheader('Historical Data Insights')
    train_data["year"] = pd.to_datetime(train_data["date"]).dt.year
    train_data["month"] = pd.to_datetime(train_data["date"]).dt.month
    selected_year = st.sidebar.selectbox('Select Year', train_data['year'].unique())
    selected_month = st.sidebar.selectbox('Select Month', train_data['month'].unique())

    train_df_year = train_data[train_data["year"] == selected_year]

    # Add your insights content here
    country_dw = (
        train_df_year
        .groupby(["country", "dayweek"], as_index=False)
        .agg({"phase": "mean", "brand": "size"})
        .rename(columns={"brand": "count"})
        # transform dayweek to names
        .assign(dayweek=lambda x: x["dayweek"].map({0: "0 Monday", 1: "1 Tuesday", 2: "2 Wednesday", 3: "3 Thursday", 4: "4 Friday", 5: "5 Saturday", 6: "6 Sunday"}))
    )

    country_wd = (
        train_df_year
        .groupby(["country", "wd"], as_index=False)
        .agg({"phase": "mean", "brand": "size"})
        .rename(columns={"brand": "count"})
    )

    brand_wd = (
        train_df_year
        .groupby(["brand", "wd"], as_index=False)
        .agg({"phase": "mean", "country": "size"})
        .rename(columns={"country": "count"})
    )

    month_wd = (
        train_df_year
        .groupby(["month", "wd"], as_index=False)
        .agg({"phase": "mean", "country": "size"})
        .rename(columns={"country": "count"})
    )

    top_brands = (
        train_df_year
        .groupby("brand", as_index=False)
        .agg({"phase": "mean", "monthly": "sum"})
        .rename(columns={"monthly": "count"})
        .sort_values("count", ascending=False)
        .head(12)
    )

    country_month_wd = (
        train_df_year
        .groupby(["country", "month", "wd"], as_index=False)
        .agg({"phase": "mean", "brand": "size"})
        .rename(columns={"brand": "count"})
    )

    p = (
        pn.ggplot(
            country_wd,
            pn.aes(x="wd", y="phase")
        ) + 
        pn.geom_point() +
        pn.geom_smooth(method="loess") +
        pn.facet_wrap("country")
        + pn.theme_minimal()
        # rotate x axis labels
        + pn.theme(axis_text_x=pn.element_text(angle=90))
        # remove legend title
        + pn.theme(legend_title=pn.element_blank())
        + pn.labs(x="Working day", y="Phase")

    
)

    st.pyplot(p.draw())

    # Additional functionalities can be added based on your use case
    # For example, you can visualize the data using charts or provide more details.
