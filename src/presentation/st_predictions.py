from datetime import datetime

import pandas as pd
import plotnine as pn
import streamlit as st

# Sample data (replace this with your actual data)
df = pd.read_csv("submissions/submission_ensemble_104.csv")
df["month"] = df["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").month)

# Streamlit app
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
        # + pn.geom_point()
        + pn.geom_point()
        + pn.theme_minimal()
        + pn.theme(axis_text_x=pn.element_text(angle=90)) 
        + pn.labs(x="Date", y="Prediction")
    )

    st.pyplot(ggplot_fig.draw())
else:
    st.warning('No data available for the selected criteria.')

# Additional functionalities can be added based on your use case
# For example, you can visualize the data using charts or provide more details.

