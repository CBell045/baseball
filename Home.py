import streamlit as st


st.header("Batting Average Prediction ⚾")

st.write(
    "This app explores batting averages for Major League Baseball players and predicts future seasons using historical data."
)

st.write(
    "The data is sourced from [retrosheet](https://www.retrosheet.org/) and includes batting statistics for each player, including games played, at-bats, hits, walks, and more."
)

st.write(
    """To get started, select a page from the sidebar. Options include:
- Time Series Models
- XGBoost Model
- Stadium Effects
- Historical Averages
- Aggregate Predictions
"""
)

st.write(
    "The app is built using Streamlit, Plotly, and Polars, and is hosted on Streamlit Cloud."
)
