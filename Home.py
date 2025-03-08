import streamlit as st
import pybaseball
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import plotly.express as px

pd.options.mode.copy_on_write = True


@st.cache_data()
def load_data():
    # Load data from pybaseball
    return pybaseball.batting_stats(2000, 2024)


st.header("Batting Average Prediction ⚾")

data = load_data()

# Dropdown to select player(s)
player: str = st.selectbox("Select Player", data["Name"].unique())

if player:
    # Filter data based on selected player(s)
    player_df = data[data["Name"] == player]

    # Player df sort
    player_df = player_df.sort_values(by="Season")

    # Dropdown to select year
    year = st.selectbox(
        "Select Year", sorted(player_df["Season"].unique().tolist(), reverse=True)
    )

    if year:
        # Filter data based on selected year
        df = player_df[player_df["Season"] < year]

        df["Season"] = pd.to_datetime(df["Season"], format="%Y")
        df = df.set_index("Season").sort_index().asfreq("YS")

        # Predict
        model = ARIMA(df["AVG"], order=(1, 1, 1))
        fitted_model = model.fit()

        forecast = fitted_model.forecast(steps=1)

        # Display prediction
        st.write(f"Predicted Batting Average for {year}: {forecast.iloc[0]:.3f}")

        st.write(
            f"Actual Batting Average for {year}: {player_df[player_df['Season'] == year]['AVG'].values[0]:.3f}"
        )

        # Create chart
        fig = px.line(
            player_df,
            x="Season",
            y="AVG",
            title=f"Batting Average Prediction for {player}",
        )

        fig.add_scatter(
            x=[year],
            y=[forecast.iloc[0]],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Prediction",
        )

        st.plotly_chart(fig)
