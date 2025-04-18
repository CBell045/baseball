import streamlit as st
import plotly.express as px
import polars as pl
import statsforecast.models
import statsforecast
from statsmodels.tsa.statespace.sarimax import SARIMAX


st.header("Time Series Models")


st.write(
    "The model is trained on the player's historical data, excluding the season of interest."
)

st.write(
    "You may choose to use an Arima, ETS (Exponential Smoothing), or Historic Average model for the prediction."
)


# Model selection
model_options = ["AutoARIMA", "AutoETS", "HistoricAverage", "SARIMA"]

# Select model for prediction
model_name = st.selectbox(
    "Select Forecasting Model",
    model_options,
)


year_options = range(2024, 1910, -1)

year: int = st.selectbox(
    "Select Year to Predict Batting Average",
    year_options,
)


# Players who played in year
active_players = (
    (
        pl.scan_parquet("parquets/allplayers.parquet")
        .filter(pl.col("season") == year)
        .select("id")
        .unique()
    )
    .collect()
    .to_series()
)


# Dropdown to select player(s)
players = (
    pl.scan_parquet("parquets/allplayers.parquet")
    .filter(pl.col("id").is_in(active_players))
    # Filter out pitchers
    .filter((pl.col("g_p") + pl.col("g_sp") + pl.col("g_rp")) < 10)
    .filter(pl.col("g") > 60)
    .with_columns(
        (pl.col("first") + pl.lit(" ") + pl.col("last")).alias("name"),
        pl.count("season").over("id").alias("count"),
    )
    .filter(pl.col("name").is_not_null())
    .filter(pl.col("count") > 5)
    .select(
        pl.col("name"),
        pl.col("id"),
    )
    .unique(["id"])
    .sort("name")
).collect()

player: str = st.selectbox(
    "Select Player",
    players,
)


if player and year and model_name:
    player_id = players.filter(pl.col("name") == player).select("id").item()

    # Filter data based on selected player(s)
    batting = (
        pl.scan_parquet("parquets/batting.parquet")
        .filter(pl.col("id") == player_id)
        .with_columns(
            pl.col("date")
            .cast(pl.String)
            .str.strptime(pl.Date, "%Y%m%d")
            .dt.year()
            .alias("year")
        )
        .filter(pl.col("year") <= year)
        .group_by("id", "year")
        .agg(pl.col("b_h").sum(), pl.col("b_ab").sum())
        .with_columns((pl.col("b_h") / pl.col("b_ab")).alias("avg"))
        .filter(pl.col("b_ab") > 1)
        .sort("year")
    ).collect()

    if model_name != "SARIMA":
        model = statsforecast.models.__dict__.get(model_name)

        try:
            # Instantiate StatsForecast class as sf
            sf = statsforecast.StatsForecast(
                models=[model()],
                freq=1,
                n_jobs=-1,
                verbose=True,
            )
            forecasts_df = sf.forecast(
                df=(
                    batting.filter(pl.col("year") < year).select(
                        pl.col("id").alias("unique_id"),
                        pl.col("year").alias("ds"),
                        pl.col("avg").alias("y"),
                    )
                ),
                h=1,
                level=[75],
            )

            actual_avg = batting.filter(pl.col("year") == year).select("avg").item()

            mse = (forecasts_df.select(model_name).item() - actual_avg) ** 2
            mae = abs(forecasts_df.select(model_name).item() - actual_avg)

            # Display evaluation metrics
            st.write(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

            st.dataframe(
                forecasts_df,
                column_config={
                    "unique_id": "Player ID",
                    "ds": st.column_config.TextColumn(
                        "Year",
                        help="Year / Season",
                    ),
                },
            )
            # Create chart
            fig = px.line(
                batting,
                x="year",
                y="avg",
                title=f"Batting Average Prediction for {player}",
            )

            fig.add_scatter(
                x=[year],
                y=[forecasts_df.select(model_name).item()],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Prediction",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[
                        forecasts_df.select(f"{model_name}-hi-75").item()
                        - forecasts_df.select(model_name).item()
                    ],
                    arrayminus=[
                        forecasts_df.select(model_name).item()
                        - forecasts_df.select(f"{model_name}-lo-75").item()
                    ],
                ),
            )

            st.plotly_chart(fig)

            # Display prediction
            st.write(
                f"Predicted Batting Average for {year}: {forecasts_df.select(model_name).item():.3f} (Confidence range of {forecasts_df.select(f'{model_name}-lo-75').item():.3f} to {forecasts_df.select(f'{model_name}-hi-75').item():.3f})"
            )
            # Display actual batting average
            if batting.filter(pl.col("year") == year).shape[0] > 0:
                st.write(
                    f"Actual Batting Average for {year}: {batting.filter(pl.col('year') == year).select('avg').item():.3f}"
                )
            else:
                st.write(
                    f"No data available for {year}. The player may not have played that year."
                )

        except Exception as e:
            st.error("Error making prediction. Please try another player")

    else:
        try:
            # Prepare data for SARIMAX
            sarimax_data = (
                batting.filter(pl.col("year") < year).select(pl.col("avg")).to_pandas()
            )

            # Instantiate SARIMAX model
            model = SARIMAX(
                endog=sarimax_data["avg"],
                order=(1, 0, 0),
                seasonal_order=(1, 0, 0, 12),  # Adjusted for yearly seasonality
                suppress_warnings=True,
            )

            # Fit the model
            fitted_model = model.fit(disp=False)

            # Forecast for the next period
            forecast_result = fitted_model.get_forecast(steps=1)
            forecast_mean = forecast_result.predicted_mean.iloc[0]
            forecast_ci = forecast_result.conf_int(alpha=0.25)

            # Prepare forecast DataFrame
            forecasts_df = pl.DataFrame(
                {
                    "unique_id": [player_id],
                    "ds": [year],
                    model_name: [forecast_mean],
                    f"{model_name}-lo-75": [forecast_ci.iloc[0, 0]],
                    f"{model_name}-hi-75": [forecast_ci.iloc[0, 1]],
                }
            )

            actual_avg = batting.filter(pl.col("year") == year).select("avg").item()
            mse = (forecast_mean - actual_avg) ** 2
            mae = abs(forecast_mean - actual_avg)
            # Display evaluation metrics
            st.write(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

            st.dataframe(
                forecasts_df,
                column_config={
                    "unique_id": "Player ID",
                    "ds": st.column_config.TextColumn(
                        "Year",
                        help="Year / Season",
                    ),
                },
            )
            # Create chart
            fig = px.line(
                batting,
                x="year",
                y="avg",
                title=f"Batting Average Prediction for {player}",
            )

            fig.add_scatter(
                x=[year],
                y=[forecasts_df.select(model_name).item()],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Prediction",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[
                        forecasts_df.select(f"{model_name}-hi-75").item()
                        - forecasts_df.select(model_name).item()
                    ],
                    arrayminus=[
                        forecasts_df.select(model_name).item()
                        - forecasts_df.select(f"{model_name}-lo-75").item()
                    ],
                ),
            )

            st.plotly_chart(fig)

            # Display prediction
            st.write(
                f"Predicted Batting Average for {year}: {forecasts_df.select(model_name).item():.3f} (Confidence range of {forecasts_df.select(f'{model_name}-lo-75').item():.3f} to {forecasts_df.select(f'{model_name}-hi-75').item():.3f})"
            )
            # Display actual batting average
            if batting.filter(pl.col("year") == year).shape[0] > 0:
                st.write(
                    f"Actual Batting Average for {year}: {batting.filter(pl.col('year') == year).select('avg').item():.3f}"
                )
            else:
                st.write(
                    f"No data available for {year}. The player may not have played that year."
                )
        except Exception as e:
            st.error("Error making prediction. Please try another player")
