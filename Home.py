import streamlit as st
import plotly.express as px
import polars as pl
import statsforecast.models
import statsforecast
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


st.header("Batting Average Prediction ⚾")

st.write(
    "This app predicts the batting average of a player for the next season using data from retrosheet."
)

st.write(
    "The model is trained on the player's historical data, excluding the season of interest."
)

st.write(
    "You may choose to use an Arima, ETS (Exponential Smoothing), or Historic Average model for the prediction."
)


# Model selection
model_options = [
    "AutoARIMA",
    "AutoETS",
    "HistoricAverage",
    "XGBoost"
]

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

    if model_name == "XGBoost":
        batting_adv = (
            pl.scan_parquet("parquets/batting-yearly-adv.parquet")
                .filter(pl.col("id") == player_id)
                .filter(pl.col("year") <= year)
        ).collect()
        batting_adv = batting_adv.sort(["id", "year"])
        
        # Lag OBP, SLG, and K% by 1 year to use as features for predicting next year
        batting_adv = batting_adv.with_columns([
            pl.col("b_obp").shift(1).alias("lag_obp"),
            pl.col("b_slg").shift(1).alias("lag_slg"),
            pl.col("b_k_pct").shift(1).alias("lag_k_pct")
        ])
        
        training_data = batting_adv.filter(pl.col("year") < year)
        X = training_data.select(["lag_obp", "lag_slg", "lag_k_pct"]).to_numpy()
        y = training_data.select("b_ba").to_numpy().ravel()
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=123)
        model.fit(X, y)
        
        prediction = model.predict(
            np.array([[batting_adv.filter(pl.col("year") == year).select("lag_obp").item()],
                [batting_adv.filter(pl.col("year") == year).select("lag_slg").item()],
                [batting_adv.filter(pl.col("year") == year).select("lag_k_pct").item()]]).reshape(1, -1)
        )
        forecast_value = prediction[0]
        forecasts_df = pl.DataFrame({
            'unique_id': [player_id],
            'ds': [year],
            'XGBoost': [forecast_value]
        })

        mse = mean_squared_error(y, model.predict(X))
        mae = mean_absolute_error(y, model.predict(X))
        r2 = r2_score(y, model.predict(X))

        # Display evaluation metrics
        st.write(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R-squared: {r2:.4f}")
        
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
            batting_adv,
            x="year",
            y="b_ba",
            title=f"Batting Average Prediction for {player}",
        )
        
        fig.add_scatter(
            x=[year],
            y=[forecast_value],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Prediction",
        )
        
        st.plotly_chart(fig)
        
        # Display prediction
        st.write(
            f"Predicted Batting Average for {year}: {forecast_value:.3f}"
        )
        
        # Display actual batting average
        st.write(
            f"Actual Batting Average for {year}: {batting_adv.filter(pl.col('year') == year).select('b_ba').item():.3f}"
        )
    else:
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
    
        model = statsforecast.models.__dict__.get(model_name)

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
        st.write(
            f"Actual Batting Average for {year}: {batting.filter(pl.col('year') == year).select('avg').item():.3f}"
        )
