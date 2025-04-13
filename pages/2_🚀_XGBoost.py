import polars as pl
import streamlit as st
import plotly.express as px
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

st.header("XGBoost Model")


st.write(
    "The model is trained on all players' historical data, up to 2019."
)


year_options = range(2024, 2019, -1)

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


if player and year:
    player_id = players.filter(pl.col("name") == player).select("id").item()

    batting_adv = (
        pl.scan_parquet("parquets/batting-yearly-adv.parquet")
        .filter(pl.col("id") == player_id)
        .filter(pl.col("year") <= year)
    ).collect()
    batting_adv = batting_adv.sort(["id", "year"])

    training_data = batting_adv.filter(pl.col("year") < year)
    X = training_data.select(["lag_b_obp", "lag_b_slg", "lag_b_k_pct"]).to_numpy()
    y = training_data.select("b_ba").to_numpy().ravel()

    model = xgb.XGBRegressor()
    model.load_model("models/xgboost.json")

    prediction = model.predict(
        np.array(
            [
                [batting_adv.filter(pl.col("year") == year).select("lag_b_obp").item()],
                [batting_adv.filter(pl.col("year") == year).select("lag_b_slg").item()],
                [batting_adv.filter(pl.col("year") == year).select("lag_b_k_pct").item()],
            ]
        ).reshape(1, -1)
    )
    forecast_value = prediction[0]
    forecasts_df = pl.DataFrame(
        {"unique_id": [player_id], "ds": [year], "XGBoost": [forecast_value]}
    )

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
    st.write(f"Predicted Batting Average for {year}: {forecast_value:.3f}")

    # Display actual batting average
    st.write(
        f"Actual Batting Average for {year}: {batting_adv.filter(pl.col('year') == year).select('b_ba').item():.3f}"
    )
