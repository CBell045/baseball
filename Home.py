import streamlit as st
import plotly.express as px
import polars as pl
import statsforecast


st.header("Batting Average Prediction ⚾")

st.write(
    "This app predicts the batting average of a player for the next season using data from retrosheet."
    "The model is trained on the player's historical data, excluding their last season played."
    "The model then predicts the batting average for the last season and compares it with the actual batting average."
)

# Dropdown to select player(s)
players = (
    pl.scan_parquet("parquets/allplayers.parquet")
    .filter(pl.col("g") > 30)
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
    .unique()
    .sort("name")
).collect()

player: str = st.selectbox(
    "Select Player",
    players,
)

if player:
    player_id = players.filter(pl.col("name") == player).select("id").item()

    # Filter data based on selected player(s)
    batting = (
        pl.scan_parquet("parquets/batting.parquet")
        .filter(pl.col("id") == player_id)
        .filter(pl.col("ab") > 100)
        .with_columns(
            pl.col("date")
            .cast(pl.String)
            .str.strptime(pl.Date, "%Y%m%d")
            .dt.year()
            .alias("year")
        )
        .group_by("id", "year")
        .agg(pl.col("b_h").sum(), pl.col("b_ab").sum())
        .with_columns((pl.col("b_h") / pl.col("b_ab")).alias("avg"))
        .sort("year")
    ).collect()

    # Last year
    year = batting.select("year").max().item()

    models = [
        statsforecast.models.AutoARIMA(),
        # statsforecast.models.AutoETS(),
        # statsforecast.models.AutoRegressive(10),
        # statsforecast.models.HoltWinters(),
        # statsforecast.models.HistoricAverage(),
    ]

    # Instantiate StatsForecast class as sf
    sf = statsforecast.StatsForecast(
        models=models,
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
    st.dataframe(forecasts_df)

    # Display prediction
    st.write(
        f"Predicted Batting Average for {year}: {forecasts_df.select('AutoARIMA').item():.3f} (Confidence range of {forecasts_df.select('AutoARIMA-lo-75').item():.3f} to {forecasts_df.select('AutoARIMA-hi-75').item():.3f})"
    )

    st.write(
        f"Actual Batting Average for {year}: {batting.filter(pl.col('year') == year).select('avg').item():.3f}"
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
        y=[forecasts_df.select("AutoARIMA").item()],
        mode="markers",
        marker=dict(color="red", size=10),
        name="Prediction",
        error_y=dict(
            type="data",
            symmetric=False,
            array=[
                forecasts_df.select("AutoARIMA-hi-75").item()
                - forecasts_df.select("AutoARIMA").item()
            ],
            arrayminus=[
                forecasts_df.select("AutoARIMA").item()
                - forecasts_df.select("AutoARIMA-lo-75").item()
            ],
        ),
    )

    # Add confidence intervals

    st.plotly_chart(fig)
