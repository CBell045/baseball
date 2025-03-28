import streamlit as st
import polars as pl


import polars as pl

players = (
    pl.scan_parquet("parquets/allplayers.parquet")
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
        .group_by("id", "year")
        .agg(pl.col("b_h").sum(), pl.col("b_ab").sum())
        .with_columns((pl.col("b_h") / pl.col("b_ab")).alias("avg"))
        .filter(pl.col("b_ab") > 100)
        .select(
            pl.col("id").alias("unique_id"),
            pl.col("year").alias("ds"),
            pl.col("avg").alias("y"),
        )
    ).collect()

    import statsforecast.models

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

    forecasts_df = sf.forecast(df=batting, h=1, level=[90])
    st.dataframe(forecasts_df)
