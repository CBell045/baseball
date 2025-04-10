import streamlit as st
import polars as pl
import plotly.express as px

st.header("Stadiums")

st.subheader("Which MLB stadiums are the best and worst for batters?")

st.write(
    """
It may not seem to be a big difference to some... but in a career where the difference between .3 and .2 
could be worth millions of dollars, it can make a big difference.
"""
)

batting = pl.scan_parquet("parquets/batting.parquet")

# Stadium differences
stadium = (
    batting.group_by("site")
    .agg(pl.col("b_h").sum(), pl.col("b_ab").sum())
    .with_columns((pl.col("b_h") / pl.col("b_ab")).alias("avg"))
    .filter(pl.col("b_ab") > 100000)
    # Get an average of all the stadiums
    .with_columns((pl.col("avg") - pl.col("avg").mean()).alias("diff"))
    # Get the best and worst stadiums
    .filter(
        (pl.col("diff") > pl.col("diff").mean() + pl.col("diff").std())
        | (pl.col("diff") < pl.col("diff").mean() - pl.col("diff").std())
    )
    .sort("avg")
    .with_columns(pl.col("diff").round(3), pl.col("avg").round(3))
    .rename({"avg": "Batting Average", "diff": "Difference from average"})
    .join(
        pl.scan_parquet("parquets/parks.parquet").select(["ID", "City", "Park Name"]),
        left_on="site",
        right_on="ID",
        how="left",
    )
    .collect()
)


# Visualize with plotly
fig = px.bar(
    stadium,
    x="Difference from average",
    y="Park Name",
    title="Batting average by stadium",
    orientation="h",
    color="Difference from average",
    color_continuous_scale="RdYlGn",
)

fig.update_layout(
    coloraxis_showscale=False,
)

st.plotly_chart(fig)

st.dataframe(
    stadium.select(["Park Name", "City", "Batting Average", "Difference from average"]),
    use_container_width=True,
)
