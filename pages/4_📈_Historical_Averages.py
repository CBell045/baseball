import streamlit as st
import polars as pl
import plotly.express as px

st.header("Historical Averages")

st.subheader("How have batting averages changed over time?")


st.write(
    """
Despite MLB's efforts to liven the game, such as banning the shift and the designated hitter,
the 2024 batting averages rival the "Dead Ball Era" and the "Year of the Pitcher" as some of the lowest in MLB history.
    """
)


batting = pl.scan_parquet("parquets/batting.parquet")

# League wide batting average per year
df = (
    batting.with_columns(
        pl.col("date")
        .cast(pl.String)
        .str.strptime(pl.Date, "%Y%m%d")
        .dt.year()
        .alias("year")
    )
    .group_by("year")
    .agg(pl.col("b_h").sum(), pl.col("b_ab").sum())
    .with_columns((pl.col("b_h") / pl.col("b_ab")).alias("avg").round(3))
    .sort("year")
    .cast({"year": pl.String})
    .rename({"year": "Year", "avg": "Average"})
    .collect()
)

# Visualize with plotly
fig = px.line(
    df,
    x="Year",
    y="Average",
    title="League wide batting average per year",
    line_shape="spline",
)

# Make annotation in 1968
fig.add_annotation(
    x="1968",
    y=df.filter(pl.col("Year") == "1968")["Average"][0],
    text="The Year of the Pitcher",
    showarrow=True,
    arrowhead=1,
    ax=-80,
    ay=-20,
    standoff=5,
)

# Make annotation of dead ball era
fig.add_annotation(
    x="1910",
    y=".275",
    text="Dead Ball Era",
    showarrow=False,
    ax=80,
    ay=-20,
)

# Add last year annotation
fig.add_annotation(
    x=df["Year"][-1],
    y=df["Average"][-1],
    text="Now",
    showarrow=True,
    ax=40,
    ay=-20,
    standoff=5,
)


st.plotly_chart(fig)


st.dataframe(
    df.select(pl.col("Year"), pl.col("Average"), pl.col("Average").alias("Bar Chart")),
    column_config={
        "Bar Chart": st.column_config.ProgressColumn(
            "Bar Chart",
            help="League wide batting average",
            format="%.3f",
            min_value=df["Average"].min(),
            max_value=df["Average"].max(),
        ),
    },
    use_container_width=True,
)


st.write(
    """
    ## References
    - [Dead Ball Era](https://en.wikipedia.org/wiki/Dead-ball_era) - "The dead-ball era refers to a period from about 1900 to 1920 in which run scoring was low and home runs were rare in comparison to the years that followed."
    - [The Year of the Pitcher](https://en.wikipedia.org/wiki/1968_in_baseball) - "The collective batting average of .231 is the all-time lowest. As a result of the dropping offensive statistics, Major League Baseball Rules Committee took steps to reduce the advantage held by pitchers by lowering the height of the pitchers mound from 15 inches to 10 inches, and by reducing the size of the strike zone for the 1969 season."
    """
)
