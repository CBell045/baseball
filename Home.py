import streamlit as st
import polars as pl
import altair as alt

# Sample batting average data
df = pl.DataFrame(
    {
        "Player": [
            "Mike Trout",
            "Mookie Betts",
            "Aaron Judge",
            "Shohei Ohtani",
            "Freddie Freeman",
        ],
        "Batting Average": [0.312, 0.292, 0.287, 0.280, 0.305],
    }
)

st.header("Batting Average Prediction ⚾")

# Dropdown to select player(s)
players = st.multiselect("Select Player", df["Player"].to_list())

if players:
    # Filter data based on selected player(s)
    df = df.filter(df["Player"].is_in(players))

# Create bar chart
chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("Player", sort="-y"),
        y=alt.Y("Batting Average", title="Batting Average"),
    )
    .properties(title="MLB Players - Batting Averages")
)

# Display chart in Streamlit
st.altair_chart(chart, use_container_width=True)
