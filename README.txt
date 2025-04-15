# Baseball Batting Average Prediction

Site URL: https://baseball-ba.streamlit.app/

## Description

Predicting a baseball player’s batting average (BA) for an upcoming season is a challenging yet valuable task. Our project aims to develop a data-driven predictive models utilizing historical player performance and key influencing factors.

Currently, there is a lack of open-source, up-to-date, and visually interactive tools that provide comprehensive insights into player performance trends. By leveraging machine learning and statistical modeling, we bridge this gap and offer a robust predictive solution that is both accurate and accessible.

This project uses Streamlit, Plotly, Polars, StatsForecast, and other libraries to create an interactive web application. Users can explore player specific data and see how well models are able to predict their batting average.

## Installation

1. Install uv: https://docs.astral.sh/uv/#installation

2. Download and open the project in an IDE of your choice.

3. Open a new terminal in the project folder and install dependencies using `uv sync`. This will create a virtual environment, install python, and install the dependencies.

## Execution

1. In the terminal, run the command `uv run streamlit run Home.py` to start the Streamlit server.

2. Open a web browser and navigate to `http://localhost:8501` to view the application.

3. Use the sidebar to select a page and explore the associated models and visualizations.

## Demo Video

https://www.youtube.com/watch?v=X3P2Xt1U1Ts
[![Watch the video](https://img.youtube.com/vi/X3P2Xt1U1Ts/0.jpg)](https://www.youtube.com/watch?v=X3P2Xt1U1Ts)

## Contributors

- Wyatt Pangan
- Madison Wilderman
- Afra Ismail
- Chad Bell
- Kristen Hanold

## Data Sources

This project uses data from [Retrosheet](https://retrosheet.org/). A data dictionary for the Retrosheet csv files can be found [here](https://www.retrosheet.org/downloads/csvcontents.html).
