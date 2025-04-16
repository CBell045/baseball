import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set page config
st.set_page_config(
    page_title="Baseball Player Analysis Dashboard",
    page_icon="⚾",
    layout="wide"
)

# Title and description
st.title("⚾ Baseball Player Performance Analysis")
st.markdown("""
This dashboard provides insights into player performance and predictions based on historical data.
Explore current batting averages and predictions for the upcoming season.
""")

# Load the data
@st.cache_data
def load_data():
    try:
        return pd.read_csv('csvs/player_analysis.csv')
    except FileNotFoundError:
        st.error("Error: player_analysis.csv not found. Please run multi_regression.py first.")
        return None

# Load the data
df = load_data()

if df is not None:
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Year filter
    years = sorted(df['current_year'].unique())
    selected_year = st.sidebar.selectbox(
        "Select Year",
        years,
        index=len(years)-1
    )
    
    # Filter data based on selection
    filtered_df = df[df['current_year'] == selected_year]
    
    # Add names to filtered_df by joining with allplayers.parquet
    allplayers_df = pd.read_parquet("parquets/allplayers.parquet").drop_duplicates(subset=['id'])
    filtered_df = filtered_df.merge(
        allplayers_df[['id', 'first', 'last']],
        on='id',
        how='left'
    )
    filtered_df['name'] = filtered_df['first'] + ' ' + filtered_df['last']
    filtered_df.drop(columns=['first', 'last'], inplace=True)
    filtered_df = filtered_df[['name'] + [col for col in filtered_df.columns if col != 'name']]
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Average Current Batting Average",
            f"{filtered_df['current_batting_avg'].mean():.3f}"
        )
    
    with col2:
        st.metric(
            "Average Predicted Batting Average",
            f"{filtered_df['combined_prediction'].mean():.3f}"
        )
    
    with col3:
        st.metric(
            "Number of Players",
            len(filtered_df)
        )
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Performance Comparison", "Prediction Analysis", "Player Details"])
    
    with tab1:
        # Scatter plot of current vs predicted batting averages
        fig = px.scatter(
            filtered_df,
            x='current_batting_avg',
            y='combined_prediction',
            title='Current vs Predicted Batting Averages',
            labels={
                'current_batting_avg': 'Current Batting Average',
                'combined_prediction': 'Predicted Batting Average'
            }
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Bar chart comparing different prediction methods
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Model Prediction',
            x=filtered_df['name'],
            y=filtered_df['model_prediction']
        ))
        fig.add_trace(go.Bar(
            name='Trend Prediction',
            x=filtered_df['name'],
            y=filtered_df['trend_prediction']
        ))
        fig.add_trace(go.Bar(
            name='Combined Prediction',
            x=filtered_df['name'],
            y=filtered_df['combined_prediction']
        ))
        fig.update_layout(
            title='Comparison of Prediction Methods',
            barmode='group',
            xaxis_title='Player Name',
            yaxis_title='Predicted Batting Average'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Detailed player statistics table
        st.dataframe(
            filtered_df.sort_values('current_batting_avg', ascending=False),
            use_container_width=True
        )
    
    # Additional insights
    st.header("Key Insights")
    
    # Calculate some statistics
    improvement_pct = ((filtered_df['combined_prediction'] - filtered_df['current_batting_avg']) / 
                      filtered_df['current_batting_avg'] * 100)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 5 Players Expected to Improve")
        top_improvers = filtered_df.nlargest(5, 'combined_prediction')
        for _, player in top_improvers.iterrows():
            st.write(f"Player {player['name']}: {player['current_batting_avg']:.3f} → {player['combined_prediction']:.3f}")
    
    with col2:
        st.subheader("Prediction Confidence")
        st.write(f"Average prediction difference: {abs(filtered_df['model_prediction'] - filtered_df['trend_prediction']).mean():.3f}")
        st.write(f"Players with consistent predictions: {(abs(filtered_df['model_prediction'] - filtered_df['trend_prediction']) < 0.05).sum()}") 
