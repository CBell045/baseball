import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import gc
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def process_chunk(chunk):
    """Process a chunk of data and calculate statistics."""
    # Extract year from date
    chunk['yearID'] = chunk['date'].astype(str).str[:4].astype(int)
    
    # Group by player and year
    stats = chunk.groupby(['id', 'yearID']).agg({
        'b_ab': 'sum',
        'b_h': 'sum',
        'b_d': 'sum',
        'b_t': 'sum',
        'b_hr': 'sum',
        'b_w': 'sum',
        'b_hbp': 'sum',
        'b_k': 'sum',
        'b_sf': 'sum'
    }).reset_index()
    
    # Calculate derived statistics
    stats['batting_avg'] = stats['b_h'] / stats['b_ab']
    stats['on_base_pct'] = (stats['b_h'] + stats['b_w'] + stats['b_hbp']) / (stats['b_ab'] + stats['b_w'] + stats['b_hbp'] + stats['b_sf'])
    stats['slugging_pct'] = (stats['b_h'] + stats['b_d'] + 2*stats['b_t'] + 3*stats['b_hr']) / stats['b_ab']
    stats['strikeout_rate'] = stats['b_k'] / stats['b_ab']
    stats['isolated_power'] = (stats['b_d'] + stats['b_t'] + stats['b_hr']) / stats['b_ab']
    stats['walk_rate'] = stats['b_w'] / (stats['b_ab'] + stats['b_w'])
    stats['extra_base_hit_rate'] = (stats['b_d'] + stats['b_t'] + stats['b_hr']) / stats['b_h']
    
    # Filter by minimum at bats
    stats = stats[stats['b_ab'] >= 100]
    
    return stats

def load_and_process_data(csv_file, chunk_size=100000):
    """Load and process data in parallel using multiple CPU cores."""
    print(f"Loading and processing data from {csv_file}...")
    
    # Get the number of CPU cores
    num_cores = os.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # Read the CSV file in chunks
    chunks = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_chunk, chunks))
    
    # Combine all results
    print("Combining processed chunks...")
    combined_stats = pd.concat(results, ignore_index=True)
    
    # Group by player and year again to handle any overlapping chunks
    final_stats = combined_stats.groupby(['id', 'yearID']).agg({
        'b_ab': 'sum',
        'b_h': 'sum',
        'batting_avg': 'mean',
        'on_base_pct': 'mean',
        'slugging_pct': 'mean',
        'strikeout_rate': 'mean',
        'isolated_power': 'mean',
        'walk_rate': 'mean',
        'extra_base_hit_rate': 'mean'
    }).reset_index()
    
    return final_stats

def list_available_players(stats_df, min_at_bats=100):
    """List available players with their batting averages."""
    player_stats = stats_df.groupby('id').agg({
        'b_ab': 'sum',
        'b_h': 'sum',
        'yearID': 'count'
    }).reset_index()
    
    player_stats['batting_avg'] = player_stats['b_h'] / player_stats['b_ab']
    player_stats = player_stats[player_stats['b_ab'] >= min_at_bats]
    player_stats = player_stats.sort_values('batting_avg', ascending=False)
    
    return player_stats

def predict_player_stats(player_id, model, scaler, imputer, stats_df, features):
    """Predict player statistics using the trained model."""
    player_stats = stats_df[stats_df['id'] == player_id].copy()
    
    if player_stats.empty:
        return None
    
    # Handle missing values using the same imputer used for training
    X = pd.DataFrame(
        imputer.transform(player_stats[features]),
        columns=features,
        index=player_stats.index
    )
    
    # Scale features using the same scaler used for training
    X = pd.DataFrame(
        scaler.transform(X),
        columns=features,
        index=player_stats.index
    )
    
    # Make predictions
    player_stats['predicted_avg'] = model.predict(X)
    
    return player_stats

def predict_future_performance(player_data, num_years=3):
    """Predict future performance based on historical trends."""
    if len(player_data) < 2:  # Need at least 2 data points for trend
        return None, None
        
    # Calculate trend using linear regression
    X = player_data['yearID'].values.reshape(-1, 1)
    y = player_data['batting_avg'].values
    trend_model = LinearRegression()
    trend_model.fit(X, y)
    
    # Generate future years
    last_year = player_data['yearID'].max()
    future_years = np.array(range(last_year + 1, last_year + num_years + 1)).reshape(-1, 1)
    
    # Predict future batting averages
    future_predictions = trend_model.predict(future_years)
    
    # Ensure predictions stay within reasonable bounds (0 to 1)
    future_predictions = np.clip(future_predictions, 0, 1)
    
    return future_years.flatten(), future_predictions

def generate_player_analysis(stats_df, model, scaler, imputer, features, min_at_bats=100):
    """Generate a comprehensive analysis of each player's current and predicted batting averages."""
    # Get the most recent year for each player
    latest_year = stats_df.groupby('id')['yearID'].max().reset_index()
    
    # Get player stats for their most recent year
    player_latest_stats = pd.merge(stats_df, latest_year, on=['id', 'yearID'])
    
    # Calculate current batting averages
    player_analysis = player_latest_stats.groupby('id').agg({
        'b_ab': 'sum',
        'b_h': 'sum',
        'yearID': 'first'
    }).reset_index()
    
    player_analysis['current_batting_avg'] = player_analysis['b_h'] / player_analysis['b_ab']
    
    # Filter by minimum at bats
    player_analysis = player_analysis[player_analysis['b_ab'] >= min_at_bats]
    
    # Get historical data for each player
    player_historical_data = {}
    for player_id in player_analysis['id']:
        player_historical_data[player_id] = stats_df[stats_df['id'] == player_id].copy()
    
    # Predict next year's batting average for each player
    predictions = []
    for _, player in player_analysis.iterrows():
        player_id = player['id']
        current_year = player['yearID']
        next_year = current_year + 1
        
        # Get historical data for this player
        hist_data = player_historical_data[player_id]
        
        # Predict using the model
        player_stats = hist_data.copy()
        if not player_stats.empty:
            # Handle missing values
            X = pd.DataFrame(
                imputer.transform(player_stats[features]),
                columns=features,
                index=player_stats.index
            )
            
            # Scale features
            X = pd.DataFrame(
                scaler.transform(X),
                columns=features,
                index=player_stats.index
            )
            
            # Make predictions
            player_stats['predicted_avg'] = model.predict(X)
            
            # Get the prediction for the most recent year
            latest_prediction = player_stats[player_stats['yearID'] == current_year]['predicted_avg'].values
            if len(latest_prediction) > 0:
                model_prediction = latest_prediction[0]
            else:
                model_prediction = player['current_batting_avg']
        else:
            model_prediction = player['current_batting_avg']
        
        # Predict using trend
        future_years, future_predictions = predict_future_performance(hist_data, num_years=1)
        if future_years is not None and len(future_predictions) > 0:
            trend_prediction = future_predictions[0]
        else:
            trend_prediction = player['current_batting_avg']
        
        # Combine predictions (average of model and trend)
        combined_prediction = (model_prediction + trend_prediction) / 2
        
        predictions.append({
            'id': player_id,
            'current_year': current_year,
            'next_year': next_year,
            'current_batting_avg': player['current_batting_avg'],
            'model_prediction': model_prediction,
            'trend_prediction': trend_prediction,
            'combined_prediction': combined_prediction
        })
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # Sort by current batting average
    predictions_df = predictions_df.sort_values('current_batting_avg', ascending=False)
    
    return predictions_df

def main():
    # File paths
    csv_file = 'Data Download/batting.csv'
    
    try:
        # Load and process data
        stats_df = load_and_process_data(csv_file)
        
        # List available players
        print("\nFetching available players (top 10 by batting average):")
        available_players = list_available_players(stats_df)
        print(available_players[['id', 'batting_avg', 'b_ab', 'yearID']].head(10).to_string())
        
        # Prepare features and target
        features = ['on_base_pct', 'slugging_pct', 'strikeout_rate', 
                   'isolated_power', 'walk_rate', 'extra_base_hit_rate']
        target = 'batting_avg'
        
        # Handle missing values
        print("\nHandling missing values...")
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(
            imputer.fit_transform(stats_df[features]),
            columns=features,
            index=stats_df.index
        )
        y = stats_df[target]
        
        # Scale the features
        print("Scaling features and fitting model...")
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=features,
            index=stats_df.index
        )
        
        # Fit the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R-squared and adjusted R-squared
        r2_score = model.score(X, y)
        n = len(y)
        p = len(features)
        adjusted_r2 = 1 - (1 - r2_score) * (n - 1) / (n - p - 1)
        
        # Print model performance
        print("\nModel Performance:")
        print(f"R-squared Score: {r2_score:.4f}")
        print(f"Adjusted R-squared Score: {adjusted_r2:.4f}")
        print("\nFeature Coefficients:")
        for feat, coef in zip(features, model.coef_):
            print(f"{feat}: {coef:.4f}")
        
        # Generate comprehensive player analysis
        print("\nGenerating player analysis...")
        player_analysis = generate_player_analysis(stats_df, model, scaler, imputer, features)
        
        # Print the analysis
        print("\nPlayer Analysis (Current and Predicted Batting Averages):")
        print("=" * 100)
        print(f"{'Player ID':<15} {'Current Year':<12} {'Next Year':<12} {'Current BA':<12} {'Model Pred':<12} {'Trend Pred':<12} {'Combined Pred':<12}")
        print("-" * 100)
        
        for _, player in player_analysis.iterrows():
            print(f"{player['id']:<15} {player['current_year']:<12} {player['next_year']:<12} "
                  f"{player['current_batting_avg']:.3f}      {player['model_prediction']:.3f}      "
                  f"{player['trend_prediction']:.3f}      {player['combined_prediction']:.3f}")
        
        # Save the analysis to a CSV file
        player_analysis.to_csv('player_analysis.csv', index=False)
        print("\nAnalysis saved to 'player_analysis.csv'")
        
        # Clear memory
        del stats_df
        del X
        del y
        del model
        gc.collect()
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
