import pandas as pd
import numpy as np
from datetime import datetime
import random

# Auxiliary function provided
def getTickerPrice(ticker: str, date: datetime) -> float:
    """
    Returns a simulated market price for the given ticker at the specified date.
    For this exercise, it returns a random number between 1 and 100.
    """
    return random.uniform(1, 100)

def preprocess_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the trade data:
    - Rename columns to match required format
    - Convert 'Date' column to datetime
    - Map 'Side' to 'buy' or 'sell'
    - Map amount ranges to numeric values and calculate 'Size'
    
    Args:
    df (pd.DataFrame): DataFrame containing trade data
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame
    """
    # Handle empty DataFrame
    if df.empty:
        print("The DataFrame is empty.")
        return df
    
    # Rename columns
    df = df.rename(columns={
        'transactionDate': 'Date',
        'ticker': 'Symbol',
        'type': 'Side'
    })

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Map 'Side' to 'buy' and 'sell'
    df['Side'] = df['Side'].apply(lambda x: 'buy' if 'Purchase' in x else 'sell')

    # Map amount ranges to numeric values
    amount_map = {
        '$1,001 - $15,000': (1001 + 15000) / 2,
        '$15,001 - $50,000': (15001 + 50000) / 2,
        '$50,001 - $100,000': (50001 + 100000) / 2,
        '$100,001 - $250,000': (100001 + 250000) / 2,
        '$250,001 - $500,000': (250001 + 500000) / 2,
        '$500,001 - $1,000,000': (500001 + 1000000) / 2,
        '$1,000,001 - $5,000,000': (1000001 + 5000000) / 2
    }
    df['Amount'] = df['amount'].map(amount_map)
    
    # Estimate 'Price' using the auxiliary function
    df['Price'] = [getTickerPrice(row['Symbol'], row['Date']) for _, row in df.iterrows()]

    # Calculate 'Size'
    df['Size'] = df['Amount'] / df['Price']
    return df

def calculate_trade_performance(df: pd.DataFrame) -> pd.Series:
    """
    Calculate 10 key financial metrics to track trade performance:
    - Total Trades
    - Total Volume
    - Average Trade Size
    - Gross Profit
    - Gross Loss
    - Net Profit
    - Total Buys
    - Total Sells
    - Win Rate
    - Profit Factor
    
    Args:
    df (pd.DataFrame): DataFrame containing trade data
    
    Returns:
    pd.Series: Series containing calculated metrics
    """
    # Handle empty DataFrame
    if df.empty:
        print("The DataFrame is empty.")
        return pd.Series({
            'total_trades': 0,
            'total_volume': 0,
            'average_trade_size': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'net_profit': 0,
            'total_buys': 0,
            'total_sells': 0,
            'win_rate': 0,
            'profit_factor': np.inf
        })

    # Ensure 'Size' is present and set default if missing
    df['Size'] = df['Size'].fillna(1)

    # Calculate metrics using vectorized operations
    total_trades = len(df)
    total_volume = df['Size'].sum()
    average_trade_size = df['Size'].mean()

    # Compute profit and loss
    df['Profit'] = np.where(df['Side'] == 'sell', df['Price'] * df['Size'], -df['Price'] * df['Size'])
    gross_profit = df.loc[df['Profit'] > 0, 'Profit'].sum()
    gross_loss = df.loc[df['Profit'] < 0, 'Profit'].sum()
    net_profit = gross_profit + gross_loss

    total_buys = df.loc[df['Side'] == 'buy', 'Size'].sum()
    total_sells = df.loc[df['Side'] == 'sell', 'Size'].sum()
    win_rate = (df['Profit'] > 0).mean()
    profit_factor = gross_profit / -gross_loss if gross_loss != 0 else np.inf

    # Return metrics as a Series
    metrics = {
        'total_trades': total_trades,
        'total_volume': total_volume,
        'average_trade_size': average_trade_size,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'net_profit': net_profit,
        'total_buys': total_buys,
        'total_sells': total_sells,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }
    return pd.Series(metrics)

def analyze_nancy_pelosi_trades(file_path: str, num_samples: int = 100):
    """
    Analyzes Nancy Pelosi's trading performance using the trade performance calculation function.
    
    Args:
    file_path (str): Path to the CSV file containing Nancy Pelosi's trades
    num_samples (int): Number of samples to average the metrics over
    
    Returns:
    pd.Series: Series containing averaged trade performance metrics
    """
    df_trades = pd.read_csv(file_path)
    metrics_list = []
    
    for _ in range(num_samples):
        # Ensure different randomness for each iteration
        df_trades_sampled = preprocess_trades(df_trades.copy())
        metrics = calculate_trade_performance(df_trades_sampled)
        metrics_list.append(metrics)
    
    averaged_metrics = pd.DataFrame(metrics_list).mean()
    return averaged_metrics

# Main execution block
if __name__ == '__main__':
    file_path = "C:\\Users\\BenPrendergast\\Documents\\Python\\Trading Test\\testData.csv"
    performance_metrics = analyze_nancy_pelosi_trades(file_path, num_samples=100)
    print("Averaged Trade Performance Metrics:")
    print(performance_metrics)
