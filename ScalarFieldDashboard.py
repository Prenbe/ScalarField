import pandas as pd
import numpy as np
import random
from datetime import datetime
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Auxiliary function to get ticker price at a given date
def getTickerPrice(ticker: str, date: datetime) -> float:
    return random.uniform(1, 100)

# Function to preprocess the 'amount' column and calculate 'Size'
def preprocess_trades(df: pd.DataFrame) -> pd.DataFrame:
    amount_map = {
        '$1,001 - $15,000': (1001 + 15000) / 2,
        '$15,001 - $50,000': (15001 + 50000) / 2,
        '$50,001 - $100,000': (50001 + 100000) / 2,
        '$100,001 - $250,000': (100001 + 250000) / 2,
        '$250,001 - $500,000': (250001 + 500000) / 2,
        '$500,001 - $1,000,000': (500001 + 1000000) / 2,
        '$1,000,001 - $5,000,000': (1000001 + 5000000) / 2
    }
    min_amount_map = {
        '$1,001 - $15,000': 1001,
        '$15,001 - $50,000': 15001,
        '$50,001 - $100,000': 50001,
        '$100,001 - $250,000': 100001,
        '$250,001 - $500,000': 250001,
        '$500,001 - $1,000,000': 500001,
        '$1,000,001 - $5,000,000': 1000001
    }
    max_amount_map = {
        '$1,001 - $15,000': 15000,
        '$15,001 - $50,000': 50000,
        '$50,001 - $100,000': 100000,
        '$100,001 - $250,000': 250000,
        '$250,001 - $500,000': 500000,
        '$500,001 - $1,000,000': 1000000,
        '$1,000,001 - $5,000,000': 5000000
    }
    
    df['Amount'] = df['amount'].map(amount_map)
    df['Min_Amount'] = df['amount'].map(min_amount_map)
    df['Max_Amount'] = df['amount'].map(max_amount_map)
    df['Size'] = df['Amount'] / df['Price']
    df['Min_Size'] = df['Min_Amount'] / df['Price']
    df['Max_Size'] = df['Max_Amount'] / df['Price']
    return df

# Function to calculate trade performance
def calculate_trade_performance(df: pd.DataFrame) -> pd.Series:
    # Ensure default Size is 1 if not provided
    if 'Size' not in df.columns:
        df['Size'] = 1
    else:
        df['Size'] = df['Size'].fillna(1)

    # Define metrics
    metrics = {
        'total_trades': len(df),
        'total_volume': df['Size'].sum(),
        'average_trade_size': df['Size'].mean(),
        'min_total_volume': df['Min_Size'].sum(),
        'max_total_volume': df['Max_Size'].sum(),
        'min_average_trade_size': df['Min_Size'].mean(),
        'max_average_trade_size': df['Max_Size'].mean(),
        'gross_profit': 0,
        'gross_loss': 0,
        'net_profit': 0,
        'total_buys': df[df['Side'] == 'buy']['Size'].sum(),
        'total_sells': df[df['Side'] == 'sell']['Size'].sum(),
        'average_buy_price': df[df['Side'] == 'buy']['Price'].mean(),
        'average_sell_price': df[df['Side'] == 'sell']['Price'].mean(),
        'total_profit_trades': 0,
        'total_loss_trades': 0,
        'win_rate': 0,
        'profit_factor': 0,
    }

    # Initialize variables for calculation
    position = {}
    profit = []

    for _, row in df.iterrows():
        symbol = row['Symbol']
        size = row['Size']
        price = row['Price']
        side = row['Side']

        # Adjust position for buys
        if side == 'buy':
            if symbol not in position:
                position[symbol] = {'size': 0, 'average_price': 0}
            position[symbol]['average_price'] = (position[symbol]['average_price'] * position[symbol]['size'] + price * size) / (position[symbol]['size'] + size)
            position[symbol]['size'] += size

        # Adjust position for sells and calculate profit
        elif side == 'sell':
            if symbol in position and position[symbol]['size'] >= size:
                avg_buy_price = position[symbol]['average_price']
                pnl = (price - avg_buy_price) * size
                profit.append(pnl)
                position[symbol]['size'] -= size

                # Update gross profit/loss
                if pnl > 0:
                    metrics['gross_profit'] += pnl
                else:
                    metrics['gross_loss'] += pnl

    # Calculate net profit
    metrics['net_profit'] = metrics['gross_profit'] + metrics['gross_loss']

    # Additional metrics
    metrics['total_profit_trades'] = len([p for p in profit if p > 0])
    metrics['total_loss_trades'] = len([p for p in profit if p <= 0])
    metrics['win_rate'] = metrics['total_profit_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
    metrics['profit_factor'] = -metrics['gross_profit'] / metrics['gross_loss'] if metrics['gross_loss'] != 0 else np.inf

    return pd.Series(metrics)

# Load and preprocess the test data file
df_trades = pd.read_csv("testData.csv")

df_trades = df_trades.rename(columns={
    'transactionDate': 'Date',
    'ticker': 'Symbol',
    'type': 'Side'
})

df_trades['Date'] = pd.to_datetime(df_trades['Date'])
df_trades['Side'] = df_trades['Side'].apply(lambda x: 'buy' if 'Purchase' in x else 'sell')
df_trades['Price'] = df_trades.apply(lambda row: getTickerPrice(row['Symbol'], row['Date']), axis=1)

# Preprocess to calculate Size based on Amount and Price
df_trades = preprocess_trades(df_trades)

# Calculate trade performance
trade_performance_metrics = calculate_trade_performance(df_trades)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Trade Performance Dashboard"),
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=df_trades['Date'].min(),
        end_date=df_trades['Date'].max(),
        display_format='YYYY-MM-DD',
        style={'margin-bottom': '20px'}
    ),
    dcc.Dropdown(
        id='stock-dropdown',
        options=[{'label': symbol, 'value': symbol} for symbol in df_trades['Symbol'].unique()],
        multi=True,
        placeholder="Select stocks",
        style={'margin-bottom': '20px'}
    ),
    dcc.Dropdown(
        id='metric-dropdown',
        options=[
            {'label': 'Total Trades', 'value': 'total_trades'},
            {'label': 'Total Volume', 'value': 'total_volume'},
            {'label': 'Average Trade Size', 'value': 'average_trade_size'},
            {'label': 'Gross Profit', 'value': 'gross_profit'},
            {'label': 'Gross Loss', 'value': 'gross_loss'},
            {'label': 'Net Profit', 'value': 'net_profit'},
            {'label': 'Total Buys', 'value': 'total_buys'},
            {'label': 'Total Sells', 'value': 'total_sells'},
            {'label': 'Average Buy Price', 'value': 'average_buy_price'},
            {'label': 'Average Sell Price', 'value': 'average_sell_price'},
            {'label': 'Total Profit Trades', 'value': 'total_profit_trades'},
            {'label': 'Total Loss Trades', 'value': 'total_loss_trades'},
            {'label': 'Win Rate', 'value': 'win_rate'},
            {'label': 'Profit Factor', 'value': 'profit_factor'},
            {'label': 'Min Total Volume', 'value': 'min_total_volume'},
            {'label': 'Max Total Volume', 'value': 'max_total_volume'},
            {'label': 'Min Average Trade Size', 'value': 'min_average_trade_size'},
            {'label': 'Max Average Trade Size', 'value': 'max_average_trade_size'}
        ],
        value='total_trades',
        style={'margin-bottom': '20px'}
    ),
    dcc.Markdown(id='metric-info'),
    dcc.Graph(id='monthly-bar-chart')
])

# Function to aggregate metrics for monthly bar chart
def aggregate_monthly_metrics(df, metric):
    df['Month'] = df['Date'].dt.to_period('M')
    if metric == 'total_trades':
        monthly_data = df.groupby(['Month', 'Symbol']).size().reset_index(name=metric)
    elif metric == 'total_volume':
        monthly_data = df.groupby(['Month', 'Symbol'])['Size'].sum().reset_index(name=metric)
    elif metric == 'average_trade_size':
        monthly_data = df.groupby(['Month', 'Symbol'])['Size'].mean().reset_index(name=metric)
    elif metric in ['gross_profit', 'gross_loss', 'net_profit']:
        profit = df.apply(lambda row: (row['Price'] - df[(df['Symbol'] == row['Symbol']) & (df['Side'] == 'buy')]['Price'].mean()) * row['Size']
                          if row['Side'] == 'sell' else 0, axis=1)
        if metric == 'gross_profit':
            monthly_data = df.assign(profit=profit).groupby(['Month', 'Symbol'])['profit'].apply(lambda x: x[x > 0].sum()).reset_index(name=metric)
        elif metric == 'gross_loss':
            monthly_data = df.assign(profit=profit).groupby(['Month', 'Symbol'])['profit'].apply(lambda x: x[x <= 0].sum()).reset_index(name=metric)
        elif metric == 'net_profit':
            monthly_data = df.assign(profit=profit).groupby(['Month', 'Symbol'])['profit'].sum().reset_index(name=metric)
    elif metric == 'total_buys':
        monthly_data = df[df['Side'] == 'buy'].groupby(['Month', 'Symbol'])['Size'].sum().reset_index(name=metric)
    elif metric == 'total_sells':
        monthly_data = df[df['Side'] == 'sell'].groupby(['Month', 'Symbol'])['Size'].sum().reset_index(name=metric)
    elif metric == 'average_buy_price':
        monthly_data = df[df['Side'] == 'buy'].groupby(['Month', 'Symbol'])['Price'].mean().reset_index(name=metric)
    elif metric == 'average_sell_price':
        monthly_data = df[df['Side'] == 'sell'].groupby(['Month', 'Symbol'])['Price'].mean().reset_index(name=metric)
    elif metric == 'total_profit_trades':
        profit = df.apply(lambda row: (row['Price'] - df[(df['Symbol'] == row['Symbol']) & (df['Side'] == 'buy')]['Price'].mean()) * row['Size']
                          if row['Side'] == 'sell' else 0, axis=1)
        monthly_data = df.assign(profit=profit).groupby(['Month', 'Symbol'])['profit'].apply(lambda x: (x > 0).sum()).reset_index(name=metric)
    elif metric == 'total_loss_trades':
        profit = df.apply(lambda row: (row['Price'] - df[(df['Symbol'] == row['Symbol']) & (df['Side'] == 'buy')]['Price'].mean()) * row['Size']
                          if row['Side'] == 'sell' else 0, axis=1)
        monthly_data = df.assign(profit=profit).groupby(['Month', 'Symbol'])['profit'].apply(lambda x: (x <= 0).sum()).reset_index(name=metric)
    elif metric == 'win_rate':
        profit = df.apply(lambda row: (row['Price'] - df[(df['Symbol'] == row['Symbol']) & (df['Side'] == 'buy')]['Price'].mean()) * row['Size']
                          if row['Side'] == 'sell' else 0, axis=1)
        monthly_data = df.assign(profit=profit).groupby(['Month', 'Symbol']).apply(lambda x: (x['profit'] > 0).sum() / len(x) if len(x) > 0 else 0).reset_index(name=metric)
    elif metric == 'profit_factor':
        profit = df.apply(lambda row: (row['Price'] - df[(df['Symbol'] == row['Symbol']) & (df['Side'] == 'buy')]['Price'].mean()) * row['Size']
                          if row['Side'] == 'sell' else 0, axis=1)
        grouped = df.assign(profit=profit).groupby(['Month', 'Symbol'])
        gross_profit = grouped['profit'].apply(lambda x: x[x > 0].sum())
        gross_loss = grouped['profit'].apply(lambda x: x[x <= 0].sum())
        monthly_data = (gross_profit / gross_loss.abs()).reset_index(name=metric)
    elif metric in ['min_total_volume', 'max_total_volume']:
        column = 'Min_Size' if metric == 'min_total_volume' else 'Max_Size'
        monthly_data = df.groupby(['Month', 'Symbol'])[column].sum().reset_index(name=metric)
    elif metric in ['min_average_trade_size', 'max_average_trade_size']:
        column = 'Min_Size' if metric == 'min_average_trade_size' else 'Max_Size'
        monthly_data = df.groupby(['Month', 'Symbol'])[column].mean().reset_index(name=metric)
    
    monthly_data['Month'] = monthly_data['Month'].astype(str)
    return monthly_data

# Define the callback to update the graphs and the metric info
@app.callback(
    [Output('metric-info', 'children'),
     Output('monthly-bar-chart', 'figure')],
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('stock-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_graphs(start_date, end_date, selected_stocks, selected_metric):
    # Filter the dataframe based on the selected date range and stocks
    filtered_df = df_trades[(df_trades['Date'] >= start_date) & (df_trades['Date'] <= end_date)]
    if selected_stocks:
        filtered_df = filtered_df[filtered_df['Symbol'].isin(selected_stocks)]
    
    # Calculate the trade performance metrics for the filtered dataframe
    filtered_metrics = calculate_trade_performance(filtered_df)
    
    # Create the metric info display with units
    metric_info = f"**Selected Metric: {selected_metric}**\n\n"
    unit = ' (shares)' if 'volume' in selected_metric or 'size' in selected_metric else ''
    metric_info += f"**Value:** {filtered_metrics[selected_metric]:,.2f}{unit}\n\n"
    metric_info += f"**Details:**\n"
    for key, value in filtered_metrics.items():
        unit = ' (shares)' if 'volume' in key or 'size' in key else ''
        metric_info += f"- **{key.replace('_', ' ').capitalize()}:** {value:,.2f}{unit}\n"

    # Aggregate data by month and symbol for the monthly bar chart
    monthly_data = aggregate_monthly_metrics(filtered_df, selected_metric)

    yaxis_title = selected_metric.replace('_', ' ').capitalize()
    yaxis_title += ' (shares)' if 'volume' in selected_metric or 'size' in selected_metric else ''
    monthly_bar_chart = px.bar(monthly_data, x='Month', y=selected_metric, color='Symbol', barmode='group',
                               title=f"Monthly {selected_metric.replace('_', ' ').capitalize()} for Selected Stocks")
    monthly_bar_chart.update_layout(xaxis_title="Month", yaxis_title=yaxis_title)

    return metric_info, monthly_bar_chart

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

