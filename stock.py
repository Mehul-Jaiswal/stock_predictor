# Stock Analysis Script

# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data

# Function to calculate moving averages
def moving_averages(data, window):
    data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
    return data

# Function to calculate Bollinger Bands
def bollinger_bands(data, window=20):
    data['MA20'] = data['Close'].rolling(window=window).mean()
    data['STD20'] = data['Close'].rolling(window=window).std()
    data['Upper_BB'] = data['MA20'] + (data['STD20'] * 2)
    data['Lower_BB'] = data['MA20'] - (data['STD20'] * 2)
    return data

# Function to calculate RSI (Relative Strength Index)
def rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Function to calculate MACD (Moving Average Convergence Divergence)
def macd(data):
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data

# Function to perform basic linear regression prediction
def linear_regression_prediction(data):
    data['Prediction'] = data['Close'].shift(-1)
    X = np.array(data[['Close']])
    X = X[:-1]
    y = np.array(data['Prediction'])
    y = y[:-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='True Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.title('Linear Regression Stock Price Prediction')
    plt.legend()
    plt.show()

    return model

# Function to visualize stock data with indicators
def visualize_stock_data(data, ticker):
    plt.figure(figsize=(14, 7))

    # Plot close price and moving averages
    plt.subplot(3, 1, 1)
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['MA_50'], label='50-Day MA')
    plt.plot(data['MA_200'], label='200-Day MA')
    plt.title(f'{ticker} Stock Price and Moving Averages')
    plt.legend()

    # Plot Bollinger Bands
    plt.subplot(3, 1, 2)
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['Upper_BB'], label='Upper Bollinger Band')
    plt.plot(data['Lower_BB'], label='Lower Bollinger Band')
    plt.title(f'{ticker} Bollinger Bands')
    plt.legend()

    # Plot MACD
    plt.subplot(3, 1, 3)
    plt.plot(data['MACD'], label='MACD')
    plt.plot(data['Signal_Line'], label='Signal Line')
    plt.title(f'{ticker} MACD')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Function for fundamental analysis
def fundamental_analysis(ticker):
    stock = yf.Ticker(ticker)
    print("Company Information:")
    print(stock.info)

    # Display financials
    print("\nFinancials:")
    print(stock.financials)

    # Display major holders
    print("\nMajor Holders:")
    print(stock.major_holders)

    # Display institutional holders
    print("\nInstitutional Holders:")
    print(stock.institutional_holders)

    # Display recommendations
    print("\nAnalyst Recommendations:")
    print(stock.recommendations)

# Function to calculate average return and volatility
def average_return_volatility(data):
    data['Daily_Return'] = data['Close'].pct_change()
    avg_return = data['Daily_Return'].mean()
    volatility = data['Daily_Return'].std()
    return avg_return, volatility

# Function to calculate Sharpe Ratio
def sharpe_ratio(data, risk_free_rate=0.01):
    avg_return, volatility = average_return_volatility(data)
    sharpe = (avg_return - risk_free_rate) / volatility
    return sharpe

# Function to calculate and plot correlations between different stocks
def correlation_analysis(tickers, start_date, end_date):
    df = pd.DataFrame()
    for ticker in tickers:
        df[ticker] = fetch_stock_data(ticker, start_date, end_date)['Close']
    
    correlation = df.corr()
    print("Correlation Matrix:")
    print(correlation)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Stock Correlation Matrix')
    plt.show()

# Main function to run the analysis
def main():
    ticker = 'AAPL'  # Example stock ticker
    start_date = '2020-01-01'
    end_date = '2023-01-01'

    # Fetch data
    data = fetch_stock_data(ticker, start_date, end_date)

    # Calculate technical indicators
    data = moving_averages(data, 50)
    data = moving_averages(data, 200)
    data = bollinger_bands(data)
    data = rsi(data)
    data = macd(data)

    # Perform linear regression prediction
    model = linear_regression_prediction(data)

    # Visualization
    visualize_stock_data(data, ticker)

    # Fundamental analysis
    fundamental_analysis(ticker)

    # Calculate Sharpe Ratio
    sharpe = sharpe_ratio(data)
    print(f"Sharpe Ratio: {sharpe}")

    # Correlation Analysis with other stocks
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Example tickers
    correlation_analysis(tickers, start_date, end_date)

# Execute main function
if __name__ == "__main__":
    main()
