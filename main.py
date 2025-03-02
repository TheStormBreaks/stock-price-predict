import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fetching the stock data from yfinance API
ticker = 'GOOG'  # Company name
start_date = '2023-01-01'
end_date = '2024-01-01'

stock_data = yf.download(ticker, start=start_date, end=end_date)
print(stock_data.head())  # Debug: Check if data is loaded properly

# Feature Engineering
stock_data['Lag_1'] = stock_data['Close'].shift(1)
stock_data['MA_7'] = stock_data['Close'].rolling(window=7).mean()  # 7-day moving average
stock_data['MA_30'] = stock_data['Close'].rolling(window=30).mean()  # 30-day moving average
stock_data['Return'] = stock_data['Close'].pct_change()  # Daily returns
stock_data.dropna(inplace=True)

# Check the shape of the data after preprocessing
print(stock_data.shape)  # Debug: Check if data is valid after preprocessing

# Splitting data into features (X) and target (Y)
X = stock_data[['Lag_1', 'MA_7', 'MA_30', 'Return']].values
Y = stock_data['Close'].values

# Time-series split for cross-validation
tscv = TimeSeriesSplit(n_splits=3)  # Reduced number of splits for smaller datasets
for train_index, test_index in tscv.split(X):
    print(f"Train indices: {train_index}, Test indices: {test_index}")  # Debug: Check splits
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Linear Regression
    model_lr = LinearRegression()
    model_lr.fit(X_train, Y_train)
    predictions_lr = model_lr.predict(X_test)

    # Random Forest Regressor
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, Y_train)
    predictions_rf = model_rf.predict(X_test)

    # Evaluate models
    mae_lr = mean_absolute_error(Y_test, predictions_lr)
    mse_lr = mean_squared_error(Y_test, predictions_lr)
    r2_lr = r2_score(Y_test, predictions_lr)

    mae_rf = mean_absolute_error(Y_test, predictions_rf)
    mse_rf = mean_squared_error(Y_test, predictions_rf)
    r2_rf = r2_score(Y_test, predictions_rf)

    print(f"Linear Regression - MAE: {mae_lr}, MSE: {mse_lr}, R-squared: {r2_lr}")
    print(f"Random Forest - MAE: {mae_rf}, MSE: {mse_rf}, R-squared: {r2_rf}")

    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test, label='Actual Prices')
    plt.plot(predictions_lr, label='Linear Regression Predictions')
    plt.plot(predictions_rf, label='Random Forest Predictions')
    plt.legend()
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

# LSTM Model for Time-Series Forecasting
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Prepare data for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60  # Use past 60 days to predict the next day
X_lstm, Y_lstm = create_dataset(X, time_step)
X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

# Train LSTM model
model_lstm = create_lstm_model((X_lstm.shape[1], 1))
model_lstm.fit(X_lstm, Y_lstm, epochs=20, batch_size=32, verbose=1)

# Predict using LSTM
predictions_lstm = model_lstm.predict(X_lstm)

# Plot LSTM predictions
plt.figure(figsize=(10, 6))
plt.plot(Y_lstm, label='Actual Prices')
plt.plot(predictions_lstm, label='LSTM Predictions')
plt.legend()
plt.title(f'{ticker} Stock Price Prediction (LSTM)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()