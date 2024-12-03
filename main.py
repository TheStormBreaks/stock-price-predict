import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Fetching the stock data from yfinance api
ticker = 'GOOG'  # Company name
start_date = '2020-01-01'
end_date = '2024-01-01'

stock_data = yf.download(ticker, start = start_date, end = end_date)
stock_data.to_csv('stock_data.csv')

# Preprocess data
stock_data = stock_data[['Close']]
stock_data['Lag_1'] = stock_data['Close'].shift(1)
stock_data.dropna(implace = True)

# Linear Regression
X = stock_data[['Lag_1']].values
Y = stock_data['Close'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, shuffle = False)

model1 = LinearRegression()
model1.fit(X_train, Y_train)
predictions1 = model1.predict(X_test)

mae1 = mean_absolute_error(Y_test, predictions1)
mse1 = mean_squared_error(Y_test, predictions1)
r2_1 = r2_score(Y_test, predictions1)

print(f"Model 1 - Linear Regression - MAE: {mae1}, MSE: {mse1}, R-squared: {r2_1}")


# Random Forest Model 
