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

