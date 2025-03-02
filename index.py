# Import necessary libraries
import yfinance as yf  # For fetching stock data
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV  # For model selection and validation
from sklearn.linear_model import LinearRegression  # For linear regression model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor  # For ensemble methods
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error  # For model evaluation metrics
from sklearn.preprocessing import MinMaxScaler  # For scaling data
from tensorflow.keras.models import Sequential  # For building LSTM model
from tensorflow.keras.layers import LSTM, Dense  # For LSTM and Dense layers in the neural network
from xgboost import XGBRegressor  # For XGBoost regressor
import logging  # For logging information

# Constants for configuration
TICKER = 'GOOG'  # Stock ticker symbol for Google
START_DATE = '2023-01-01'  # Start date for fetching stock data
END_DATE = '2024-01-01'  # End date for fetching stock data
TIME_STEP = 60  # Number of past days to consider for LSTM
EPOCHS = 50  # Number of epochs for training the LSTM model
BATCH_SIZE = 32  # Batch size for training the LSTM model
N_SPLITS = 5  # Number of splits for time-series cross-validation

# Configure logging to display informational messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to fetch stock data using yfinance API
def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data using yfinance API.
    """
    try:
        # Download stock data for the specified ticker and date range
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        # Check if data is empty and raise an error if so
        if stock_data.empty:
            raise ValueError(f"No data fetched for ticker {ticker}. Please check the ticker symbol and try again.")
        return stock_data
    except Exception as e:
        # Log any errors that occur during data fetching
        logging.error(f"Error fetching data: {e}")
        exit()  # Exit the program if there is an error

# Function to preprocess stock data
def preprocess_data(stock_data):
    """
    Preprocess stock data by adding features like lagged values, moving averages, and returns.
    """
    # Create lagged features and moving averages
    stock_data['Lag_1'] = stock_data['Close'].shift(1)  # Lagged close price (1 day)
    stock_data['MA_7'] = stock_data['Close'].rolling(window=7).mean()  # 7-day moving average
    stock_data['MA_30'] = stock_data['Close'].rolling(window=30).mean()  # 30-day moving average
    stock_data['Return'] = stock_data['Close'].pct_change()  # Daily returns
    stock_data.dropna(inplace=True)  # Drop rows with missing values
    return stock_data  # Return the preprocessed data

# Function to evaluate a model's performance
def evaluate_model(model, X_test, Y_test, model_name):
    """
    Evaluate the model using MAE, RMSE, R², and MAPE metrics.
    """
    predictions = model.predict(X_test)  # Make predictions using the test data
    # Calculate evaluation metrics
    mae = mean_absolute_error(Y_test, predictions)
    mse = mean_squared_error(Y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, predictions)
    mape = mean_absolute_percentage_error(Y_test, predictions)
    # Log the evaluation results
    logging.info(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, MAPE: {mape:.2f}")
    return predictions, mae, rmse, r2, mape  # Return predictions and metrics

# Function for hyperparameter tuning of Random Forest model
def tune_random_forest(X_train, Y_train):
    """
    Perform hyperparameter tuning for Random Forest using GridSearchCV.
    """
    # Define the parameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10]  # Minimum number of samples required to split an internal node
    }
    model = RandomForestRegressor(random_state=42)  # Initialize the Random Forest model
    # Set up GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, Y_train)  # Fit the model to the training data
    logging.info(f"Best parameters for Random Forest: {grid_search.best_params_}")  # Log best parameters found
    return grid_search.best_estimator_  # Return the best estimator

# Function for hyperparameter tuning of XGBoost model
def tune_xgboost(X_train, Y_train):
    """
    Perform hyperparameter tuning for XGBoost using RandomizedSearchCV.
    """
    # Define the parameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of trees
        'max_depth': [3, 6, 9],  # Maximum depth of the tree
        'learning_rate': [0.01, 0.1, 0.2]  # Step size shrinkage
    }
    model = XGBRegressor(random_state=42)  # Initialize the XGBoost model
    # Set up RandomizedSearchCV for hyperparameter tuning
    grid_search = RandomizedSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, n_iter=10)
    grid_search.fit(X_train, Y_train)  # Fit the model to the training data
    logging.info(f"Best parameters for XGBoost: {grid_search.best_params_}")  # Log best parameters found
    return grid_search.best_estimator_  # Return the best estimator

# Function to create an LSTM model
def create_lstm_model(input_shape):
    """
    Create an LSTM model for time-series forecasting.
    """
    model = Sequential()  # Initialize a sequential model
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))  # First LSTM layer
    model.add(LSTM(50, return_sequences=False))  # Second LSTM layer
    model.add(Dense(25))  # Dense layer with 25 units
    model.add(Dense(1))  # Output layer with 1 unit (predicted value)
    model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model
    return model  # Return the created model

# Function to prepare dataset for LSTM training
def create_dataset(data, time_step=1):
    """
    Prepare dataset for LSTM by creating sequences of past time steps.
    """
    X, Y = [], []  # Initialize input and output lists
    # Create sequences of past time steps
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])  # Use past 'time_step' days as input
        Y.append(data[i + time_step, 0])  # Use the next day's value as target
    return np.array(X), np.array(Y)  # Return as numpy arrays

# Function to plot results
def plot_results(Y_test, predictions_lr, predictions_rf, predictions_xgb, predictions_ensemble, fold):
    """
    Plot actual vs predicted prices for a single fold.
    """
    plt.figure(figsize=(10, 6))  # Create a new figure
    plt.plot(Y_test, label='Actual Prices', color='blue', linewidth=2)  # Plot actual prices
    plt.plot(predictions_lr, label='Linear Regression', color='orange', linestyle='--')  # Plot LR predictions
    plt.plot(predictions_rf, label='Random Forest', color='green', linestyle='-.')  # Plot RF predictions
    plt.plot(predictions_xgb, label='XGBoost', color='red', linestyle=':')  # Plot XGBoost predictions
    plt.plot(predictions_ensemble, label='Ensemble', color='purple', linestyle='-')  # Plot ensemble predictions
    plt.title(f'{TICKER} Stock Price Prediction (Fold {fold + 1})')  # Set title
    plt.xlabel('Time')  # Set x-axis label
    plt.ylabel('Price')  # Set y-axis label
    plt.legend()  # Show legend
    plt.grid(True)  # Enable grid for better readability

# Main function to run the stock price prediction workflow
def main():
    # Fetch and preprocess stock data
    stock_data = fetch_stock_data(TICKER, START_DATE, END_DATE)  # Fetch stock data
    logging.info(stock_data.head())  # Log the first few rows of the data
    stock_data = preprocess_data(stock_data)  # Preprocess the data
    logging.info(stock_data.shape)  # Log the shape of the preprocessed data

    # Split data into features (X) and target (Y)
    X = stock_data[['Lag_1', 'MA_7', 'MA_30', 'Return']].values  # Feature set
    Y = stock_data['Close'].values  # Target variable

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)  # Initialize time-series cross-validation
    results = []  # List to store results
    plot_figures = []  # List to store plot figures

    # Loop through each fold for cross-validation
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        logging.info(f"Fold {fold + 1}: Train indices: {train_index}, Test indices: {test_index}")  # Log indices
        X_train, X_test = X[train_index], X[test_index]  # Split the data into training and testing sets
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Linear Regression model
        model_lr = LinearRegression()  # Initialize the Linear Regression model
        model_lr.fit(X_train, Y_train)  # Fit the model to the training data
        predictions_lr, mae_lr, rmse_lr, r2_lr, mape_lr = evaluate_model(model_lr, X_test, Y_test, "Linear Regression")  # Evaluate the model

        # Random Forest model with hyperparameter tuning
        model_rf = tune_random_forest(X_train, Y_train)  # Tune and fit the Random Forest model
        predictions_rf, mae_rf, rmse_rf, r2_rf, mape_rf = evaluate_model(model_rf, X_test, Y_test, "Random Forest")  # Evaluate the model

        # XGBoost model with hyperparameter tuning
        model_xgb = tune_xgboost(X_train, Y_train)  # Tune and fit the XGBoost model
        predictions_xgb, mae_xgb, rmse_xgb, r2_xgb, mape_xgb = evaluate_model(model_xgb, X_test, Y_test, "XGBoost")  # Evaluate the model

        # Ensemble model using Voting Regressor
        ensemble_model = VotingRegressor([
            ('lr', model_lr),  # Add Linear Regression model to the ensemble
            ('rf', model_rf),  # Add Random Forest model to the ensemble
            ('xgb', model_xgb)  # Add XGBoost model to the ensemble
        ])
        ensemble_model.fit(X_train, Y_train)  # Fit the ensemble model
        predictions_ensemble, mae_ensemble, rmse_ensemble, r2_ensemble, mape_ensemble = evaluate_model(ensemble_model, X_test, Y_test, "Ensemble")  # Evaluate the model

        # Store results for the current fold
        results.append({
            'fold': fold + 1,
            'Linear Regression': (mae_lr, rmse_lr, r2_lr, mape_lr),
            'Random Forest': (mae_rf, rmse_rf, r2_rf, mape_rf),
            'XGBoost': (mae_xgb, rmse_xgb, r2_xgb, mape_xgb),
            'Ensemble': (mae_ensemble, rmse_ensemble, r2_ensemble, mape_ensemble)
        })

        # Plot results for the current fold
        plot_results(Y_test, predictions_lr, predictions_rf, predictions_xgb, predictions_ensemble, fold)
        plot_figures.append(plt.gcf())  # Store the current figure

    # LSTM Model
    scaler = MinMaxScaler(feature_range=(0, 1))  # Initialize MinMaxScaler
    scaled_data = scaler.fit_transform(stock_data[['Close']].values)  # Scale the closing prices
    X_lstm, Y_lstm = create_dataset(scaled_data, TIME_STEP)  # Create dataset for LSTM
    X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)  # Reshape input for LSTM
    model_lstm = create_lstm_model((X_lstm.shape[1], 1))  # Create LSTM model
    model_lstm.fit(X_lstm, Y_lstm, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)  # Train the LSTM model

    # Make predictions using the LSTM model
    predictions_lstm = model_lstm.predict(X_lstm)
    predictions_lstm = scaler.inverse_transform(predictions_lstm)  # Inverse scale the predictions
    Y_lstm_original = scaler.inverse_transform(Y_lstm.reshape(-1, 1))  # Inverse scale the actual values

    # Plot LSTM results
    plt.figure(figsize=(10, 6))  # Create a new figure
    plt.plot(Y_lstm_original, label='Actual Prices', color='blue', linewidth=2)  # Plot actual prices
    plt.plot(predictions_lstm, label='LSTM Predictions', color='red', linestyle='--')  # Plot LSTM predictions
    plt.title(f'{TICKER} Stock Price Prediction (LSTM Model)')  # Set title
    plt.xlabel('Time')  # Set x-axis label
    plt.ylabel('Price')  # Set y-axis label
    plt.legend()  # Show legend
    plt.grid(True)  # Enable grid for better readability
    plot_figures.append(plt.gcf())  # Store the LSTM figure

    # Show all plots at once
    plt.show()

# Entry point for the script
if __name__ == "__main__":
    main()  # Call the main function to execute the workflow