import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima 
import warnings # remove the module warnings

warnings.filterwarnings("ignore")

# initialie the data set
data = {
    'Year': np.arange(2010, 2024), 
    'Rice Special': [38.4, 39.26, 40.51, 40.87, 45.31, 45.6, 45, 47.4, 50.06, 50.27, 47.19, 46.08, 46.1, 48.2],
    'Rice Premium': [34.7, 35.63, 36.28, 37.95, 42.71, 43.19, 42.85, 43.4, 46.06, 43.35, 41.96, 41.79, 42.03, 45.43],
    'Well Milled Rice': [31.72, 32.01, 32.7, 34.5, 39.36, 38.31, 38.1, 38.91, 42.42, 38.8, 37.87, 37.7, 38.36, 42.95],
    'Regular Milled Rice': [28.55, 29.07, 30, 31.54, 36.45, 34.44, 34.26, 34.61, 38.54, 34.67, 33.87, 33.76, 35.05, 39.48]
}

# convert to pandas DataFrame
df = pd.DataFrame(data)
df.set_index('Year', inplace=True)

# perform ARIMA and plot results
def perform_arima_and_plot(rice_type):
    try:
        # ensure the data to be used for ARIMA model training is from 2010-2020 
        train = df[rice_type][:11]  # 2010-2020

        # use auto_arima to find the best (p, d, q) parameters
        auto_model = auto_arima(train, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
        print(f"Best ARIMA parameters for {rice_type}: {auto_model.order}")

        # fit the ARIMA model with the best parameters
        model = ARIMA(train, order=auto_model.order)
        model_fit = model.fit()

        # print the summary of the ARIMA model
        print(f"\nARIMA Model Summary for {rice_type}:\n")
        print(model_fit.summary())

        # forecast prices for 2021-2023
        forecast = model_fit.forecast(steps=3)  # 2021-2023
        forecast_years = np.arange(2021, 2024)

        # plotting
        plt.figure(figsize=(10, 6))

        # Plot training data (2010-2020)
        plt.plot(train.index, train, label=f'Training Data (2010-2020)', color='blue', marker='o')

        # Plot historical data (2010-2023), combine training data and actual historical data for this period
        historical_data = pd.concat([train, df[rice_type][11:]])  # Concatenate the forecasted historical data
        plt.plot(historical_data.index, historical_data, label=f'Historical Data (2010-2023)', color='green', linestyle='--')

        # Plot forecasted data (2021-2023)
        plt.plot(forecast_years, forecast, label=f'Forecast (2021-2023)', color='red', marker='o')

        # Add labels and title
        plt.xlabel('Year')
        plt.ylabel('Price')
        plt.title(f'{rice_type} Price Prediction (2010-2023)')

        # Add a legend
        plt.legend()

        # Display the plot
        plt.grid(True)
        plt.show()

        residuals = model_fit.resid

        # Mean Squared Error (MSE) on training data (2010-2020)
        mse = np.mean(residuals**2)
        print(f"Mean Squared Error (MSE) for {rice_type}: {mse}")

    except Exception as e:
        print(f"Error fitting ARIMA model for {rice_type}: {e}")

# apply ARIMA for each type of rice
for rice_type in df.columns:
    perform_arima_and_plot(rice_type)
