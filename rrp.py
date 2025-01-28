import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# input data
data = {
    "Year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "Rice Special": [45.6, 45, 47.4, 50.06, 50.27, 47.19, 46.08, 46.1, 48.2],
    "Rice Premium": [43.19, 42.85, 43.4, 46.06, 43.35, 41.96, 41.79, 42.03, 45.43],
    "Well Milled Rice": [38.31, 38.1, 38.91, 42.42, 38.8, 37.87, 37.7, 38.36, 42.95],
    "Regular Milled Rice": [34.44, 34.26, 34.61, 38.54, 34.67, 33.87, 33.76, 35.05, 39.48]
}

df = pd.DataFrame(data)
df.set_index("Year", inplace=True)

# function for calculating SMA and forecast
def calculate_sma_and_forecast(data, window=3, forecast_years=2):
    sma = data.rolling(window=window).mean()
    
    # initialize the forecast series with the last SMA value
    forecast_index = range(data.index[-1] + 1, data.index[-1] + 1 + forecast_years)
    forecast = pd.Series(index=forecast_index, dtype=float)
    
    # calculate the forecast iteratively
    for year in forecast_index:
        # get the last window years of SMA data
        last_sma_values = sma.iloc[-window:]
        # calculate the SMA for the forecasted year
        forecast_sma = last_sma_values.mean()
        # concatenate the forecasted SMA to the SMA series
        sma = pd.concat([sma, pd.Series([forecast_sma], index=[year])])
        # store the forecasted value
        forecast[year] = forecast_sma
    
    return sma

# forecast for 2024 and 2025
forecast_years = 2
sma_rice_special = calculate_sma_and_forecast(df["Rice Special"], forecast_years=forecast_years)
sma_rice_premium = calculate_sma_and_forecast(df["Rice Premium"], forecast_years=forecast_years)
sma_well_milled = calculate_sma_and_forecast(df["Well Milled Rice"], forecast_years=forecast_years)
sma_regular_milled = calculate_sma_and_forecast(df["Regular Milled Rice"], forecast_years=forecast_years)

# separate plots for each rice category
categories = ["Rice Special", "Rice Premium", "Well Milled Rice", "Regular Milled Rice"]
sma_data = {
    "Rice Special": sma_rice_special,
    "Rice Premium": sma_rice_premium,
    "Well Milled Rice": sma_well_milled,
    "Regular Milled Rice": sma_regular_milled
}

# plot each category
for category in categories:
    plt.figure(figsize=(8, 5))
    # historical data plot
    plt.plot(df.index, df[category], label= "Historical Data", marker="o")
    # SMA with forecast plot
    plt.plot(sma_data[category].index, sma_data[category], label="SMA with Forecast", linestyle="--", marker="o")
    
    # labels, title, and legend
    plt.title(f"{category} Prices Forecast (2015-2025) using 3-Year SMA", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()