import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot

# Load data
data = pd.read_csv('C:/Users/amanf/OneDrive/Desktop/Cleaned_PSDH_Power_Supply_Evaluation.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)

# Basic plot of Duration over time
data['Duration (Hours)'].plot(title='Power Outage Duration Over Time')
plt.ylabel('Duration in Hours')
plt.show()

# Seasonal Decomposition
result = seasonal_decompose(data['Duration (Hours)'], model='additive', period=365)
result.plot()
plt.show()

# Stationarity Test
def adf_test(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

adf_test(data['Duration (Hours)'])

# Autocorrelation to determine periodicity
autocorrelation_plot(data['Duration (Hours)'])
plt.show()

# SARIMA Model - adjust parameters based on your analysis
model = SARIMAX(data['Duration (Hours)'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=False)

# Forecast future values
forecast = model_fit.forecast(steps=12)
print(forecast)

# Plot the results
data['Duration (Hours)'].plot(legend=True, label='Actual', title='Forecast vs Actuals')
forecast.plot(legend=True, label='Forecast')
plt.show()
