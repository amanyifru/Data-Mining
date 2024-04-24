import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame with a DateTime index and you have a 'Duration (Hours)' column
# Load your data (make sure 'Date' is the index and in DateTime format)
data = pd.read_csv('C:/Users/amanf/OneDrive/Desktop/Cleaned_PSDH_Power_Supply_Evaluation.csv', index_col='Date', parse_dates=True)

# Make sure your time series data is at a consistent frequency
# If not, you may need to resample it:
# data = data.resample('D').mean()  # Example for daily resampling

# Perform seasonal decomposition
# The period parameter should be set according to the expected seasonality in your data,
# such as 12 for monthly data with yearly seasonality, 7 for daily data with weekly seasonality, etc.
decomp_result = seasonal_decompose(data['Duration (Hours)'], model='additive', period=12)

# Plot the decomposed time series
plt.rcParams['figure.figsize'] = [10, 6]  # Optional: changes the default figure size
decomp_result.plot()
plt.show()
