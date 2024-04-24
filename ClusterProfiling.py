import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv('C:/Users/amanf/OneDrive/Desktop/Cleaned_PSDH_Power_Supply_Evaluation.csv')
data['Date'] = pd.to_datetime(data['Date'])  # Ensure 'Date' is in datetime format
data['Restoration Time'] = pd.to_datetime(data['Restoration Time'])  # Convert if necessary
data['Response Time (Hours)'] = (data['Restoration Time'] - data['Date']).dt.total_seconds() / 3600

# Visualization setup
sns.set(style="whitegrid")

# Distribution of Outage Causes
plt.figure(figsize=(8, 6))
data['Cause'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange', 'green'])
plt.title('Distribution of Outage Causes')
plt.ylabel('')  # Hide the y-label as it's not necessary for pie charts

# Trends Over Time
plt.figure(figsize=(12, 6))
data.set_index('Date')['Number of Customers Affected'].resample('M').sum().plot()
plt.title('Trend of Number of Customers Affected Over Time')
plt.ylabel('Total Customers Affected')
plt.xlabel('Date')

# Outage Duration Distribution
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(data['Duration (Hours)'], bins=30, kde=True, color='blue')
plt.title('Histogram of Outage Durations')
plt.xlabel('Duration (Hours)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(x=data['Duration (Hours)'], color='green')
plt.title('Box Plot of Outage Durations')
plt.xlabel('Duration (Hours)')

# Cluster Variables Visualization
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='Duration (Hours)', y='Number of Customers Affected', hue='Cause', style='Cause', palette='deep')
plt.title('Scatter Plot of Duration vs. Number of Customers Affected')
plt.xlabel('Duration (Hours)')
plt.ylabel('Number of Customers Affected')

plt.show()
