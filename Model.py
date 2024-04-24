import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import kruskal

# Load data
data = pd.read_csv('C:/Users/amanf/OneDrive/Desktop/Cleaned_PSDH_Power_Supply_Evaluation.csv')

# Example: Visualize the distribution of 'Duration (Hours)' for each cluster
sns.boxplot(x='Cluster', y='Duration (Hours)', data=data)
plt.show()

# Example: Regression Analysis within Cluster 0
cluster_0 = data[data['Cluster'] == 0]
X = cluster_0[['Number of Customers Affected']]  # predictor
y = cluster_0['Duration (Hours)']  # response variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Performance metrics
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R²:", r2_score(y_test, y_pred))

# Example: Statistical Test
# Comparing 'Duration (Hours)' across clusters using Kruskal-Wallis Test
stat, p = kruskal(data[data['Cluster'] == 0]['Duration (Hours)'],
                  data[data['Cluster'] == 1]['Duration (Hours)'])

print('Statistics=%.3f, p=%.3f' % (stat, p))
