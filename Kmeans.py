from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# Load and prepare your data
data = pd.read_csv('C:/Users/amanf/OneDrive/Desktop/Cleaned_PSDH_Power_Supply_Evaluation.csv')
features = data[['Duration (Hours)', 'Number of Customers Affected', 'Response Time (Hours)']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determining the optimal number of clusters using MiniBatchKMeans
inertia = []
for k in range(1, 11):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=100)  # Adjust batch_size as needed
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal k with MiniBatchKMeans')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
