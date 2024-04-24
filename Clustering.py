import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load your data
# Make sure to adjust the path to where you have your dataset stored
data = pd.read_csv('C:/Users/amanf/OneDrive/Desktop/Cleaned_PSDH_Power_Supply_Evaluation.csv')

# Select features for clustering
# Adjust these column names based on your actual dataset's column names
features = data[['Duration (Hours)', 'Number of Customers Affected', 'Response Time (Hours)']]

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA to reduce the data to 2 dimensions for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_features)

# Fit KMeans for 2 clusters
kmeans_k2 = KMeans(n_clusters=2, random_state=42)
clusters_k2 = kmeans_k2.fit_predict(reduced_data)

# Fit KMeans for 3 clusters
kmeans_k3 = KMeans(n_clusters=3, random_state=42)
clusters_k3 = kmeans_k3.fit_predict(reduced_data)

# Plotting the results side by side
plt.figure(figsize=(14, 7))

# Plot for k=2
plt.subplot(1, 2, 1)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters_k2, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(kmeans_k2.cluster_centers_[:, 0], kmeans_k2.cluster_centers_[:, 1], s=300, c='red', marker='x', label='Centroids')
plt.title('Clusters for k=2')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

# Plot for k=3
plt.subplot(1, 2, 2)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters_k3, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(kmeans_k3.cluster_centers_[:, 0], kmeans_k3.cluster_centers_[:, 1], s=300, c='red', marker='x', label='Centroids')
plt.title('Clusters for k=3')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

plt.tight_layout()
plt.show()
