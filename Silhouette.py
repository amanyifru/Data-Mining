import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and prepare your data
# Adjust the file path to your dataset location
data = pd.read_csv('C:/Users/amanf/OneDrive/Desktop/Cleaned_PSDH_Power_Supply_Evaluation.csv')
# If necessary, perform any required data cleaning steps here

# Selecting features for clustering
# Replace these column names with the actual column names from your dataset
features = data[['Duration (Hours)', 'Number of Customers Affected', 'Response Time (Hours)']]
scaler = StandardScaler()

# Scaling the features
scaled_features = scaler.fit_transform(features)

# Calculating silhouette scores for different numbers of clusters
silhouette_scores = []
K_range = range(2, 11)  # Silhouette score is not defined for k=1

for K in K_range:
    kmeans = KMeans(n_clusters=K, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    silhouette_avg = silhouette_score(scaled_features, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Silhouette score for k = {K} is {silhouette_avg}")

# Plotting the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'bo-', color='blue', linewidth=2, markersize=8)
plt.title('Silhouette Method For Determining Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()
