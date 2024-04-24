import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np


# Load your dataset
data = pd.read_csv('C:/Users/amanf/OneDrive/Desktop/Cleaned_PSDH_Power_Supply_Evaluation.csv')  # Adjust the file path

# Check the data types to ensure we select only numeric columns for clustering
print(data.dtypes)

# Select features for clustering
features = data.select_dtypes(include=[np.number])  # This selects only numeric columns

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA to reduce the features to 2 components for easier visualization (optional)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)  # You can adjust the number of clusters based on your analysis
data['Cluster'] = kmeans.fit_predict(reduced_data)

# Now calculate and print the cluster characteristics for only numeric data
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()  # ensures we are only dealing with numeric columns
cluster_means = data[numeric_columns].groupby('Cluster').mean()

print("Cluster Means:")
print(cluster_means)

# If you still want to look at the distribution of a categorical variable like 'Cause', do this separately
if 'Cause' in data.columns:
    cause_distribution = data.groupby('Cluster')['Cause'].value_counts().unstack(fill_value=0)
    print("\nCause Distribution:")
    print(cause_distribution)
